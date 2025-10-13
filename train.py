import argparse
import os
import math
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

# --- Utils & Models ---
from utils.config import load_config
from utils.datasets import get_dataset
from utils import logger as logger_module
from utils.logger import Logger
from utils.quant_utils import alpha_regularization

from models.resnet_pact import resnet20_pact, resnet18_pact
from models.quant_layers import QuantConv2d, QuantLinear
import torch.nn.functional as F


# ---- Helper: inject torch into utils.logger if needed (save_checkpoint uses torch) ----
logger_module.torch = torch


# ---- Optional: ResNet50 (Bottleneck) with Quant layers for ImageNet ----
class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, bit_width_w=4, bit_width_a=4):
        super().__init__()
        self.conv1 = QuantConv2d(in_planes, planes, kernel_size=1, stride=1, padding=0,
                                 bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.conv2 = QuantConv2d(planes, planes, kernel_size=3, stride=stride, padding=1,
                                 bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.conv3 = QuantConv2d(planes, planes * self.expansion, kernel_size=1, stride=1, padding=0,
                                 bit_width_w=bit_width_w, bit_width_a=bit_width_a)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                QuantConv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, padding=0,
                            bit_width_w=bit_width_w, bit_width_a=bit_width_a),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + self.shortcut(x)
        return out

class ResNet50PACT(nn.Module):
    def __init__(self, num_classes=1000, bit_width_w=4, bit_width_a=4):
        super().__init__()
        self.in_planes = 64

        # Stem
        self.conv1 = QuantConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                 bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers (3, 4, 6, 3)
        self.layer1 = self._make_layer(Bottleneck, 64, 3, stride=1,
                                       bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2,
                                       bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2,
                                       bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2,
                                       bit_width_w=bit_width_w, bit_width_a=bit_width_a)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = QuantLinear(512 * Bottleneck.expansion, num_classes,
                              bit_width_w=bit_width_w, bit_width_a=bit_width_a)

    def _make_layer(self, block, planes, num_blocks, stride, bit_width_w, bit_width_a):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, bit_width_w, bit_width_a))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


# ---- Builders ----
def build_model(cfg):
    arch = cfg.model.architecture.lower()
    pact = cfg.model.pact
    bw_a = int(pact.bit_width)
    bw_w = int(pact.bit_width)  # par défaut : même bitwidth pour W et A
    num_classes = cfg.dataset.num_classes

    if arch == "resnet20":
        model = resnet20_pact(num_classes=num_classes, bit_width_w=bw_w, bit_width_a=bw_a)
    elif arch == "resnet18":
        model = resnet18_pact(num_classes=num_classes, bit_width_w=bw_w, bit_width_a=bw_a)
    elif arch == "resnet50":
        model = ResNet50PACT(num_classes=num_classes, bit_width_w=bw_w, bit_width_a=bw_a)
    else:
        raise ValueError(f"Architecture non supportée: {arch}")

    # NOTE: Pour respecter strictement "quantize_first_last: false", on pourrait remplacer
    # la première/dernière couche par des versions non quantifiées. Version simple ci-dessous :
    if hasattr(cfg.model.pact, "quantize_first_last") and not cfg.model.pact.quantize_first_last:
        # First conv -> déquantifier (remplacement simple : désactiver quant W en le mettant à 32 bits)
        if hasattr(model, "conv1"):
            model.conv1.bit_width_w = 32
            model.conv1.act.bit_width = None  # pas de quant sur activation de la 1ère couche

        # Last linear
        if hasattr(model, "linear"):
            model.linear.bit_width_w = 32
            model.linear.act.bit_width = None
        if hasattr(model, "fc"):
            model.fc.bit_width_w = 32
            model.fc.act.bit_width = None

    return model


def build_optimizer(cfg, model):
    opt_name = cfg.training.optimizer.lower()
    wd = float(cfg.training.weight_decay)
    lr = float(cfg.training.learning_rate.initial)

    if opt_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(cfg.training.momentum),
            weight_decay=wd,
            nesterov=False
        )
    elif opt_name == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )
    else:
        raise ValueError(f"Optimiseur non supporté: {opt_name}")
    return optimizer


def build_scheduler(cfg, optimizer):
    sched = cfg.training.learning_rate
    milestones = [int(m) for m in sched.schedule.milestones]
    gamma = float(sched.schedule.gamma)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    return scheduler


# ---- Metrics ----
def accuracy(output, target, topk=(1,)):
    """Compute the top-k accuracy for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k.mul_(100.0 / batch_size)).item())
        return res


# ---- Train & Eval Loops ----
def train_one_epoch(model, loader, criterion, optimizer, device, scaler, lambda_alpha, use_amp):
    model.train()
    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            outputs = model(images)
            ce_loss = criterion(outputs, targets)

            # L2 regularization over alpha parameters only
            alpha_reg = alpha_regularization(model)
            loss = ce_loss + lambda_alpha * alpha_reg

        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # metrics
        batch_size = images.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5 if outputs.size(1) >= 5 else 1))
        running_acc1 += acc1 * batch_size
        running_acc5 += acc5 * batch_size

    epoch_loss = running_loss / total
    epoch_acc1 = running_acc1 / total
    epoch_acc5 = running_acc5 / total
    return epoch_loss, epoch_acc1, epoch_acc5


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_acc1 = 0.0
    running_acc5 = 0.0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = images.size(0)
        total += batch_size
        running_loss += loss.item() * batch_size
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5 if outputs.size(1) >= 5 else 1))
        running_acc1 += acc1 * batch_size
        running_acc5 += acc5 * batch_size

    epoch_loss = running_loss / total
    epoch_acc1 = running_acc1 / total
    epoch_acc5 = running_acc5 / total
    return epoch_loss, epoch_acc1, epoch_acc5


def collect_alpha_stats(model):
    vals = []
    for n, p in model.named_parameters():
        if "alpha" in n:
            vals.append(p.detach().float().mean().item())
    if len(vals) == 0:
        return None
    return sum(vals) / len(vals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Chemin du fichier YAML de config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Device
    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # Datasets
    train_loader, val_loader = get_dataset(
        name=cfg.dataset.name,
        image_size=cfg.dataset.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers
    )

    # Model
    model = build_model(cfg).to(device)

    # Loss / Optim / Sched
    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    # Logger
    save_dir = cfg.logging.save_dir
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(save_dir)
    logger.log(str(cfg))

    # AMP
    use_amp = getattr(cfg.hardware, "fp16", False)
    from torch.amp import GradScaler as CudaGradScaler
    scaler = CudaGradScaler('cuda', enabled=use_amp)

    # Training params
    epochs = int(cfg.training.epochs)
    lambda_alpha = float(cfg.model.pact.lambda_alpha)

    best_acc1 = 0.0
    alpha_history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc1, train_acc5 = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, lambda_alpha, use_amp
        )
        val_loss, val_acc1, val_acc5 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        # Alpha tracking
        alpha_mean = collect_alpha_stats(model)
        if alpha_mean is not None and getattr(cfg.logging, "plot_alpha_evolution", True):
            alpha_history.append({"epoch": epoch, "alpha_mean": alpha_mean})

        # Logging
        dt = time.time() - t0
        logger.log(f"[{epoch:03d}/{epochs}] "
                   f"train_loss={train_loss:.4f} acc1={train_acc1:.2f} acc5={train_acc5:.2f} | "
                   f"val_loss={val_loss:.4f} acc1={val_acc1:.2f} acc5={val_acc5:.2f} | "
                   f"alpha_mean={alpha_mean if alpha_mean is not None else 'NA'} | "
                   f"{dt:.1f}s")
        logger.log_metrics(epoch, train_loss, val_loss, {"top1": val_acc1, "top5": val_acc5})

        # Checkpointing
        is_best = val_acc1 > best_acc1
        best_acc1 = max(best_acc1, val_acc1)
        # Sauvegarde manuelle pour éviter dépendance à utils.logger.save_checkpoint
        ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_acc1": best_acc1,
            "config": cfg.__dict__,
        }, ckpt_path)
        if is_best:
            best_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            logger.log(f"✅ New best (Top-1 {best_acc1:.2f}) — saved to {best_path}")

    # Sauvegarder l’évolution de alpha si dispo
    if len(alpha_history) > 0:
        import json
        with open(os.path.join(save_dir, "alpha_evolution.json"), "w") as f:
            json.dump(alpha_history, f, indent=2)

    logger.log("Training finished.")


if __name__ == "__main__":
    main()
