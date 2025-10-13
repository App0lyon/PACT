import argparse
import json
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Utils ----
from utils.config import load_config
from utils.datasets import get_dataset

# ---- Models (imports) ----
from models.resnet_pact import resnet20_pact, resnet18_pact
from models.quant_layers import QuantConv2d, QuantLinear


# ---------- (self-contained) ResNet50 PACT (identique à train.py) ----------
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
        self.conv1 = QuantConv2d(3, 64, kernel_size=7, stride=2, padding=3,
                                 bit_width_w=bit_width_w, bit_width_a=bit_width_a)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
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
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
# --------------------------------------------------------------------------


def build_model(cfg):
    arch = cfg.model.architecture.lower()
    pact = cfg.model.pact
    bw_a = int(pact.bit_width)
    bw_w = int(pact.bit_width)
    num_classes = cfg.dataset.num_classes

    if arch == "resnet20":
        model = resnet20_pact(num_classes=num_classes, bit_width_w=bw_w, bit_width_a=bw_a)
    elif arch == "resnet18":
        model = resnet18_pact(num_classes=num_classes, bit_width_w=bw_w, bit_width_a=bw_a)
    elif arch == "resnet50":
        model = ResNet50PACT(num_classes=num_classes, bit_width_w=bw_w, bit_width_a=bw_a)
    else:
        raise ValueError(f"Architecture non supportée: {arch}")

    # Respecter quantize_first_last: false
    if hasattr(cfg.model.pact, "quantize_first_last") and not cfg.model.pact.quantize_first_last:
        if hasattr(model, "conv1"):
            model.conv1.bit_width_w = 32
            model.conv1.act.bit_width = None
        if hasattr(model, "linear"):
            model.linear.bit_width_w = 32
            model.linear.act.bit_width = None
        if hasattr(model, "fc"):
            model.fc.bit_width_w = 32
            model.fc.act.bit_width = None

    return model


@torch.no_grad()
def compute_confusion_matrix(model, loader, device, num_classes):
    import numpy as np
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        preds = outputs.argmax(1)
        for t, p in zip(targets.view(-1), preds.view(-1)):
            cm[t.long().item(), p.long().item()] += 1
    return cm


@torch.no_grad()
def evaluate(model, loader, device, dump_preds=False, save_dir=None):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total = 0
    correct1 = 0
    correct5 = 0
    preds_dump = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = crit(outputs, targets)

        total_loss += loss.item()
        total += images.size(0)

        # Top-1/Top-5
        maxk = 5 if outputs.size(1) >= 5 else 1
        _, pred = outputs.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct1 += correct[:1].reshape(-1).float().sum(0).item()
        if maxk >= 5:
            correct5 += correct[:5].reshape(-1).float().sum(0).item()
        else:
            correct5 += correct1  # fallback

        if dump_preds:
            probs = F.softmax(outputs, dim=1).detach().cpu()
            preds = pred[0].detach().cpu()
            t_cpu = targets.detach().cpu()
            for i in range(images.size(0)):
                preds_dump.append({
                    "target": int(t_cpu[i]),
                    "pred_top1": int(preds[i]),
                    "prob_top1": float(probs[i, preds[i]]),
                })

    avg_loss = total_loss / total
    top1 = 100.0 * correct1 / total
    top5 = 100.0 * correct5 / total

    if dump_preds and save_dir is not None:
        with open(os.path.join(save_dir, "predictions.jsonl"), "w") as f:
            for row in preds_dump:
                f.write(json.dumps(row) + "\n")

    return {"loss": avg_loss, "top1": top1, "top5": top5, "n": total}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Chemin du fichier YAML")
    parser.add_argument("--checkpoint", type=str, required=True, help="Chemin du .pth")
    parser.add_argument("--confusion", action="store_true", help="Générer la matrice de confusion")
    parser.add_argument("--dump_preds", action="store_true", help="Sauver predictions.jsonl")
    args = parser.parse_args()

    cfg = load_config(args.config)

    device = torch.device(cfg.hardware.device if torch.cuda.is_available() else "cpu")

    # Dataloader (utilise split=val/test selon dataset)
    _, val_loader = get_dataset(
        name=cfg.dataset.name,
        image_size=cfg.dataset.image_size,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.hardware.num_workers
    )

    # Build model et charger checkpoint
    model = build_model(cfg).to(device)

    # Supporter checkpoints "best_model.pth" (state_dict) ou "checkpoint_epochX.pth" (dict)
    ckpt = torch.load(args.checkpoint, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)

    model.eval()

    # Dossier de sortie
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(cfg.logging.save_dir, f"eval_{ts}")
    os.makedirs(save_dir, exist_ok=True)

    # Évaluation
    t0 = time.time()
    metrics = evaluate(model, val_loader, device, dump_preds=args.dump_preds, save_dir=save_dir)
    dt = time.time() - t0

    # Sauvegarde des métriques
    report = {
        "config": {
            "experiment": cfg.experiment_name,
            "arch": cfg.model.architecture,
            "bit_width": int(cfg.model.pact.bit_width),
            "quantize_first_last": bool(getattr(cfg.model.pact, "quantize_first_last", False)),
            "dataset": cfg.dataset.name,
        },
        "metrics": metrics,
        "elapsed_sec": dt,
        "checkpoint": os.path.abspath(args.checkpoint),
    }
    with open(os.path.join(save_dir, "eval_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n===== Evaluation Report =====")
    print(json.dumps(report, indent=2))

    # Matrice de confusion
    if args.confusion:
        import numpy as np
        try:
            import matplotlib.pyplot as plt
            cm = compute_confusion_matrix(model, val_loader, device, num_classes=cfg.dataset.num_classes)
            fig = plt.figure(figsize=(6, 5))
            plt.imshow(cm, interpolation="nearest")
            plt.title("Confusion Matrix")
            plt.colorbar()
            plt.tight_layout()
            plt.xlabel("Predicted")
            plt.ylabel("True")
            out_path = os.path.join(save_dir, "confusion_matrix.png")
            fig.savefig(out_path, dpi=160, bbox_inches="tight")
            print(f"Confusion matrix saved to: {out_path}")
        except Exception as e:
            # Fallback : sauver la matrice brute en .npy
            cm = compute_confusion_matrix(model, val_loader, device, num_classes=cfg.dataset.num_classes)
            npy_path = os.path.join(save_dir, "confusion_matrix.npy")
            import numpy as np
            np.save(npy_path, cm)
            print(f"Matplotlib indisponible — matrice de confusion sauvegardée en binaire: {npy_path}")


if __name__ == "__main__":
    main()
