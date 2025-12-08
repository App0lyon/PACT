from __future__ import annotations

import argparse
import os

import torch
from torch import nn, optim

import models
from data import cifar10_loaders, svhn_loaders, imagenet_loaders
from train_utils import train_one_epoch, evaluate
from pact import PACTActivation


def parse_args():
    parser = argparse.ArgumentParser(description="Full-precision baselines for PACT paper")
    parser.add_argument("--dataset", choices=["cifar10", "svhn", "imagenet"], required=True)
    parser.add_argument("--model", choices=["resnet20", "svhn_convnet", "alexnet_bn", "resnet18_preact", "resnet50_preact"], help="Model name")
    parser.add_argument("--data-root", default="data", help="Dataset root")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--lr-milestones", type=int, nargs="*")
    parser.add_argument("--lr-gamma", type=float)
    parser.add_argument("--lr-step", type=int, help="Step size for StepLR (used for SVHN/Adam)")
    parser.add_argument("--lambda-alpha", type=float, help="L2 regularization strength for PACT alpha parameters (defaults to weight decay).")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--output-dir", default=None, help="Optional directory to save checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--act-bits", type=int, default=4, help="Activation quantization bits for PACT (set <=0 to disable quantization).")
    parser.add_argument("--w-bits", type=int, default=4, help="Weight quantization bits (set <=0 to disable quantization).")
    parser.add_argument("--quantize-first-last", action="store_true", help="Also quantize activations of the first and last layers.")
    parser.add_argument("--quant-warmup-epochs", type=int, default=0, help="Number of initial epochs to train with PACT clipping but without quantization.")
    parser.add_argument("--log-interval", type=int, default=50)
    return parser.parse_args()


def set_defaults(args):
    if args.dataset == "cifar10":
        args.model = args.model or "resnet20"
        args.epochs = args.epochs or 200
        args.batch_size = args.batch_size or 128
        args.lr = args.lr or 0.1
        args.momentum = args.momentum or 0.9
        args.weight_decay = args.weight_decay or 2e-4
        args.lr_milestones = args.lr_milestones or [60, 120]
        args.lr_gamma = args.lr_gamma or 0.1
        args.workers = args.workers or 4
    elif args.dataset == "svhn":
        args.model = args.model or "svhn_convnet"
        args.epochs = args.epochs or 200
        args.batch_size = args.batch_size or 128
        args.lr = args.lr or 1e-3
        args.weight_decay = args.weight_decay or 1e-7
        args.lr_step = args.lr_step or 50
        args.lr_gamma = args.lr_gamma or 0.5
        args.workers = args.workers or 4
    elif args.dataset == "imagenet":
        if args.model is None:
            raise ValueError("--model required for ImageNet")
        if args.model == "alexnet_bn":
            args.epochs = args.epochs or 100
            args.batch_size = args.batch_size or 128
            args.lr = args.lr or 1e-4
            args.weight_decay = args.weight_decay or 5e-6
            args.lr_milestones = args.lr_milestones or [56, 64]
            args.lr_gamma = args.lr_gamma or 0.2
        else:
            args.epochs = args.epochs or 110
            args.batch_size = args.batch_size or 256
            args.lr = args.lr or 0.1
            args.momentum = args.momentum or 0.9
            args.weight_decay = args.weight_decay or 1e-4
            args.lr_milestones = args.lr_milestones or [30, 60, 85, 95]
            args.lr_gamma = args.lr_gamma or 0.1
        args.workers = args.workers or 8
    return args


def get_alpha_parameters(model: nn.Module):
    """Collect all PACT alpha parameters (one per activation layer)."""
    alphas = []
    for module in model.modules():
        if isinstance(module, PACTActivation):
            alphas.append(module.alpha)
    return alphas


def set_quantization_enabled(model: nn.Module, enabled: bool):
    """Toggle quantization for activations and weights while preserving clipping/FP params."""
    for module in model.modules():
        if isinstance(module, PACTActivation):
            module.num_bits = module.base_num_bits if enabled else None
        if hasattr(module, "base_w_bits"):
            module.w_bits = module.base_w_bits if enabled else None


def build_model(name: str, num_classes: int, act_bits: int | None, w_bits: int | None, quantize_first_last: bool):
    if name == "resnet20":
        return models.resnet20(num_classes=num_classes, act_bits=act_bits, w_bits=w_bits, quantize_first_last=quantize_first_last)
    if name == "svhn_convnet":
        return models.svhn_convnet(num_classes=num_classes, act_bits=act_bits, w_bits=w_bits, quantize_first_last=quantize_first_last)
    if name == "alexnet_bn":
        return models.alexnet_bn(num_classes=num_classes, act_bits=act_bits, w_bits=w_bits, quantize_first_last=quantize_first_last)
    if name == "resnet18_preact":
        return models.preact_resnet18(num_classes=num_classes, act_bits=act_bits, w_bits=w_bits, quantize_first_last=quantize_first_last)
    if name == "resnet50_preact":
        return models.preact_resnet50(num_classes=num_classes, act_bits=act_bits, w_bits=w_bits, quantize_first_last=quantize_first_last)
    raise ValueError(f"Unknown model {name}")


def get_loaders(args):
    if args.dataset == "cifar10":
        return cifar10_loaders(args.data_root, args.batch_size, args.workers)
    if args.dataset == "svhn":
        return svhn_loaders(args.data_root, args.batch_size, args.workers)
    if args.dataset == "imagenet":
        return imagenet_loaders(args.data_root, args.batch_size, args.workers)
    raise ValueError(f"Unknown dataset {args.dataset}")


def make_optimizer(args, model, alpha_params):
    alpha_ids = {id(p) for p in alpha_params}
    main_params = [p for p in model.parameters() if id(p) not in alpha_ids]
    param_groups = [
        {"params": main_params, "weight_decay": args.weight_decay},
        {"params": alpha_params, "weight_decay": 0.0},
    ]
    if args.dataset == "svhn" or args.model == "alexnet_bn":
        return optim.Adam(param_groups, lr=args.lr, eps=1e-5)
    return optim.SGD(param_groups, lr=args.lr, momentum=args.momentum or 0.9, weight_decay=0.0, nesterov=False)


def make_scheduler(args, optimizer):
    if args.dataset == "svhn":
        return optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma)


def save_checkpoint(state, is_best: bool, output_dir: str, filename: str):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    torch.save(state, path)
    if is_best:
        best_path = os.path.join(output_dir, "model_best.pth")
        torch.save(state, best_path)


def main():
    args = parse_args()
    args = set_defaults(args)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    act_bits = None if args.act_bits is None or args.act_bits <= 0 else args.act_bits
    w_bits = None if args.w_bits is None or args.w_bits <= 0 else args.w_bits

    train_loader, val_loader, num_classes = get_loaders(args)
    model = build_model(
        args.model,
        num_classes=num_classes,
        act_bits=act_bits,
        w_bits=w_bits,
        quantize_first_last=args.quantize_first_last,
    ).to(device)
    alpha_params = get_alpha_parameters(model)
    lambda_alpha = args.lambda_alpha if args.lambda_alpha is not None else args.weight_decay
    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(args, model, alpha_params)
    scheduler = make_scheduler(args, optimizer)

    best_top1 = 0.0
    for epoch in range(args.epochs):
        quant_enabled = epoch >= args.quant_warmup_epochs
        set_quantization_enabled(model, enabled=quant_enabled)
        train_stats = train_one_epoch(
            model, criterion, optimizer, train_loader, device, lambda_alpha=lambda_alpha, alpha_params=alpha_params
        )
        scheduler.step()
        val_stats = evaluate(model, criterion, val_loader, device)

        best_top1 = max(best_top1, val_stats["top1"])
        alpha_vals = [a.detach().item() for a in alpha_params] if alpha_params else []
        alpha_mean = sum(alpha_vals) / len(alpha_vals) if alpha_vals else 0.0
        alpha_max = max(alpha_vals) if alpha_vals else 0.0
        print(
            f"Epoch {epoch+1:03d}/{args.epochs} | "
            f"Train loss {train_stats['loss']:.4f} top1 {train_stats['top1']:.2f} top5 {train_stats['top5']:.2f} | "
            f"Val loss {val_stats['loss']:.4f} top1 {val_stats['top1']:.2f} top5 {val_stats['top5']:.2f} | "
            f"Best top1 {best_top1:.2f} | alpha mean {alpha_mean:.4f} max {alpha_max:.4f}"
        )

        if args.output_dir and ((epoch + 1) % 5 == 0 or val_stats["top1"] >= best_top1):
            checkpoint = {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_top1": best_top1,
                "args": vars(args),
            }
            save_checkpoint(checkpoint, val_stats["top1"] >= best_top1, args.output_dir, f"checkpoint_{epoch+1:03d}.pth")


if __name__ == "__main__":
    main()
