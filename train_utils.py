from __future__ import annotations
import torch
from torch import nn
from typing import Tuple, Iterable, Optional


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += val * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train_one_epoch(
    model: nn.Module,
    criterion,
    optimizer,
    dataloader,
    device: torch.device,
    lambda_alpha: float = 0.0,
    alpha_params: Optional[Iterable[torch.Tensor]] = None,
):
    model.train()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)
        if lambda_alpha > 0.0 and alpha_params is not None:
            reg = torch.stack([a.pow(2) for a in alpha_params]).sum()
            loss = loss + lambda_alpha * reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        loss_meter.update(loss.item(), images.size(0))
        top1_meter.update(acc1.item(), images.size(0))
        top5_meter.update(acc5.item(), images.size(0))

    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
    }


def evaluate(model: nn.Module, criterion, dataloader, device: torch.device):
    model.eval()
    loss_meter = AverageMeter()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, targets)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            loss_meter.update(loss.item(), images.size(0))
            top1_meter.update(acc1.item(), images.size(0))
            top5_meter.update(acc5.item(), images.size(0))

    return {
        "loss": loss_meter.avg,
        "top1": top1_meter.avg,
        "top5": top5_meter.avg,
    }
