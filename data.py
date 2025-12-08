from typing import Tuple
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
SVHN_MEAN = (0.4377, 0.4438, 0.4728)
SVHN_STD = (0.1980, 0.2010, 0.1970)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def cifar10_loaders(data_root: str, batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int]:
    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 10


def svhn_loaders(data_root: str, batch_size: int, num_workers: int = 4) -> Tuple[DataLoader, DataLoader, int]:
    train_tf = T.Compose([
        T.Resize(40),
        T.RandomCrop(32),
        T.ToTensor(),
        T.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    test_tf = T.Compose([
        T.Resize(32),
        T.ToTensor(),
        T.Normalize(SVHN_MEAN, SVHN_STD),
    ])
    train_set = torchvision.datasets.SVHN(root=data_root, split='train', download=True, transform=train_tf)
    extra_set = torchvision.datasets.SVHN(root=data_root, split='extra', download=True, transform=train_tf)
    train_set = torch.utils.data.ConcatDataset([train_set, extra_set])
    test_set = torchvision.datasets.SVHN(root=data_root, split='test', download=True, transform=test_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader, 10


def imagenet_loaders(data_root: str, batch_size: int, num_workers: int = 8) -> Tuple[DataLoader, DataLoader, int]:
    train_tf = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_dir = f"{data_root}/train"
    val_dir = f"{data_root}/val"
    train_set = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = torchvision.datasets.ImageFolder(val_dir, transform=val_tf)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, 1000
