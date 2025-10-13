import torch
import torchvision
import torchvision.transforms as transforms

def get_dataset(name, image_size=32, batch_size=128, num_workers=4):
    """
    Retourne les DataLoader d'entraînement et de validation pour CIFAR10, SVHN, ou ImageNet.
    """
    name = name.lower()

    if name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                                download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                               download=True, transform=transform_test)

    elif name == "svhn":
        transform = transforms.Compose([
            transforms.Resize(40),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970)),
        ])
        trainset = torchvision.datasets.SVHN(root="./data", split="train",
                                             download=True, transform=transform)
        testset = torchvision.datasets.SVHN(root="./data", split="test",
                                            download=True, transform=transform)

    elif name == "imagenet":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        transform_val = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        trainset = torchvision.datasets.ImageNet(root="./data/imagenet",
                                                 split="train", transform=transform_train)
        testset = torchvision.datasets.ImageNet(root="./data/imagenet",
                                                split="val", transform=transform_val)

    else:
        raise ValueError(f"Dataset {name} non supporté")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return trainloader, testloader
