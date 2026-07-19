"""CIFAR-10 loaders matching the original NORTON experiment."""
import torch
from torchvision import datasets, transforms


def make_cifar10_loaders(data_path, batch_size=256, workers=2,
                         download=True, pin_memory=False):
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.4914, 0.4822, 0.4465),
            (0.2023, 0.1994, 0.2010),
        ),
    ])

    train_set = datasets.CIFAR10(
        root=data_path, train=True, download=download, transform=train_transform
    )
    val_set = datasets.CIFAR10(
        root=data_path, train=False, download=download, transform=val_transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=workers,
        pin_memory=pin_memory,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
