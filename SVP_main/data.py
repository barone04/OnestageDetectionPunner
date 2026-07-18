import os
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from utils import RandomMixup, RandomCutmix


def get_transforms():
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.TrivialAugmentWide(),
            transforms.ToTensor(),
            normalize,
            transforms.Resize(32, antialias=True),
            transforms.RandomErasing(0.1),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            normalize,
            transforms.Resize(32, antialias=True),
        ]
    )

    return train_transform, test_transform


def get_mixupcutmix(mixup_alpha, cutmix_alpha, num_classes):
    return torchvision.transforms.RandomChoice(
        [
            RandomMixup(num_classes, p=1.0, alpha=mixup_alpha),
            RandomCutmix(num_classes, p=1.0, alpha=cutmix_alpha),
        ]
    )


def get_dataloaders(num_classes, dataset_path, mixup_alpha, cutmix_alpha, batch_size):
    if num_classes == 10:
        tvdset = torchvision.datasets.CIFAR10
    elif num_classes == 100:
        tvdset = torchvision.datasets.CIFAR100

    train_transform, test_transform = get_transforms()

    train_dataset = tvdset(
        root=dataset_path,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = tvdset(
        root=dataset_path,
        train=False,
        download=True,
        transform=test_transform,
    )

    mixupcutmix = get_mixupcutmix(mixup_alpha, cutmix_alpha, num_classes)

    def collate_fn(batch):
        return mixupcutmix(*default_collate(batch))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(10, os.cpu_count()),
        drop_last=True,
        pin_memory=True,
        collate_fn=collate_fn,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(10, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
    )

    return train_loader, test_loader
