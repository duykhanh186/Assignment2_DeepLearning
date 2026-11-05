CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def get_cifar10_loaders(
    batch_size=128,
    root="./data",
    num_workers=0,
    seed=42,
    val_size=5000,
):
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    full_train = datasets.CIFAR10(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.CIFAR10(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    n_train = len(full_train) - val_size

    split_gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train,
        [n_train, val_size],
        generator=split_gen,
    )

    loader_gen = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_gen,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader


def get_fashion_mnist_loaders(
    batch_size=128,
    root="./data",
    num_workers=0,
    seed=42,
    val_size=5000,
):
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader, random_split

    transform = transforms.ToTensor()

    full_train = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=transform,
    )

    test_set = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=transform,
    )

    n_train = len(full_train) - val_size

    split_gen = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(
        full_train,
        [n_train, val_size],
        generator=split_gen,
    )

    loader_gen = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=loader_gen,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader
