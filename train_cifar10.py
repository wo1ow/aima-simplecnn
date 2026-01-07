import argparse
import os
import random
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as T


# -------------------------
# Repro
# -------------------------
def set_seed(seed: int = 0):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


# -------------------------
# Simple CNN for CIFAR-10
# -------------------------
class CifarCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),  # /2
            )

        self.features = nn.Sequential(
            block(3, 64),    # 32 -> 16
            block(64, 128),  # 16 -> 8
            block(128, 256), # 8 -> 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    crit = nn.CrossEntropyLoss()
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = crit(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    crit = nn.CrossEntropyLoss()
    loss_sum = 0.0
    total = 0
    correct = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        total += x.size(0)
        correct += (logits.argmax(dim=1) == y).sum().item()

    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save", type=str, default="cifar10_cnn_best.pt")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)

    # CIFAR-10 标准均值方差（常用做法）
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)

    train_tf = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    test_tf = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=train_tf
    )
    test_set = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=test_tf
    )

    # 从训练集划一个验证集（5000 张）
    val_size = 5000
    train_size = len(train_set) - val_size
    train_set, val_set = torch.utils.data.random_split(
        train_set, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    # val 用 test_tf（不做增强）
    val_set.dataset.transform = test_tf

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_set, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test_set, batch_size=256, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    model = CifarCNN(num_classes=10).to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:02d}/{args.epochs} | lr {lr_now:.5f} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {"model": model.state_dict(), "val_acc": best_val_acc},
                args.save
            )
            print(f"  -> saved best to {args.save} (val_acc={best_val_acc:.4f})")

    # Test with best checkpoint
    ckpt = torch.load(args.save, map_location=device)
    model.load_state_dict(ckpt["model"])
    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"[BEST] test loss {test_loss:.4f} acc {test_acc:.4f}")

    # 打印类名
    print("Classes:", train_set.dataset.classes)


if __name__ == "__main__":
    main()
