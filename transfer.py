import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
LOG_DIR = OUT_DIR / "logs"

from src.data.dataset import FlowerDataset, build_transforms, load_stats

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_pretrained_resnet18(num_classes=102,pretrained=True,freeze_backbone=False):

    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights = weights)
        print(f"已加载预训练权重: {weights}")

    else:
        model = resnet18(weights =None)
        print(f"未使用预训练权重（从零训练，作为对照组）")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    print(f"分类头已替换: Linear({in_features}, 1000) → Linear({in_features}, {num_classes})")

    if freeze_backbone:
        for name ,param in model.named_parameters():
            if not name.startswith("fc"):
                param.requires_grad = False
            n_frozen = sum(1 for p in model.parameters() if not p.requires_grad)
            n_total = sum(1 for p in model.parameters())
            print(f"已冻结主干: {n_frozen}/{n_total} 个参数张量不参与训练")

    return model


def build_optimizer( model , lr , freeze_backbone):

    if freeze_backbone:
        params = [p for p in model.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=lr ,weight_decay=1e-4)

    head_params ,backbone_params = [],[]
    for name ,param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("fc."):
            head_params.append(param)

        else:
            backbone_params.append(param)

    print(f"学习率分组: 主干 lr={lr * 0.1:.1e} ({len(backbone_params)} 组)  "
          f"分类头 lr={lr:.1e} ({len(head_params)} 组)")
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr * 0.1},
        {"params": head_params, "lr": lr},
    ], weight_decay=1e-4)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += bs
    return total_loss / n, correct / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += bs
    return total_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser(description="迁移学习对比实验")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="只训练分类头，冻结主干")
    ap.add_argument("--no-pretrained", action="store_true",
                    help="不用预训练权重（对照组）")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    if device.type == "cuda":
        print(f"显卡: {torch.cuda.get_device_name(0)}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 数据 ----------
    print("\n[1/3] 加载数据 ...")
    train_df = pd.read_csv(OUT_DIR / "train.csv", encoding="utf-8")
    val_df = pd.read_csv(OUT_DIR / "val.csv", encoding="utf-8")
    mean, std = load_stats()
    print(f"      训练 {len(train_df)} 张，验证 {len(val_df)} 张")

    train_tf, eval_tf = build_transforms(args.img_size, mean, std)
    train_loader = DataLoader(
        FlowerDataset(train_df, train_tf), batch_size=args.batch_size,
        shuffle=True, num_workers=args.workers, drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        FlowerDataset(val_df, eval_tf), batch_size=args.batch_size,
        shuffle=False, num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # ---------- 模型 ----------
    print("\n[2/3] 构建模型 ...")
    try:
        model = build_pretrained_resnet18(
            num_classes=102,
            pretrained=not args.no_pretrained,
            freeze_backbone=args.freeze_backbone,
        )
    except Exception as e:
        # 常见于：公司网络/校园网拦了 download.pytorch.org
        print(f"\n[×] 加载预训练权重失败: {type(e).__name__}: {e}")
        print("\n可能原因和解决办法:")
        print("  1) 网络访问不了 download.pytorch.org")
        print("     手动下载后放进缓存目录:")
        print("     https://download.pytorch.org/models/resnet18-f37072fd.pth")
        print("     放到: C:\\Users\\<你的用户名>\\.cache\\torch\\hub\\checkpoints\\")
        print("  2) 或者用 --no-pretrained 先跑通流程")
        return

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"      参数量 总计 {total:,}，其中可训练 {trainable:,}")

    # ---------- 优化器 / 损失 ----------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = build_optimizer(model, args.lr, args.freeze_backbone)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---------- 训练 ----------
    print(f"\n[3/3] 开始训练，共 {args.epochs} 轮\n")
    history = []
    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion,
                                          optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4), "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4), "val_acc": round(va_acc, 4),
        })

        mark = ""
        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_acc,
                "mean": mean, "std": std,
                "img_size": args.img_size,
                "num_classes": 102,
                "source": ("frozen" if args.freeze_backbone else
                           "scratch" if args.no_pretrained else "finetune"),
            }, CKPT_DIR / "transfer_best.pt")
            mark = "  ← 保存最优"

        print(f"Epoch {epoch:>2}/{args.epochs}  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  |  "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}{mark}")

    elapsed = (time.time() - t0) / 60
    print(f"\n训练完成，用时 {elapsed:.1f} 分钟")
    print(f"最佳验证准确率 {best_acc:.4f}")

    pd.DataFrame(history).to_csv(LOG_DIR / "transfer_log.csv",
                                 index=False, encoding="utf-8")
    print(f"训练日志 -> {LOG_DIR / 'transfer_log.csv'}")
    print(f"最优模型 -> {CKPT_DIR / 'transfer_best.pt'}")


if __name__ == "__main__":
    main()


