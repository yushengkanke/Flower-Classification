import argparse
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "outputs"
CKPT_DIR = OUT_DIR / "checkpoints"
LOG_DIR = OUT_DIR / "logs"


from src.data.dataset import FlowerDataset, build_transforms, load_stats
from src.models.resnet import build_resnet18

def set_seed(seed =42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("[!] CUDA 不可用，改用 CPU 训练（会很慢）")
    return torch.device("cpu")

def train_one_epoch(model,loader,criterion,optimizer,device,epoch,total_epochs):
    model.train()
    total_loss,correct,n = 0.0,0,0
    for imgs,labels in loader:
        imgs = imgs.to(device,non_blocking=True)
        labels = labels.to(device,non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits,labels)

        loss.backward()
        optimizer.step()
        bs = labels.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(dim=1) == labels).sum().item()
        n += bs
    return total_loss / n,correct / n

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
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=0,
                    help="Windows 上建议 0，调大需要 __main__ 保护（我们已有）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()


    set_seed(args.seed)
    device = get_device()
    print(f"设备       : {device}")
    if device.type == "cuda":
        print(f"显卡       : {torch.cuda.get_device_name(0)}")

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


    print("\n[1/3] 加载数据 ...")
    train_df = pd.read_csv(OUT_DIR / "train.csv", encoding="utf-8")
    val_df = pd.read_csv(OUT_DIR / "val.csv", encoding="utf-8")
    mean, std = load_stats()
    print(f"      训练集 {len(train_df)} 张   验证集 {len(val_df)} 张")
    print(f"      归一化 mean={mean} std={std}")

    train_tf, eval_tf = build_transforms(args.img_size, mean, std)

    train_loader = DataLoader(
        FlowerDataset(train_df, train_tf),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        FlowerDataset(val_df, eval_tf),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    print(f"      训练 {len(train_loader)} 个 batch/轮")


    print("\n[2/3] 构建模型 ...")
    model = build_resnet18(num_classes=102).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      手写 ResNet-18  参数量 {n_params:,}")



    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    print(f"\n[3/3] 开始训练，共 {args.epochs} 轮\n")
    history = []
    best_val_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, args.epochs
        )
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        history.append({
            "epoch": epoch,
            "train_loss": round(tr_loss, 4),
            "train_acc": round(tr_acc, 4),
            "val_loss": round(va_loss, 4),
            "val_acc": round(va_acc, 4),
            "lr": optimizer.param_groups[0]["lr"],
        })

        mark = ""
        if va_acc > best_val_acc:
            best_val_acc = va_acc

            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch,
                "best_val_acc": best_val_acc,
                "mean": mean,
                "std": std,
                "img_size": args.img_size,
                "num_classes": 102,
            }, CKPT_DIR / "best.pt")
            mark = "  ← 保存最优"

        print(f"Epoch {epoch:>3}/{args.epochs}  "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  |  "
              f"val loss {va_loss:.4f} acc {va_acc:.4f}{mark}")


    elapsed = (time.time() - t0) / 60
    print(f"\n训练完成，用时 {elapsed:.1f} 分钟")
    print(f"最佳验证准确率 {best_val_acc:.4f}（随机猜是 {1/102:.4f}）")

    log_df = pd.DataFrame(history)
    log_path = LOG_DIR / "train_log.csv"
    log_df.to_csv(log_path, index=False, encoding="utf-8")
    print(f"训练日志已保存 -> {log_path}")
    print(f"最优模型已保存 -> {CKPT_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()