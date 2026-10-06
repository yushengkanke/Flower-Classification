import json
import sys
from pathlib import Path

import argparse
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
sys.stdout.reconfigure(encoding='utf-8')

matplotlib.use("Agg")
import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
from src.data.dataset import FlowerDataset, build_transforms
from src.models.resnet import build_resnet18

def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

@torch.no_grad()
def run_inference(model,loader,device):

    model.eval()
    all_preds,all_labels = [],[]
    all_top5,all_probs = [],[]

    for imgs,labels in loader:
        imgs  = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits,dim=1)

        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels)

        all_top5.append(logits.topk(5,dim=1).indices.cpu())
        all_probs.append(probs.max(dim=1).values.cpu())

    return (
        torch.cat(all_preds).numpy(),
        torch.cat(all_labels).numpy(),
        torch.cat(all_top5).numpy(),
        torch.cat(all_probs).numpy(),
    )

def plot_confusion(cm_norm, acc, save_dir=FIG_DIR, tick_step=10):

    save_dir.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()

    n = cm_norm.shape[0]
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)


    ticks = list(range(0, n, tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks], fontsize=7, rotation=90)
    ax.set_yticklabels([str(t) for t in ticks], fontsize=7)

    ax.set_xlabel("预测类别编号 →（每格 = 该类被判成该列编号的比例）")
    ax.set_ylabel("真实类别编号 ↓（每行加起来 = 1）")
    ax.set_title(f"测试集混淆矩阵（102 类）  整体 Top-1 准确率 {acc:.4f}\n"
                 f"对角线越白 = 该类认得越准；某行暗 = 该类学得差\n"
                 f"花名对照见 outputs/logs/per_class_accuracy.csv 与'易混淆类别对'图")
    fig.colorbar(im, ax=ax, fraction=0.046, label="占比")
    fig.tight_layout()
    path = save_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path
def plot_per_class(per_class_acc,save_dir = FIG_DIR):
    save_dir.mkdir(parents=True,exist_ok=True)
    setup_chinese_font()
    s = pd.Series(per_class_acc).sort_values()
    fig,ax = plt.subplots(figsize = (14,5))
    ax.bar(range(len(s)),s.values,width=1.0,color="#4C72B0")
    ax.axhline(s.mean(),color="red",ls = "--",lw=1.5,label=f"平均 {s.mean():.3f}")
    ax.set_xlabel("类别（按准确率从低到高排序）")
    ax.set_ylabel("该类准确率")
    ax.set_title(f"每类准确率 — 最低 {s.min():.2f}，最高 {s.max():.2f}，"
                 f"有 {int((s == 0).sum())} 个类完全认不出")
    ax.legend()
    fig.tight_layout()
    path = save_dir / "per_class_acc.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path

def plot_predictions(model, test_df, device, mean, std, img_size,
                     n_correct=3, n_wrong=3, save_dir=FIG_DIR):

    from PIL import Image

    save_dir.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()
    _, eval_tf = build_transforms(img_size, mean, std)


    records = []
    model.eval()
    for _, row in test_df.iterrows():
        with Image.open(row["path"]) as im:
            rgb = im.convert("RGB")
            x = eval_tf(rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
        pred = int(probs.argmax())
        records.append({
            "path": row["path"], "true": int(row["label"]),
            "true_name": row["name"], "pred": pred,
            "pred_name": test_df[test_df["label"] == pred]["name"].iloc[0]
                          if (test_df["label"] == pred).any() else str(pred),
            "conf": float(probs[pred]),
            "correct": pred == int(row["label"]),
        })

    rec_df = pd.DataFrame(records)
    correct_df = rec_df[rec_df["correct"]].nlargest(n_correct, "conf")
    wrong_df = rec_df[~rec_df["correct"]].nlargest(n_wrong, "conf")
    show = pd.concat([correct_df, wrong_df])

    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.5))
    for ax, (_, r) in zip(axes.ravel(), show.iterrows()):
        with Image.open(r["path"]) as im:
            ax.imshow(im.convert("RGB"))
        ax.axis("off")
        mark = "[对]" if r["correct"] else "[错]"
        color = "green" if r["correct"] else "red"
        ax.set_title(f"{mark} 预测: {r['pred_name'][:18]} ({r['conf']:.1%})\n"
                     f"真实: {r['true_name'][:18]}",
                     fontsize=9, color=color)
    fig.suptitle("预测示例（上排：有把握且猜对的；下排：有把握但猜错的）", fontsize=13)
    fig.tight_layout()
    path = save_dir / "predictions.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path, rec_df

def plot_confusion_pairs(cm, names, top_k=25, save_dir=FIG_DIR):

    save_dir.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()

    n = cm.shape[0]

    row_sum = np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cm_ratio = cm / row_sum


    pairs = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if cm[i, j] > 0:
                pairs.append({
                    "true_id": i,
                    "true_name": names[i],
                    "pred_id": j,
                    "pred_name": names[j],
                    "count": int(cm[i, j]),
                    "ratio": float(cm_ratio[i, j]),
                })

    pairs_df = pd.DataFrame(pairs).sort_values("ratio", ascending=False)


    csv_path = LOG_DIR / "confusion_pairs.csv"
    pairs_df.to_csv(csv_path, index=False, encoding="utf-8")


    top = pairs_df.head(top_k).iloc[::-1]
    labels = [
        f"{r.true_id:>3} {r.true_name[:22]}\n  → {r.pred_id:>3} {r.pred_name[:22]}"
        for r in top.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(11, max(6, top_k * 0.42)))
    bars = ax.barh(range(len(top)), top["ratio"].values, color="#C44E52")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("误判比例（真实的该类中，有多大比例被判成了下面这一类）")
    ax.set_title(f"最容易混淆的 {top_k} 组类别（真实 → 误判为）\n"
                 f"编号对应混淆矩阵的坐标轴")

    for b, r in zip(bars, top.itertuples()):
        ax.text(b.get_width() + 0.003, b.get_y() + b.get_height() / 2,
                f"{r.count}张", va="center", fontsize=7)
    ax.set_xlim(0, min(1.0, top["ratio"].max() * 1.18))
    fig.tight_layout()
    path = save_dir / "confusion_pairs.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path, pairs_df, csv_path


def main():
    ap = argparse.ArgumentParser(description="在测试集上评估模型")
    ap.add_argument("--ckpt", type=str, default="best.pt",
                    help="checkpoint 文件名。best.pt=手写模型；transfer_best.pt=迁移学习")
    ap.add_argument("--model", type=str, default="auto",
                    choices=["auto", "custom", "torchvision"],
                    help="模型类型，auto 会从 checkpoint 的键名自动判断")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # ---------- 检查模型文件 ----------
    ckpt_path = OUT_DIR / "checkpoints" / args.ckpt
    if not ckpt_path.exists():
        print(f"[×] 找不到 {ckpt_path}，请先运行 python train.py 或 python transfer.py")
        return

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    num_classes = ckpt["num_classes"]
    img_size = ckpt["img_size"]
    mean, std = ckpt["mean"], ckpt["std"]

    print(f"已加载 {ckpt_path.name}")
    print(f"  训练到第 {ckpt['epoch']} 轮，验证集最优 acc = {ckpt['best_val_acc']:.4f}")
    print(f"  归一化 mean={mean} std={std}  img_size={img_size}")

    # ---------- ★ 判断用哪个模型类来重建网络 ----------
    # 手写 ResNet 的第一层叫 stem；torchvision 的叫 conv1。
    # 所以看一眼 state_dict 的键名就知道是哪个实现。
    state_keys = list(ckpt["model_state"].keys())
    auto_is_custom = any(k.startswith("stem.") for k in state_keys)

    if args.model == "auto":
        use_custom = auto_is_custom
    else:
        use_custom = (args.model == "custom")

    print(f"  模型类型: {'手写 ResNet (residual.py + resnet.py)' if use_custom else 'torchvision ResNet-18'}")

    if use_custom:
        model = build_resnet18(num_classes=num_classes).to(device)
    else:
        from torchvision.models import resnet18
        model = resnet18(weights=None)               # 结构即可，权重马上会被覆盖
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

    # ---------- 加载权重 ----------
    # ★ 用 strict=False 更宽容，但要检查"真正缺失的是什么"
    #   我们只允许 num_batches_tracked 这类缓冲区缺失，不允许卷积/BN 权重缺失
    missing, unexpected = model.load_state_dict(ckpt["model_state"], strict=False)

    real_missing = [k for k in missing if "num_batches_tracked" not in k]
    if real_missing:
        print(f"\n[×] 权重没有完全加载，缺失 {len(real_missing)} 个关键参数:")
        for k in real_missing[:5]:
            print(f"      {k}")
        print("    说明 checkpoint 和模型结构不匹配，请检查 --model 参数")
        return
    if unexpected:
        print(f"  [!] 有 {len(unexpected)} 个多余键（通常无害）: {unexpected[:3]}")

    print(f"  权重加载完成（缺失 {len(missing)} 个缓冲区，已忽略）")

    # ---------- 数据 ----------
    test_df = pd.read_csv(OUT_DIR / "test.csv", encoding="utf-8")
    train_df = pd.read_csv(OUT_DIR / "train.csv", encoding="utf-8")
    meta_df = pd.read_csv(OUT_DIR / "metadata.csv", encoding="utf-8")

    _, eval_tf = build_transforms(img_size, mean, std)
    test_loader = DataLoader(
        FlowerDataset(test_df, eval_tf),
        batch_size=32, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"测试集: {len(test_df)} 张，训练集: {len(train_df)} 张")

    # ---------- 类别编号 -> 花名 ----------
    id2name = (meta_df.drop_duplicates("label")
               .set_index("label")["name"].to_dict())
    names = [id2name[i] for i in range(num_classes)]

    print(f"\n模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    # ---------- 推理 ----------
    print("\n正在推理 ...")
    preds, labels, top5, confs = run_inference(model, test_loader, device)

    top1 = accuracy_score(labels, preds)
    top5_acc = float((top5 == labels[:, None]).any(axis=1).mean())

    print("\n" + "=" * 62)
    print(f"测试集 Top-1 准确率 : {top1:.4f}  ({top1 * 100:.2f}%)")
    print(f"测试集 Top-5 准确率 : {top5_acc:.4f}  ({top5_acc * 100:.2f}%)")
    print(f"随机猜的基线        : {1 / num_classes:.4f}  ({1 / num_classes * 100:.2f}%)")
    print(f"相对基线提升        : {top1 / (1 / num_classes):.1f} 倍")
    print(f"平均置信度          : {confs.mean():.4f}")
    print("=" * 62)

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    p1 = plot_confusion(cm_norm, top1)
    print(f"\n混淆矩阵       -> {p1}")
    p2 = plot_per_class(per_class)
    print(f"每类准确率     -> {p2}")
    p4, pairs_df, pairs_csv = plot_confusion_pairs(cm, names)
    print(f"易混淆类别对   -> {p4}")
    print(f"                 {pairs_csv}")
    p3, rec_df = plot_predictions(model, test_df, device, mean, std, img_size)
    print(f"预测示例       -> {p3}")

    report = classification_report(
        labels, preds, labels=list(range(num_classes)),
        target_names=names, digits=3, zero_division=0,
    )
    report_path = LOG_DIR / f"classification_report_{ckpt_path.stem}.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"分类报告       -> {report_path}")

    n_train_per_class = train_df.groupby("label").size()
    detail = pd.DataFrame({
        "label": list(range(num_classes)),
        "name": names,
        "test_accuracy": per_class.round(4),
        "test_support": cm.sum(axis=1),
        "train_count": [int(n_train_per_class.get(i, 0)) for i in range(num_classes)],
    }).sort_values("test_accuracy")

    detail_csv = LOG_DIR / f"per_class_accuracy_{ckpt_path.stem}.csv"
    detail.to_csv(detail_csv, index=False, encoding="utf-8")
    print(f"每类明细       -> {detail_csv}")

    summary = {
        "checkpoint": ckpt_path.name,
        "model_type": "custom" if use_custom else "torchvision",
        "top1_acc": round(float(top1), 4),
        "top5_acc": round(float(top5_acc), 4),
        "random_baseline": round(1 / num_classes, 4),
        "test_size": int(len(test_df)),
        "train_size": int(len(train_df)),
        "mean_confidence": round(float(confs.mean()), 4),
        "num_classes_with_zero_acc": int((per_class == 0).sum()),
        "best_per_class_acc": round(float(per_class.max()), 4),
        "worst_per_class_acc": round(float(per_class.min()), 4),
    }
    summary_path = LOG_DIR / f"eval_summary_{ckpt_path.stem}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"汇总指标       -> {summary_path}")

    print("\n" + "=" * 62)
    print("最难认的 10 个类别（带花名）:")
    print("=" * 62)
    for r in detail.head(10).itertuples():
        print(f"  {r.label:>3}  {r.name[:30]:<32} "
              f"准确率 {r.test_accuracy:.2f}   "
              f"(测试 {r.test_support} 张, 训练 {r.train_count} 张)")

    print("\n" + "=" * 62)
    print("最容易混淆的 10 组类别（真实 -> 误判为）:")
    print("=" * 62)
    for r in pairs_df.head(10).itertuples():
        print(f"  {r.true_id:>3} {r.true_name[:26]:<28} "
              f"→ {r.pred_id:>3} {r.pred_name[:26]:<28} "
              f"{r.count} 张 ({r.ratio:.2f})")

    print("\n第 6 步完成")


if __name__ == "__main__":
    main()