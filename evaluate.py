import json
import sys
from pathlib import Path

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

def plot_confusion(cm_norm,acc,save_dir = FIG_DIR):
    save_dir.mkdir(parents=True,exist_ok=True)
    setup_chinese_font()

    fig,ax = plt.subplots(figsize=(11,9.5))
    im = ax.imshow(cm_norm,cmap="Blues",vmin=0,vmax=1)
    ax.set_xlabel("预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title(f"测试集混淆矩阵（102 类）\n整体 Top-1 准确率 {acc:.4f}")
    fig.colorbar(im, ax=ax, fraction=0.046, label="该类被判成预测类的比例")
    fig.tight_layout()
    path = save_dir / "confusion_matrix.png"
    fig.savefig(path, dpi=130)
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    ckpt_path = OUT_DIR / "checkpoints" / "best.pt"
    if not ckpt_path.exists():
        print(f"[×] 找不到 {ckpt_path}，请先运行 python train.py")
        return


    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    mean, std = ckpt["mean"], ckpt["std"]
    img_size = ckpt["img_size"]
    num_classes = ckpt["num_classes"]
    print(f"已加载 {ckpt_path}")
    print(f"  训练到第 {ckpt['epoch']} 轮，验证集最优 acc = {ckpt['best_val_acc']:.4f}")
    print(f"  归一化 mean={mean} std={std}  img_size={img_size}")


    test_df = pd.read_csv(OUT_DIR / "test.csv", encoding="utf-8")
    _, eval_tf = build_transforms(img_size, mean, std)
    test_loader = DataLoader(
        FlowerDataset(test_df, eval_tf),
        batch_size=32, shuffle=False, num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    print(f"测试集: {len(test_df)} 张")


    model = build_resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")


    print("\n正在推理 ...")
    preds, labels, top5, confs = run_inference(model, test_loader, device)


    top1 = accuracy_score(labels, preds)

    top5_acc = float((top5 == labels[:, None]).any(axis=1).mean())

    print("\n" + "=" * 62)
    print(f"测试集 Top-1 准确率 : {top1:.4f}  ({top1*100:.2f}%)")
    print(f"测试集 Top-5 准确率 : {top5_acc:.4f}  ({top5_acc*100:.2f}%)")
    print(f"随机猜的基线        : {1/num_classes:.4f}  ({1/num_classes*100:.2f}%)")
    print(f"相对基线提升        : {top1 / (1/num_classes):.1f} 倍")
    print(f"平均置信度          : {confs.mean():.4f}")
    print("=" * 62)


    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    per_class = cm.diagonal() / np.maximum(cm.sum(axis=1), 1)


    FIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    p1 = plot_confusion(cm / np.maximum(cm.sum(axis=1, keepdims=True), 1), top1)
    print(f"\n混淆矩阵     -> {p1}")
    p2 = plot_per_class(per_class)
    print(f"每类准确率   -> {p2}")

    p3, rec_df = plot_predictions(model, test_df, device, mean, std, img_size)
    print(f"预测示例     -> {p3}")


    names = sorted(test_df["name"].unique())
    report = classification_report(
        labels, preds,
        labels=list(range(num_classes)),
        target_names=names,
        digits=3, zero_division=0,
    )
    report_path = LOG_DIR / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")
    print(f"分类报告     -> {report_path}")


    summary = {
        "top1_acc": round(top1, 4),
        "top5_acc": round(top5_acc, 4),
        "random_baseline": round(1 / num_classes, 4),
        "test_size": len(test_df),
        "mean_confidence": round(float(confs.mean()), 4),
        "num_classes_with_zero_acc": int((per_class == 0).sum()),
        "best_per_class_acc": round(float(per_class.max()), 4),
        "worst_per_class_acc": round(float(per_class.min()), 4),
    }
    summary_path = LOG_DIR / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    print(f"汇总指标     -> {summary_path}")
    print("\n" + json.dumps(summary, indent=2, ensure_ascii=False))


    print("\n最难认的 10 个类别:")
    worst = np.argsort(per_class)[:10]
    for c in worst:
        cnt = cm[c].sum()
        print(f"  {names[c][:28]:<30} 准确率 {per_class[c]:.2f}  (该测试集有 {cnt} 张)")

    print("\n第 6 步完成")


if __name__ == "__main__":
    main()