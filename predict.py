import argparse
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
sys.stdout.reconfigure(encoding="utf-8")

matplotlib.use("Agg")
import matplotlib.pyplot as plt
PROJECT_ROOT = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"
LOG_DIR = OUT_DIR / "logs"
CKPT_PATH = OUT_DIR / "checkpoints" / "best.pt"

from src.data.dataset import FlowerDataset, build_transforms
from src.models.resnet import build_resnet18
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def setup_chinese_font():
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def load_model(device):
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"找不到 {CKPT_PATH}，请先运行 python train.py")
    ckpt = torch.load(CKPT_PATH,map_location=device,weights_only=False)
    mean,std = ckpt["mean"],ckpt["std"]
    img_size = ckpt["img_size"]
    num_classes = ckpt["num_classes"]

    print(f"模型文件   : {CKPT_PATH.name}")
    print(f"训练信息   : 第 {ckpt['epoch']} 轮, 验证集 acc = {ckpt['best_val_acc']:.4f}")
    print(f"预处理参数 : img_size={img_size}  mean={mean}  std={std}")

    model = build_resnet18(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = pd.read_csv(OUT_DIR / "metadata.csv",encoding="utf-8")

    id2name = meta.drop_duplicates("label").set_index("label")["name"].to_dict()
    _,eval_tf = build_transforms(img_size,mean,std)
    return model,eval_tf,id2name,meta

@torch.no_grad()
def predict_image(model, image_path, eval_tf, id2name, device, topk=5):
    with Image.open(image_path) as im:
        rgb = im.convert("RGB")

        x = eval_tf(rgb).unsqueeze(0).to(device)

        logits = model(x)

        probs = F.softmax(logits, dim=1)[0]

    top_probs, top_ids = probs.topk(topk)
    result = [
        {
            "label":int(i),"name":id2name.get(int(i),str(int(i))),
            "prob":float(p)

        }
        for p, i in zip(top_probs.cpu(), top_ids.cpu())
    ]
    return result ,rgb.copy()

def format_result(result,true_label=None,id2name=None):
    lines =[]
    for rank ,r in enumerate(result,1):
        mark=""
        if true_label is not None and r["label"]==true_label:
            mark = "   ← 真实类别"
        lines.append(f"{rank}. {r['name'][:26]:<28} {r['prob']:>6.1%}{mark}")
    return "\n".join(lines)


def plot_single(image, results, image_name, true_label=None, id2name=None,
                save_path=None):

    setup_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                             gridspec_kw={"width_ratios": [1, 1.3]})


    axes[0].imshow(image)
    axes[0].axis("off")
    title = f"输入图片: {image_name}"
    if true_label is not None and id2name is not None:
        title += f"\n真实类别: {id2name.get(true_label, true_label)}"
    axes[0].set_title(title, fontsize=10)


    names = [r["name"][:24] for r in results][::-1]
    probs = [r["prob"] for r in results][::-1]
    colors = ["#55A868" if (true_label is None or r["label"] == true_label)
              else "#C44E52" for r in results][::-1]

    ax = axes[1]
    bars = ax.barh(range(len(names)), probs, color=colors)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("预测概率")
    ax.set_xlim(0, max(probs) * 1.25)
    ax.set_title(f"Top-{len(results)} 预测" +
                 ("     绿色=正确 / 红色=错误" if true_label is not None else ""))
    for b, p in zip(bars, probs):
        ax.text(b.get_width() + max(probs) * 0.02, b.get_y() + b.get_height() / 2,
                f"{p:.1%}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    if save_path is None:
        save_path = FIG_DIR / "predict_single.png"
    fig.savefig(save_path, dpi=140)
    plt.close(fig)
    return save_path

def predict_folder(model, folder, eval_tf, id2name, device, max_show=6):

    folder = Path(folder)
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in IMG_EXTS)
    if not files:
        print(f"[!] {folder} 里没找到图片文件（支持 {', '.join(sorted(IMG_EXTS))}）")
        return None

    print(f"\n文件夹里找到 {len(files)} 张图片")


    rows = []
    for f in files:
        results, _ = predict_image(model, f, eval_tf, id2name, device)
        rows.append({
            "file": f.name,
            "pred_label": results[0]["label"],
            "pred_name": results[0]["name"],
            "confidence": round(results[0]["prob"], 4),
            "top2_name": results[1]["name"],
            "top2_prob": round(results[1]["prob"], 4),
            "top3_name": results[2]["name"],
            "top3_prob": round(results[2]["prob"], 4),
        })
    result_df = pd.DataFrame(rows)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = LOG_DIR / "predict_folder_result.csv"
    result_df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\n结果明细已保存 -> {csv_path}")


    print("\n预测结果:")
    for r in result_df.itertuples():
        print(f"  {r.file[:30]:<32} → {r.pred_name[:26]:<28} {r.confidence:.1%}")


    show = files[:max_show]
    n = len(show)
    cols = min(3, n)
    rows_n = (n + cols - 1) // cols
    setup_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.4, rows_n * 4.2),
                             squeeze=False)
    for ax, f in zip(axes.ravel(), show):
        results, img = predict_image(model, f, eval_tf, id2name, device, topk=3)
        ax.imshow(img)
        ax.axis("off")
        lines = [f"{i}. {r['name'][:22]} {r['prob']:.0%}"
                 for i, r in enumerate(results, 1)]
        ax.set_title(f"{f.name[:26]}\n" + "\n".join(lines), fontsize=8)
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(f"批量预测（显示前 {n} 张，共 {len(files)} 张）", fontsize=13)
    fig.tight_layout()
    grid_path = FIG_DIR / "predict_folder.png"
    fig.savefig(grid_path, dpi=140)
    plt.close(fig)
    print(f"网格图已保存 -> {grid_path}")

    return result_df

def interactive_loop(model, eval_tf, id2name, device, topk=5):

    print("=" * 62)
    print("交互式预测模式")
    print("=" * 62)
    print("使用说明：")
    print("  · 输入图片路径，回车即可预测")
    print("  · 输入文件夹路径，会批量预测文件夹里所有图片")
    print("  · 路径可以直接从资源管理器拖进来，或者复制粘贴")
    print("  · 直接回车 / 输入 q / quit → 退出程序")
    print("  · 相对路径以项目根目录为基准")
    print("  例如: data/flowers-102/jpg/image_00001.jpg")
    print("=" * 62)

    n_predicted = 0

    while True:

        try:
            raw = input("\n请输入图片路径 > ").strip()
        except (EOFError, KeyboardInterrupt):

            print("\n已退出。")
            break


        if raw == "" or raw.lower() in ("q", "quit", "exit"):
            print(f"已退出。本次共预测 {n_predicted} 次。")
            break


        clean = raw.strip().strip('"').strip("'")

        target = Path(clean)

        if not target.is_absolute():
            target = (PROJECT_ROOT / target).resolve()

        if not target.exists():
            print(f"  [×] 路径不存在: {target}")
            print("      提示：可以把文件从资源管理器直接拖进这个窗口")
            continue


        if target.is_dir():
            print(f"\n检测到文件夹: {target.name}")
            predict_folder(model, target, eval_tf, id2name, device)
            n_predicted += 1
            continue


        if target.suffix.lower() not in IMG_EXTS:
            print(f"  [!] 这可能不是图片文件（后缀 {target.suffix}）")
            print(f"      支持的格式: {', '.join(sorted(IMG_EXTS))}")
            print("      继续尝试读取 ...")

        try:
            results, img = predict_image(model, target, eval_tf, id2name,
                                         device, topk)
        except Exception as e:

            print(f"  [×] 读取图片失败: {type(e).__name__}: {e}")
            print("      常见原因: HEIC/RAW 等格式不支持，请先转成 JPG")
            continue


        print(f"\n图片: {target.name}")
        print("-" * 62)
        print(format_result(results))
        print("-" * 62)
        print(f"最可能的类别: {results[0]['name']}  "
              f"（置信度 {results[0]['prob']:.1%}）")


        safe_name = "".join(c if c.isalnum() or c in "-_" else "_"
                            for c in target.stem)[:40]
        save_path = FIG_DIR / f"predict_{safe_name}.png"
        try:
            plot_single(img, results, target.name, save_path=save_path)
            print(f"结果图已保存 -> {save_path}")
        except Exception as e:
            print(f"  [!] 保存结果图失败: {e}")

        n_predicted += 1
def main():
    ap = argparse.ArgumentParser(description="用训练好的 ResNet-18 预测花卉类别")

    ap.add_argument("--image", type=str, default=None,
                    help="图片路径或文件夹路径；不填则随机抽测试集图片演示")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--num-demo", type=int, default=6,
                    help="演示模式下随机抽几张")
    ap.add_argument("--interactive", action="store_true", default=True,
                    help="交互模式：启动后手动输入图片路径（默认开启）")
    ap.add_argument("--no-interactive", dest="interactive",
                    action="store_false",
                    help="关闭交互模式，走原来的演示/命令行模式")

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 62)
    model, eval_tf, id2name, meta = load_model(device)
    print("=" * 62)


    if args.interactive:
        interactive_loop(model, eval_tf, id2name, device, args.topk)
        return


    if args.image is not None:
        target = Path(args.image)
        if not target.exists():
            print(f"[×] 路径不存在: {target}")
            return

        if target.is_dir():
            predict_folder(model, target, eval_tf, id2name, device)
        else:
            results, img = predict_image(model, target, eval_tf, id2name,
                                         device, args.topk)
            print(f"\n预测结果（Top-{args.topk}）:")
            print(format_result(results))
            print(f"\n最可能的类别: {results[0]['name']}  "
                  f"（置信度 {results[0]['prob']:.1%}）")
            p = plot_single(img, results, target.name, save_path=FIG_DIR / "predict_single.png")
            print(f"结果图已保存 -> {p}")
        return


    print("\n没指定 --image，随机抽取测试集图片演示\n")
    test_df = pd.read_csv(OUT_DIR / "test.csv", encoding="utf-8")
    sample = test_df.sample(args.num_demo, random_state=7).reset_index(drop=True)

    n = len(sample)
    cols = 3
    rows_n = (n + cols - 1) // cols
    setup_chinese_font()
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(rows_n, cols, figsize=(cols * 4.6, rows_n * 4.4),
                             squeeze=False)

    all_rows = []
    for ax, row in zip(axes.ravel(), sample.itertuples()):
        results, img = predict_image(model, row.path, eval_tf, id2name,
                                     device, args.topk)
        pred_name = results[0]["name"]
        correct = results[0]["label"] == row.label

        ax.imshow(img)
        ax.axis("off")
        lines = [f"{i}. {r['name'][:20]} {r['prob']:.0%}"
                 for i, r in enumerate(results[:3], 1)]
        ax.set_title(
            f"[{'正确' if correct else '错误'}] 预测: {pred_name[:20]}\n"
            f"真实: {row.name[:20]}\n" + "\n".join(lines),
            fontsize=8,
            color=("green" if correct else "red"),
        )
        all_rows.append({
            "file": Path(row.path).name,
            "true_name": row.name,
            "pred_name": pred_name,
            "confidence": round(results[0]["prob"], 4),
            "correct": correct,
        })
        print(f"  [{'√' if correct else '×'}] {Path(row.path).name:<22} "
              f"预测 {pred_name[:26]:<28} ({results[0]['prob']:.1%})  "
              f"真实 {row.name[:26]}")

    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.suptitle(f"随机抽取 {n} 张测试集图片预测演示", fontsize=13)
    fig.tight_layout()
    demo_path = FIG_DIR / "predict_demo.png"
    fig.savefig(demo_path, dpi=140)
    plt.close(fig)

    demo_df = pd.DataFrame(all_rows)
    demo_csv = LOG_DIR / "predict_demo.csv"
    demo_df.to_csv(demo_csv, index=False, encoding="utf-8")

    acc = demo_df["correct"].mean()
    print(f"\n这 {n} 张里预测正确 {int(demo_df['correct'].sum())} 张（{acc:.1%}）")
    print(f"演示图   -> {demo_path}")
    print(f"结果明细 -> {demo_csv}")
    print("\n第 7 步完成 —— 用你自己的图片试试:")
    print("    python predict.py --image 你的图片.jpg")


if __name__ == "__main__":
    main()