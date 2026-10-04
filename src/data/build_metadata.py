import os
import sys
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.datasets import Flowers102

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROOT = str(PROJECT_ROOT / "data")
OUT_CSV = str(PROJECT_ROOT / "outputs/metadata.csv")
FIG_DIR = str(PROJECT_ROOT / "outputs/figures")

def setup_chinese_font():
    plt.rcParams['font.sans-serif'] = [
        "Microsoft YaHei",  # 微软雅黑，Win7+ 都有
        "SimHei",  # 黑体，备选
        "DejaVu Sans",  # 最后兜底，不支持中文但不至于报错

    ]
    plt.rcParams["axes.unicode_minus"] = False

def build_metadata(root=ROOT):
    frames = []
    idx_to_name = None

    for split in ("train", "val", "test"):

        ds = Flowers102( root = root, split = split ,download=True)

        if idx_to_name is None:
            idx_to_name = dict(enumerate(ds.classes))

        frames.append(pd.DataFrame({
            "path": [str(p)for p in ds._image_files],
            "label":[int(x) for x in ds._labels],
            "split": split,
        }))

    df = pd.concat(frames,ignore_index=True)

    df["name"] = df["label"].map(idx_to_name)
    return df
def analyze(df):
     os.makedirs(FIG_DIR, exist_ok=True)
     os.makedirs(os.path.dirname(OUT_CSV),exist_ok=True)
     setup_chinese_font()

     print("="*62)
     print(f"总图片数:{len(df)}")
     print(f"类别数:{df["label"].nunique()}")
     print(f"图片路径去重后:{df["path"].nunique()}")

     counts = df.groupby("label").size().sort_values(ascending=False)
     print(f"\n每样类别数:")
     print(f"  最多 {counts.max()} 张   最少 {counts.min()} 张   "
           f"平均 {counts.mean():.1f} 张")
     print(f"  不平衡比例 {counts.max() / counts.min():.1f} : 1")
     print(f"  样本最多的 3 类:\n{counts.head(3).to_string()}")

     print("\n各集合样本数:")
     print(df.groupby("split").size().to_string())
     print("\n各集合覆盖的类别数 (都应该是 102):")
     print(df.groupby("split")["label"].nunique().to_string())


     fig, ax = plt.subplots(figsize=(14, 5))
     ax.bar(range(len(counts)), counts.values, width=1.0, color="#4C72B0")
     ax.axhline(counts.mean(), color="red", ls="--", lw=1.5,
                label=f"平均 {counts.mean():.1f} 张/类")
     ax.set_xlabel("类别编号（按样本数从多到少排序）")
     ax.set_ylabel("样本数量")
     ax.set_title(f"102 类花卉样本分布 —— 共 {len(df)} 张，"
                  f"最多 {counts.max()} 张 / 最少 {counts.min()} 张")
     ax.legend()
     fig.tight_layout()
     p1 = os.path.join(FIG_DIR, "class_dist.png")
     fig.savefig(p1, dpi=130)
     plt.close(fig)
     print(f"\n已保存 -> {p1}")

     split_counts = df.groupby("split").size()
     fig, ax = plt.subplots(figsize=(6, 4.5))
     bars = ax.bar(split_counts.index, split_counts.values,
                   color=["#4C72B0", "#DD8452", "#55A868"])
     for b, v in zip(bars, split_counts.values):
         ax.text(b.get_x() + b.get_width() / 2, v + 50, str(v),
                 ha="center", fontsize=10)
     ax.set_ylabel("样本数量")
     ax.set_title("官方三个集合的样本量")
     fig.tight_layout()
     p2 = os.path.join(FIG_DIR, "split_dist.png")
     fig.savefig(p2, dpi=130)
     plt.close(fig)
     print(f"已保存 -> {p2}")

     df.to_csv(OUT_CSV, index=False, encoding="utf-8")
     print(f"元数据表已保存 -> {OUT_CSV}")
     return df

if __name__ == "__main__":
    print("正在读取数据集（首次运行会解压 328MB，需要 1-3 分钟）...")
    df = build_metadata()
    analyze(df)
    print("\n第 2 步完成")