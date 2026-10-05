import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

def split_dataframe(df,val_ratio=0.15,test_ratio=0.15,seed=42):
    train_val_df,test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df["label"],
        random_state=seed
    )

    val_size = val_ratio/(1-test_ratio)
    train_df,val_df = train_test_split(
        train_val_df,
        test_size = val_size,
        stratify = train_val_df["label"],
        random_state=seed
    )
    return train_df,val_df,test_df


def build_transforms(img_size=224,mean=None,std=None):

    mean = mean or [0.4321,0.3772,0.2865]
    std = std or [0.2962,0.2433,0.2683]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size,scale = (0.7,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.2,0.2,0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(img_size*1.14),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean,std),
    ])
    return train_tf,eval_tf

class FlowerDataset(Dataset):

    def __init__(self,df,transform=None):
        self.paths = df["path"].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        with Image.open(self.paths[index]) as img:
            img = img.convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
        return img,self.labels[index]


def build_loaders(train_df,val_df,test_df,img_size=224,batch_size=32,num_workers=0,mean=None,std=None):
    train_tf,eval_tf = build_transforms(img_size,mean,std)
    loaders ={}
    for name,sub_df,tf in [
        ("train",train_df,train_tf),
        ("val",val_df,eval_tf),
        ("test",test_df,eval_tf)
    ]:
        ds = FlowerDataset(sub_df,tf)
        loaders[name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle = (name =="train"),
            num_workers = num_workers,
            pin_memory = True,
            drop_last = (name =="train")
        )
    return loaders

def compute_stats(df,size=128,limit=1000,seed=42):
    tf = transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor()])
    paths = df["path"].sample(min(limit,len(df)),random_state=seed).tolist()
    all_pixels = []
    for p in paths:
        with Image.open(p) as im:
            all_pixels.append(tf(im.convert("RGB")).numpy())

    arr = np.stack(all_pixels)
    flat = arr.transpose(0,2,3,1).reshape(-1,3)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0)
    return [round(float(v), 4) for v in mean], [round(float(v), 4) for v in std]
def load_stats(default_mean=None, default_std=None):
    stats_path = OUT_DIR / "stats.json"
    if stats_path.exists():
        import json
        with open(stats_path, encoding="utf-8") as f:
            s = json.load(f)
        return s["mean"], s["std"]

    print(f"[!] 没找到 {stats_path}，使用默认归一化参数")
    print("    建议先运行: python src/data/dataset.py")
    return (
        default_mean or [0.4305, 0.3722, 0.282],
        default_std or [0.2901, 0.2377, 0.2621],
    )

if __name__ == "__main__":

    meta_path = OUT_DIR / "metadata.csv"
    print(f"读取 {meta_path} ...")
    df = pd.read_csv(meta_path, encoding="utf-8")
    print(f"共 {len(df)} 张，{df['label'].nunique()} 类\n")


    print("=" * 62)
    print("官方划分（Flowers102 原始 split）:")
    print(df.groupby("split").size().to_string())

    train_df, val_df, test_df = split_dataframe(df)
    print("\n自己分层切分后:")
    for name, sub in [("train", train_df), ("val", val_df), ("test", test_df)]:
        print(f"  {name:<6} {len(sub):>5} 张   覆盖类别 {sub['label'].nunique()}")
    print("=" * 62)


    train_df.to_csv(OUT_DIR / "train.csv", index=False, encoding="utf-8")
    val_df.to_csv(OUT_DIR / "val.csv", index=False, encoding="utf-8")
    test_df.to_csv(OUT_DIR / "test.csv", index=False, encoding="utf-8")
    print(f"\n划分结果已保存到 {OUT_DIR}\\ train.csv / val.csv / test.csv")


    print("\n正在统计通道均值/标准差（抽样 1000 张）...")
    mean, std = compute_stats(df)
    print(f"mean = {mean}")
    print(f"std  = {std}")
    with open(OUT_DIR / "stats.json", "w", encoding="utf-8") as f:
        json.dump({"mean": mean, "std": std}, f, indent=2)
    print(f"已保存 -> {OUT_DIR / 'stats.json'}")


    print("\n" + "=" * 62)
    print("自检：")
    loaders = build_loaders(train_df, val_df, test_df, num_workers=0)

    ds = loaders["train"].dataset
    print(f"len(dataset) = {len(ds)}")
    x, y = ds[0]
    print(f"ds[0] -> 图片 {tuple(x.shape)}  dtype={x.dtype}  标签 {y} (类型 {type(y).__name__})")

    imgs, labels = next(iter(loaders["train"]))
    print(f"一个 batch -> imgs {tuple(imgs.shape)}  labels {tuple(labels.shape)}")
    print(f"归一化后 均值={imgs.mean():.3f} 标准差={imgs.std():.3f}  (应接近 0 和 1)")
    print("=" * 62)
    print("\n第 3 步完成")