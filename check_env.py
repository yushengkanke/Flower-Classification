"""
环境自检：确认 GPU 能用、依赖齐全、数据集就位。
这个文件不参与建模，纯粹是"开工前检查工具"。
"""

import sys
import platform
import os

# Windows 控制台默认是 GBK，打印中文和特殊符号（≥ →）会乱码或直接报错，
# 这行把标准输出强制切成 UTF-8。放在文件最前面，越早越好。
sys.stdout.reconfigure(encoding="utf-8")


def check_python():
    print("【Python】")
    print(f"  版本       : {sys.version.split()[0]}")
    print(f"  解释器路径 : {sys.executable}")
    # 强烈建议确认解释器在 envs 目录下，否则你可能装到了 base 环境
    if "envs" not in sys.executable and "venv" not in sys.executable:
        print("  [!] 警告：当前用的不是独立环境，建议切到 flower 环境")


def check_torch():
    print("\n【PyTorch】")
    try:
        import torch
        import torchvision
    except ImportError as e:
        print(f"  [×] 导入失败: {e}")
        print("      请执行: pip install torch torchvision --index-url "
              "https://download.pytorch.org/whl/cu128")
        return None

    print(f"  torch          : {torch.__version__}")
    print(f"  torchvision    : {torchvision.__version__}")
    print(f"  CUDA 编译版本  : {torch.version.cuda}")

    if not torch.cuda.is_available():
        # 最常见的原因：装成了比驱动更新的 CUDA 版本
        print("  [×] CUDA 不可用")
        print("      最常见原因：PyTorch 的 CUDA 版本高于显卡驱动")
        print("      驱动 576.65 最高支持 CUDA 12.9，所以要装 cu128 而不是 cu132")
        print("      或者：torch.cuda.is_available() 在没装驱动时也会 False")
        return None

    print("  CUDA 可用      : True")
    print(f"  显卡           : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"  算力(compute)  : {cap}")
    print(f"  sm 架构        : sm_{cap[0]}{cap[1]}")
    total = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
    print(f"  显存           : {total:.1f} GB")

    # 光"检测到"不够，真跑一次矩阵乘法，确认 CUDA 真的能干活
    # 这一步能把"驱动装了但运行时报错"的情况揪出来
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print(f"  实测矩阵乘法   : OK (sum={y.sum().item():.1f})")
    print(f"  峰值显存占用   : {torch.cuda.max_memory_allocated() / 1024 ** 2:.1f} MB")

    # 为后续训练准备的建议值
    if total >= 8:
        print("  → 建议 batch_size=32, img_size=224")
    else:
        print("  → 显存偏小，建议 batch_size=16, img_size=160")
    return torch


def check_libs():
    print("\n【数据分析三件套】")
    for name in ["numpy", "pandas", "matplotlib", "sklearn", "PIL", "scipy", "tqdm"]:
        try:
            m = __import__(name)
            ver = getattr(m, "__version__", "?")
            print(f"  [√] {name:<12} {ver}")
        except ImportError:
            fix = {"sklearn": "scikit-learn", "PIL": "pillow"}.get(name, name)
            print(f"  [×] {name:<12} 缺失 → pip install {fix}")


def check_dataset(root="data"):
    print(f"\n【数据集】查找目录: {root}")
    if not os.path.isdir(root):
        print("  [×] 目录不存在")
        print("      把 102flowers.tgz / imagelabels.mat / setid.mat")
        print(f"      放进 {root}/flowers-102/ 下面")
        return

    for dirpath, _, filenames in os.walk(root):
        if not filenames:
            continue
        jpg_count = sum(1 for f in filenames if f.lower().endswith(".jpg"))
        rel = os.path.relpath(dirpath, root)
        print(f"  {rel:<28} {len(filenames):>5} 个文件"
              + (f" (其中 {jpg_count} 张 jpg)" if jpg_count else ""))

    # 关键检查：图片到底解压了没有
    jpg_dir = os.path.join(root, "flowers-102", "jpg")
    n = len([f for f in os.listdir(jpg_dir) if f.endswith(".jpg")]) if os.path.isdir(jpg_dir) else 0
    if n == 0:
        print("  [!] jpg 里还没有图片，首次运行代码时会自动解压（约 330MB → 8189 张）")
    elif n == 8189:
        print(f"  [√] {n} 张图片已就位，数据集完整")
    else:
        print(f"  [!] 只有 {n} 张，正常应该是 8189 张，可能解压没完成")


if __name__ == "__main__":
    print("=" * 62)
    print(f"系统: {platform.system()} {platform.release()}")
    print("=" * 62)
    check_python()
    check_torch()
    check_libs()
    check_dataset()
    print("\n" + "=" * 62)
    print("自检结束：上面出现 [×] 的先解决，再往下走。")