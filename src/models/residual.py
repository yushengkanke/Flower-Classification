import sys
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self,in_ch,out_ch,stride=1):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_ch,out_ch,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(
            out_ch,out_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None

        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):

        identity = x if self.downsample is None else self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out = out + identity
        out = self.relu(out)
        return out


def _test_block(name,**kwargs):

    x = torch.randn(2,kwargs["in_ch"],56,56)
    block = BasicBlock(**kwargs)
    y = block(x)
    has_ds = block.downsample is not None
    print(f"  {name}")
    print(f"    输入  {tuple(x.shape)}  →  输出 {tuple(y.shape)}")
    print(f"    downsample: {'有（需要投影）' if has_ds else '无（恒等映射）'}")


if __name__ == "__main__":
    print("=" * 66)
    print("BasicBlock 测试")
    print("=" * 66)

    _test_block("① 形状完全不变（stride=1, 通道数相同）",
                in_ch=64, out_ch=64, stride=1)

    _test_block("② 通道数变（stride=1, 64→128）",
                in_ch=64, out_ch=128, stride=1)

    _test_block("③ 尺寸减半（stride=2, 64→128）",
                in_ch=64, out_ch=128, stride=2)

    print("=" * 66)
    print("预期：")
    print("  ① 无 downsample，输出 [2, 64, 56, 56]")
    print("  ② 有 downsample，输出 [2, 128, 56, 56]（尺寸不变、通道变）")
    print("  ③ 有 downsample，输出 [2, 128, 28, 28]（尺寸减半、通道变）")
    print()
    print("如果 ② ③ 报 'The size of tensor a must match tensor b'，")
    print("说明 downsample 的判断条件写错了。")
    print("=" * 66)
