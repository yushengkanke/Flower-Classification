import sys
import torch
import torch.nn as nn
from residual import BasicBlock
sys.stdout.reconfigure(encoding='utf-8')
class ResNet(nn.Module):
    def __init__(self,block=BasicBlock,layers=(2,2,2,2),num_classes=102):
        super().__init__()
        self.in_ch = 64
        self.stem = nn.Sequential(
            nn.Conv2d(
                3,64,kernel_size=7,
                    stride=2,
                    padding=3,
                    bias=False
                    ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),

        )

        self.layer1 = self._make_layer(block,64,layers[0],stride=1)
        self.layer2 = self._make_layer(block,128,layers[1],stride=2)
        self.layer3 = self._make_layer(block,256,layers[2],stride=2)
        self.layer4 = self._make_layer(block,512,layers[3],stride=2)

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512*block.expansion,num_classes)

        self.init_weights()

    def _make_layer(self, block, out_ch, num_blocks, stride):

        layers = []
        layers.append(block(self.in_ch, out_ch, stride))

        self.in_ch = out_ch * block.expansion

        for _ in range(1, num_blocks):
            layers.append(block(self.in_ch, out_ch, stride=1))

        return nn.Sequential(*layers)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu')
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)

    def forward(self,x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x

def build_resnet18(num_classes=102):
    return ResNet(BasicBlock,[2,2,2,2],num_classes)

def build_resnet34(num_classes=102):
    return ResNet(BasicBlock,[3,4,6,3],num_classes)

if __name__ == "__main__":
    print("=" * 66)
    print("ResNet-18 结构自检")
    print("=" * 66)

    model = build_resnet18(num_classes=102)

    # ---- 检查 1：参数量 ----
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n参数量     : {n_params:,}")
    print(f"可训练参数 : {n_trainable:,}")
    print(f"参考值     : 11,228,838")
    if n_params == 11228838:
        print("            ✓ 完全匹配，结构正确")
    else:
        print(f"            ✗ 差了 {n_params - 11228838:+,}，检查层配置")

    # ---- 检查 2：逐层输出尺寸 ----
    print("\n各层输出尺寸:")
    x = torch.randn(2, 3, 224, 224)
    print(f"  {'输入':<24} {tuple(x.shape)}")
    x = model.stem(x)
    print(f"  {'stem':<24} {tuple(x.shape)}")
    for name in ("layer1", "layer2", "layer3", "layer4"):
        x = getattr(model, name)(x)
        n_blocks = len(getattr(model, name))
        print(f"  {name + f' ({n_blocks}块)':<24} {tuple(x.shape)}")
    x = model.pool(x)
    print(f"  {'pool':<24} {tuple(x.shape)}")
    x = torch.flatten(x, 1)
    x = model.fc(x)
    print(f"  {'fc':<24} {tuple(x.shape)}")

    # ---- 检查 3：各 stage 的参数量分布 ----
    print("\n各 stage 参数量分布:")
    total = n_params
    for name in ("stem", "layer1", "layer2", "layer3", "layer4", "fc"):
        module = getattr(model, name)
        n = sum(p.numel() for p in module.parameters())
        print(f"  {name:<10} {n:>10,}   {n / total * 100:>5.1f}%")

    print("\n" + "=" * 66)
    print("尺寸路径：224 → 56 → 28 → 14 → 7 → 1")
    print("通道路径：3 → 64 → 128 → 256 → 512")
    print("=" * 66)
