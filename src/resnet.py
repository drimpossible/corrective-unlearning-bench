import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import initialize_weights

__all__ = ['ResNet', 'resnet9', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnetwide28x10']


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.conv1(self.bn1(x)))
        out = self.conv2(self.bn2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.norm = self.norm = nn.Sequential(nn.AvgPool2d(8), nn.Flatten())
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.norm(out)
        out = self.fc(out)
        return out


def conv_bn(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    list = [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
    nn.BatchNorm2d(out_channels),
    nn.CELU(alpha=0.075)]
    return nn.Sequential(*list)

class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)

class ResNet9(nn.Module):
    def __init__(self, num_classes: int = 10):
        super(ResNet9, self).__init__()
        self.conv1 = conv_bn(3, 64)
        self.conv2 = conv_bn(64, 128, 5, 2, 2)
        self.res1 = Residual(
                        nn.Sequential(
                            conv_bn(128, 128),
                            conv_bn(128, 128),
                        ))
        self.conv3 = nn.Sequential(conv_bn(128, 256),nn.MaxPool2d(2))
        self.res2 = Residual(
            nn.Sequential(
                conv_bn(256, 256),
                conv_bn(256, 256),
            ))
        self.conv4 = nn.Sequential(conv_bn(256, 128, 3, 1, 0),
        nn.AdaptiveMaxPool2d((1, 1)),
        nn.Flatten())
        self.fc = nn.Linear(128, num_classes, bias=False)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out)
        out = self.conv3(out)
        out = self.res2(out)
        out = self.conv4(out)
        out = self.fc(out)
        return out


class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=10):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = nn.Conv2d(3, nstages[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.layer1 = self._wide_layer(WideBasic, nstages[1], n, stride=1)
        self.layer2 = self._wide_layer(WideBasic, nstages[2], n, stride=2)
        self.layer3 = self._wide_layer(WideBasic, nstages[3], n, stride=2)
        self.norm = nn.Sequential(nn.AvgPool2d(8), nn.Flatten())
        self.fc = nn.Linear(nstages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.norm(out)
        out = self.fc(out)
        return out

def resnet20(num_classes=10):
    net = ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)
    net.embed_size = 64
    net.apply(initialize_weights)
    return net

def resnet32(num_classes=10):
    net = ResNet(BasicBlock, [5, 5, 5], num_classes=num_classes)
    net.embed_size = 64
    net.apply(initialize_weights)
    return net

def resnet44(num_classes=10):
    net = ResNet(BasicBlock, [7, 7, 7], num_classes=num_classes)
    net.embed_size = 64
    net.apply(initialize_weights)
    return net

def resnet56(num_classes=10):
    net = ResNet(BasicBlock, [9, 9, 9], num_classes=num_classes)
    net.embed_size = 64
    net.apply(initialize_weights)
    return net

def resnet110(num_classes=10):
    net = ResNet(BasicBlock, [18, 18, 18], num_classes=num_classes)
    net.embed_size = 64
    net.apply(initialize_weights)
    return net

def resnet9(num_classes=10):
    net = ResNet9(num_classes=num_classes)
    net.embed_size = 128
    net.apply(initialize_weights)
    return net

def resnetwide28x10(num_classes=10):
    net = WideResNet(depth=28, widen_factor=10, num_classes=num_classes)
    net.embed_size = 640
    net.apply(initialize_weights)
    return net


def test(net):
    import numpy as np
    total_params = 0
    inp = torch.randn(size=(1, 3, 32, 32))
    out = net(inp)
    print(f"output: {out.size()}")
    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            