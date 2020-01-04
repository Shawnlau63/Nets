import time
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
MobileNetV3-Large
'''

class hswish(nn.Module):
    def __init__(self, inplace=True):
        super(hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = x * F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3, inplace=self.inplace) / 6
        return out


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SEModel(nn.Module):
    def __init__(self, in_channel, reduction=4):
        super(SEModel, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sub_layer = nn.Sequential(
            nn.Linear(in_channel, in_channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction, in_channel, bias=False),
            hsigmoid()
        )

    def forward(self, x):
        batch, channel, height, width = x.size()
        y = self.avg_pool(x)
        y = y.view(batch, channel)
        y = self.sub_layer(y)
        y = y.view(batch, channel, 1, 1)
        return x * y.expand_as(x)


class BottleNeck(nn.Module):
    def __init__(self, in_channel, expand_size, out_channel, kernel_size, stride, SE=False, NL='RE'):
        super(BottleNeck, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        padding = (kernel_size - 1) // 2
        if stride == 1 and in_channel == out_channel:
            self.use_res = True
        else:
            self.use_res = False

        if NL == 'RE':
            nlin_layer = nn.ReLU
        elif NL == 'HS':
            nlin_layer = hswish
        else:
            raise NotImplementedError

        if SE:
            SELayer = SEModel
        else:
            SELayer = Identity

        self.sub_layer = nn.Sequential(
            # pointwise
            nn.Conv2d(in_channel, expand_size, 1, 1, 0, bias=False),
            nn.BatchNorm2d(expand_size),
            nlin_layer(inplace=True),

            # depthwise
            nn.Conv2d(expand_size, expand_size, kernel_size, stride, padding, groups=expand_size, bias=False),
            nn.BatchNorm2d(expand_size),
            SELayer(expand_size),
            nlin_layer(inplace=True),

            # depthwise-linear
            nn.Conv2d(expand_size, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, x):
        if self.use_res:
            return x + self.sub_layer(x)
        else:
            return self.sub_layer(x)


class MobileNetV3(nn.Module):
    def __init__(self):
        super(MobileNetV3, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            hswish(inplace=True)
        )
        self.sub_layer = nn.Sequential(
            BottleNeck(16, 16, 16, 3, 1, False, 'RE'),
            BottleNeck(16, 64, 24, 3, 2, False, 'RE'),
            BottleNeck(24, 72, 24, 3, 1, False, 'RE'),
            BottleNeck(24, 72, 40, 5, 2, True, 'RE'),
            BottleNeck(40, 120, 40, 5, 1, True, 'RE'),
            BottleNeck(40, 120, 40, 5, 1, True, 'RE'),
            BottleNeck(40, 240, 80, 3, 2, False, 'HS'),
            BottleNeck(80, 200, 80, 3, 1, False, 'HS'),
            BottleNeck(80, 184, 80, 3, 1, False, 'HS'),
            BottleNeck(80, 184, 80, 3, 1, False, 'HS'),
            BottleNeck(80, 480, 112, 3, 1, True, 'HS'),
            BottleNeck(112, 672, 112, 3, 1, True, 'HS'),
            BottleNeck(112, 672, 160, 5, 2, True, 'HS'),
            BottleNeck(160, 960, 160, 5, 1, True, 'HS'),
            BottleNeck(160, 960, 160, 5, 1, True, 'HS'),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(160, 960, 1, 1, 0),
            nn.BatchNorm2d(960),
            hswish(inplace=True)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(960, 1280),
            nn.BatchNorm1d(1280),
            hswish(inplace=True)
        )
        self.classifier = nn.Linear(1280, 100)

        self.init_params()

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.sub_layer(y)
        y = self.conv2(y)
        y = self.avg_pool(y)
        # print(y.shape)
        y = y.view(y.size(0), -1)
        # print(y.shape)
        y = self.linear1(y)
        y = self.classifier(y)

        return y


if __name__ == '__main__':
    model = MobileNetV3()
    model.eval()
    model.cuda()

    input = torch.randn(10, 3, 224, 224).cuda()

    for i in range(20):
        start = time.time()
        output = model(input)

        print(output.shape, time.time() - start)