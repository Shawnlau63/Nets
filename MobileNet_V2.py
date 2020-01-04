import time
import math
import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, group=1):
        super(ConvLayer, self).__init__()
        self.sub_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=group, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.sub_layer(x)


class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand=6):
        super(BottleNeck, self).__init__()
        assert stride in [1, 2]
        if stride == 1 and in_channel == out_channel:
            self.use_res = True
        else:
            self.use_res = False

        self.sub_layer = nn.Sequential(
            ConvLayer(in_channel, in_channel * expand, 1, 1, 0),
            ConvLayer(in_channel * expand, in_channel * expand, 3, stride, 1, group=in_channel * expand),
            nn.Conv2d(in_channel * expand, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        out = (x + self.sub_layer(x)) if self.use_res else self.sub_layer(x)

        return out


class MobileNetV2(nn.Module):
    def __init__(self):
        super(MobileNetV2, self).__init__()
        self.sub_layer = nn.Sequential(
            ConvLayer(3, 32, 3, 2, 1),  # 112 * 112 * 32
            BottleNeck(32, 16, 1),  # 112 * 112 * 16
            BottleNeck(16, 16, 2),  # 56 * 56 * 16
            BottleNeck(16, 24, 1),  # 56 * 56 * 24

            BottleNeck(24, 24, 2),  # 28 * 28 * 24
            BottleNeck(24, 32, 1),  # 28 * 28 * 32
            BottleNeck(32, 32, 1),  # 28 * 28 * 32

            BottleNeck(32, 32, 1),  # 28 * 28 * 32
            BottleNeck(32, 64, 1),  # 28 * 28 * 64
            BottleNeck(64, 64, 1),  # 28 * 28 * 64
            BottleNeck(64, 64, 1),  # 28 * 28 * 64

            BottleNeck(64, 64, 2),  # 14 * 14 * 64
            BottleNeck(64, 96, 1),  # 14 * 14 * 96
            BottleNeck(96, 96, 1),  # 14 * 14 * 96

            BottleNeck(96, 96, 2),  # 7 * 7 * 96
            BottleNeck(96, 160, 1),  # 7 * 7 * 160
            BottleNeck(160, 160, 1),  # 7 * 7 * 160

            BottleNeck(160, 320, 1),  # 7 * 7 * 320
            BottleNeck(320, 1280, 1),  # 7 * 7 * 1280
        )

        self.avgpool = nn.AvgPool2d(7)  # 1 * 1 * 1280

        self.classifier = nn.Linear(1280, 100)  # class_num

    def forward(self, x):
        _out = self.sub_layer(x)
        out = self.avgpool(_out)
        out = out.view(out.size(0), -1)
        output = self.classifier(out)

        return output

    def _ini_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


if __name__ == '__main__':
    model = MobileNetV2()
    model.eval()
    model.cuda()

    input = torch.randn(10, 3, 224, 224).cuda()

    for i in range(20):
        start = time.time()
        output = model(input)

        print(output.shape, time.time() - start)
