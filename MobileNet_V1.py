import time
import torch
import torch.nn as nn


class Conv_bn(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Conv_bn, self).__init__()
        self.sub_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.sub_layer(x)


class Conv_dw(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(Conv_dw, self).__init__()
        self.sub_layer = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, 3, stride, 1, groups=in_channel, bias=False),
            nn.BatchNorm2d(in_channel),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channel, out_channel, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        return self.sub_layer(x)


# input:224 * 224 * 3
class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.model = nn.Sequential(
            Conv_bn(3, 32, 2),  # 112 * 112 * 32
            Conv_dw(32, 32, 1),  # 112 * 112 * 32
            Conv_bn(32, 64, 1),  # 112 * 112 * 64
            Conv_dw(64, 64, 2),  # 56 * 56 * 64
            Conv_bn(64, 128, 1),  # 56 * 56 * 128
            Conv_dw(128, 128, 1),  # 56 * 56 * 128
            Conv_bn(128, 128, 1),  # 56 * 56 * 128
            Conv_dw(128, 128, 2),  # 28 * 28 * 128
            Conv_bn(128, 256, 1),  # 28 * 28 * 256
            Conv_dw(256, 256, 1),  # 28 * 28 * 256
            Conv_bn(256, 256, 1),  # 28 * 28 * 256
            Conv_dw(256, 256, 2),  # 14 * 14 * 256
            Conv_bn(256, 512, 1),  # 14 * 14 * 512

            Conv_dw(512, 512, 1),  # 14 * 14 * 512
            Conv_bn(512, 512, 1),  # 14 * 14 * 512
            Conv_dw(512, 512, 1),  # 14 * 14 * 512
            Conv_bn(512, 512, 1),  # 14 * 14 * 512
            Conv_dw(512, 512, 1),  # 14 * 14 * 512
            Conv_bn(512, 512, 1),  # 14 * 14 * 512
            Conv_dw(512, 512, 1),  # 14 * 14 * 512
            Conv_bn(512, 512, 1),  # 14 * 14 * 512
            Conv_dw(512, 512, 1),  # 14 * 14 * 512
            Conv_bn(512, 512, 1),  # 14 * 14 * 512

            Conv_dw(512, 512, 2),  # 7 * 7 * 512
            Conv_bn(512, 1024, 1),  # 7 * 7 * 1024
            Conv_dw(1024, 1024, 1),  # 7 * 7 * 1024
            Conv_bn(1024, 1024, 1),  # 7 * 7 * 1024

            nn.AvgPool2d(7)
        )

        self.classifier = nn.Linear(1024, 100)  # class_num

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return x


if __name__ == '__main__':
    model = MobileNetV1()
    model.eval()
    model.cuda()

    input = torch.randn(10, 3, 224, 224).cuda()

    for i in range(20):
        start = time.time()
        out = model(input)

        print(out.shape, time.time() - start)
