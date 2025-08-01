import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class FixedStem(nn.Module):
    def __init__(self):
        super(FixedStem, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.stem(x)

class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(VGGBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x, kin=None, kout=None, ksize=None):
        if kin is None or kout is None or ksize is None:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)
            out = self.maxpool(out)
            return out
        padding = ksize // 2
        out = F.conv2d(x, self.conv1.weight[:kout, :kin, :ksize, :ksize], self.conv1.bias[:kout], padding=padding)
        out = F.batch_norm(out, self.bn1.running_mean[:kout], self.bn1.running_var[:kout], self.bn1.weight[:kout], self.bn1.bias[:kout], training=True)
        out = self.relu1(out)
        out = F.conv2d(out, self.conv2.weight[:kout, :kout, :ksize, :ksize], self.conv2.bias[:kout], padding=padding)
        out = F.batch_norm(out, self.bn2.running_mean[:kout], self.bn2.running_var[:kout], self.bn2.weight[:kout], self.bn2.bias[:kout], training=True)
        out = self.relu2(out)
        out = self.maxpool(out)
        return out

class VGGBlock_Nomax(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(VGGBlock_Nomax, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x, kin=None, kout=None, ksize=None):
        if kin is None or kout is None or ksize is None:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu1(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu2(out)
            return out
        padding = ksize // 2
        out = F.conv2d(x, self.conv1.weight[:kout, :kin, :ksize, :ksize], self.conv1.bias[:kout], padding=padding)
        out = F.batch_norm(out, self.bn1.running_mean[:kout], self.bn1.running_var[:kout], self.bn1.weight[:kout], self.bn1.bias[:kout], training=True)
        out = self.relu1(out)
        out = F.conv2d(out, self.conv2.weight[:kout, :kout, :ksize, :ksize], self.conv2.bias[:kout], padding=padding)
        out = F.batch_norm(out, self.bn2.running_mean[:kout], self.bn2.running_var[:kout], self.bn2.weight[:kout], self.bn2.bias[:kout], training=True)
        out = self.relu2(out)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x, kin=None, kout=None, ksize=None):
        if kin is None or kout is None or ksize is None:
            residual = self.conv3(x)
            residual = self.bn3(residual)
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)
            out += residual
            out = self.relu(out)
            return out
        padding = ksize // 2
        residual = x
        out1 = F.conv2d(x, self.conv1.weight[:kout, :kin, :ksize, :ksize], padding=padding)
        out1 = F.batch_norm(out1, self.bn1.running_mean[:kout], self.bn1.running_var[:kout], self.bn1.weight[:kout], self.bn1.bias[:kout], training=True)
        out1 = self.relu(out1)
        out1 = F.conv2d(out1, self.conv2.weight[:kout, :kout, :ksize, :ksize], padding=padding)
        out1 = F.batch_norm(out1, self.bn2.running_mean[:kout], self.bn2.running_var[:kout], self.bn2.weight[:kout], self.bn2.bias[:kout], training=True)
        out2 = F.conv2d(residual, self.conv3.weight[:kout, :kin, :, :])
        out2 = F.batch_norm(out2, self.bn3.running_mean[:kout], self.bn3.running_var[:kout], self.bn3.weight[:kout], self.bn3.bias[:kout], training=True)
        out = out1 + out2
        out = self.relu(out)
        return out

class SinglePath_Search(nn.Module):
    def __init__(self, dataset, classes, layers):
        super(SinglePath_Search, self).__init__()
        self.classes = classes
        self.layers = layers
        self.kernel_list = [32, 64, 128]
        self.kernel_size_choices = [1, 3, 5, 7]

        self.fixed_stem = FixedStem()

        self.fixed_block = nn.ModuleList([])
        for i in range(layers):
            layer_cb = nn.ModuleList([
                VGGBlock(in_channels=512, out_channels=128),
                VGGBlock_Nomax(in_channels=512, out_channels=128),
                ResidualBlock(in_channels=512, out_channels=128)
            ])
            self.fixed_block.append(layer_cb)

        self.global_pooling = nn.AdaptiveAvgPool2d((2, 2))
        self.fc1 = nn.Parameter(torch.randn(200, 2 * 2 * 128))
        self.fc2 = nn.Linear(200, 50)
        self.out = nn.Linear(50, self.classes)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.dp1 = nn.Dropout(p=0.2)
        self.dp2 = nn.Dropout(p=0.2)

    def forward(self, x, choice, kchoice, kernel_choice):
        x = self.fixed_stem(x)

        for i, j in enumerate(choice):
            if j == 3:
                kchoice[i] = kchoice[i - 1]
                continue
            ksize = self.kernel_size_choices[kernel_choice[i]]
            if i ==0:
                x = self.fixed_block[i][j](x, 512, self.kernel_list[kchoice[i]], ksize)
            else:
                x = self.fixed_block[i][j](x, self.kernel_list[kchoice[i-1]], self.kernel_list[kchoice[i]], ksize)

        x = self.global_pooling(x)
        x = x.view(-1, 2 * 2 * self.kernel_list[kchoice[self.layers - 1]])
        x = F.linear(x, self.fc1[:, :self.kernel_list[kchoice[self.layers - 1]] * 4])
        x = self.relu1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dp2(x)
        x = self.out(x)

        subnet = self.create_subnet(choice, kchoice, kernel_choice)
        return x, subnet

    def create_subnet(self, choice, kchoice, kernel_choice):
        layers = [('fixed_stem', FixedStem())]
        in_channels = 512

        for i, j in enumerate(choice):
            if j == 3:
                continue
            out_channels = self.kernel_list[kchoice[i]]
            ksize = self.kernel_size_choices[kernel_choice[i]]
            if j == 0:
                block = VGGBlock(in_channels, out_channels, ksize)
            elif j == 1:
                block = VGGBlock_Nomax(in_channels, out_channels, ksize)
            elif j == 2:
                block = ResidualBlock(in_channels, out_channels, ksize)
            else:
                raise ValueError(f"Invalid block type: {j}")
            layers.append((f'layer_{i}_block_{j}', block))
            in_channels = out_channels

        layers += [
            ('global_pool', nn.AdaptiveAvgPool2d((2, 2))),
            ('flatten', nn.Flatten()),
            ('fc1', nn.Linear(2 * 2 * in_channels, 200)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(p=0.2)),
            ('fc2', nn.Linear(200, 50)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(p=0.2)),
            ('output', nn.Linear(50, self.classes))
        ]

        return nn.Sequential(OrderedDict(layers))
