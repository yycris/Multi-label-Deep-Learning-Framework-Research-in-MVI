import torch
import torch.nn as nn
from model import CGA

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-b627a593.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-0676ba61.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-63fe2227.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-394f9c45.pth",
}

cfgs = {
    "resnet": [1, 1, 1, 1],
    "resnet18": [2, 2, 2, 2],
    "resnet34": [3, 4, 6, 3],
    "resnet50": [3, 4, 6, 3],
    "resnet101": [3, 4, 23, 3],
    "resnet152": [3, 8, 36, 3],
}


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1, 1), stride=(stride, stride), bias=False)


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=(3, 3), stride=(stride, stride), padding=dilation,
                     groups=groups, bias=False, dilation=(dilation, dilation))



class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1,
                 norm_layer=None, ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)


        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, in_channels=3, num_classes=1000, zero_init_residual=False, groups=1,
                 width_per_group=64, replace_stride_with_dilation=None, ):
        super(ResNet, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.num = num_classes
        self.inplanes_x = 64
        self.inplanes_y = 64
        self.dilation = 1


        self.groups = groups
        self.base_width = width_per_group
        self.conv1_x = nn.Conv2d(in_channels, self.inplanes_x, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.conv1_y = nn.Conv2d(in_channels, self.inplanes_y, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.bn1_x = norm_layer(self.inplanes_x)
        self.bn1_y = norm_layer(self.inplanes_y)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)


        self.layer1_x = self._make_layer_x(block, 64, layers[0], stride=2)
        self.layer1_y = self._make_layer_y(block, 64, layers[0], stride=2)
        self.sa1 = CGA.Self_Attn(64 * block.expansion)
        self.layer2_x = self._make_layer_x(block, 128, layers[1], stride=2, dilate=False)
        self.layer2_y = self._make_layer_y(block, 128, layers[1], stride=2, dilate=False)
        self.sa2 = CGA.Self_Attn(128 * block.expansion)
        self.layer3_x = self._make_layer_x(block, 256, layers[2], stride=2, dilate=False)
        self.layer3_y = self._make_layer_y(block, 256, layers[2], stride=2, dilate=False)
        self.sa3 = CGA.Self_Attn(256 * block.expansion)
        self.layer4_x = self._make_layer_x(block, 512, layers[3], stride=2, dilate=False)
        self.layer4_y = self._make_layer_y(block, 512, layers[3], stride=2, dilate=False)
        self.sa4 = CGA.Self_Attn(512 * block.expansion)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Flatten()
        self.fc = nn.Linear(512 * block.expansion, self.num)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer_x(self, block, planes, blocks, stride=1, dilate=False, ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes_x != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes_x, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes_x, planes, stride, downsample, self.groups, self.base_width, previous_dilation
            )
        )
        self.inplanes_x = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes_x,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def _make_layer_y(self, block, planes, blocks, stride=1, dilate=False, ):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes_y != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes_y, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.inplanes_y, planes, stride, downsample, self.groups, self.base_width, previous_dilation
            )
        )
        self.inplanes_y = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes_y,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x, y):
        x = self.conv1_x(x)
        x = self.bn1_x(x)
        x = self.relu(x)
        x = self.maxpool(x)

        y = self.conv1_y(y)
        y = self.bn1_y(y)
        y = self.relu(y)
        y = self.maxpool(y)

        x = self.layer1_x(x)
        y = self.layer1_y(y)
        x, y = self.sa1(x, y)

        x = self.layer2_x(x)
        y = self.layer2_y(y)
        x, y = self.sa2(x, y)

        x = self.layer3_x(x)
        y = self.layer3_y(y)
        x, y = self.sa3(x, y)

        x = self.layer4_x(x)
        y = self.layer4_y(y)
        x, y = self.sa4(x, y)

        z = x + y
        z = self.avgpool(z)
        z = self.flatten(z)
        z = self.fc(z)

        return z
def map_keys(k, branch):

    new_key_x = k.replace(k.split('.')[0], f'layer{branch}_x')
    new_key_y = k.replace(k.split('.')[0], f'layer{branch}_y')
    return new_key_x, new_key_y

def resnet(in_channels, num_classes, mode='resnet18', pretrained=False):
    if mode == "resnet18" or mode == "resnet34":
        block = BasicBlock
    else:
        block = Bottleneck
    model = ResNet(block, cfgs[mode], in_channels=in_channels, num_classes=num_classes)
    if pretrained:
        model_dict = model.state_dict()
        model_path = './result/dgresnet18/resnet18-f37072fd.pth'
        state_dict = torch.load(model_path)

        new_pretrained_dict = {}

        for branch in ['1', '2', '3', '4']:
            for k, v in state_dict.items():
                if f'conv1' in k and branch == '1':
                    new_key_x = k.replace(k.split('.')[0], f'conv1_x')
                    new_key_y = k.replace(k.split('.')[0], f'conv1_y')
                    new_pretrained_dict[new_key_x] = v
                    new_pretrained_dict[new_key_y] = v
                if f'layer{branch}' in k:
                    new_key_x, new_key_y = map_keys(k, branch)
                    new_pretrained_dict[new_key_x] = v
                    new_pretrained_dict[new_key_y] = v

        new_pretrained_dict = {k: v for k, v in new_pretrained_dict.items() if k in model_dict}

        if num_classes != 1000:
            num_new_classes = num_classes
            fc_weight = state_dict['fc.weight']
            fc_bias = state_dict['fc.bias']
            fc_weight_new = fc_weight[:num_new_classes, :]
            fc_bias_new = fc_bias[:num_new_classes]
            new_pretrained_dict['fc.weight'] = fc_weight_new
            new_pretrained_dict['fc.bias'] = fc_bias_new

        model_dict.update(new_pretrained_dict)
        model.load_state_dict(model_dict)
    return model

if __name__ == '__main__':
    x = torch.randn(size=(1, 3, 224, 224))
    y = torch.randn(size=(1, 3, 224, 224))
    net = resnet(3, 2, mode='resnet18', pretrained=False)
    print(net)
    out = net(x, y)
    print(out)
    print(out.shape)
