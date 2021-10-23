'''
Properly implemented ResNet-s for CIFAR10 as described in paper [1].

The implementation and structure of this file is hugely influenced by [2]
which is implemented for ImageNet and doesn't have option A for identity.
Moreover, most of the implementations on the web is copy-paste from
torchvision's resnet and has wrong number of params.

Proper ResNet-s for CIFAR10 (for fair comparision and etc.) has following
number of layers and parameters:

name      | layers | params
ResNet20  |    20  | 0.27M
ResNet32  |    32  | 0.46M
ResNet44  |    44  | 0.66M
ResNet56  |    56  | 0.85M
ResNet110 |   110  |  1.7M
ResNet1202|  1202  | 19.4m

which this implementation indeed has.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

If you use this implementation in you work, please don't forget to mention the
author, Yerlan Idelbayev.
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch.autograd import Variable
from models.quan_conv import QuanConv as Conv
from models.quan_conv import QuanConv_first as Conv_first
from models.quan_conv import QuanLinear as QuanLinear

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

def conv3x3_first(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv_first(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def Bi_Linear(in_features, out_features):
    return QuanLinear(in_features, out_features, bias=False)

class BasicBlock_bi(nn.Module):
    expansion = 1

    def __init__(self, planes, stride=1, option='A'):
        super(BasicBlock_bi, self).__init__()
        self.conv1 = conv3x3(planes[0], planes[1], stride=stride)
        self.bn1 = nn.BatchNorm2d(planes[1])
        self.conv2 = conv3x3(planes[1], planes[2], stride=1)
        self.bn2 = nn.BatchNorm2d(planes[2])
        self.shortcut = nn.Sequential()

        #if stride != 1 or in_planes != planes:
        if option == 'A':
            """
            For CIFAR10 ResNet paper uses option A.
            """
            if stride ==2:
                if planes[2] > planes[0] :
                    self.shortcut  = LambdaLayer(lambda x:
                                        F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, 0,
                                            planes[2] -planes[0]),"constant", 0))
                else:
                    self.shortcut = LambdaLayer(lambda x: x[:, :planes[2] , ::2, ::2])
            elif stride==1:
                if planes[2]  > planes[0]:
                    self.shortcut = LambdaLayer(lambda x:
                                        F.pad(x[:, :, :, :], (0, 0, 0, 0, 0,
                                            planes[2] -planes[0]),"constant", 0))
                else:
                    self.shortcut = LambdaLayer(lambda x: x[:, :planes[2] , :, :])
        elif option == 'B':
            self.shortcut = nn.Sequential(
                    conv1x1(in_planes, self.expansion * planes,stride=stride),
                    nn.BatchNorm2d(self.expansion * planes)
            )


    def forward(self, x):
        out = self.bn1(self.conv1(x))
        
        out = (self.conv2(out))
        out += self.shortcut(x)
        out = self.bn2(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        #self.out_planes = [16,16,16,16,16,16,16,32,\
        # 32,32,32,32,32,64,64,64,64,64,64]

        #HAWIS-A
        # self.out_planes = [64, 80, 80, 80, 80, 80, 80, 112, 112, 112, \
        #   112, 112, 112, 216, 216, 216, 216, 208, 216]

        #HAWIS-B
        self.out_planes =[84,84,84,84,84,80,84, 168,168,192,224,188,224,\
             284,284, 256,248,224,224]

        self.in_planes = self.out_planes[0]

        self.conv1 = conv3x3_first(int(3*32), self.in_planes, stride=1)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.out_planes[0:7], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, self.out_planes[6:13], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, self.out_planes[12:], num_blocks[2], stride=2)
        self.bn2 = nn.BatchNorm1d(self.out_planes[-1])
        self.relu = nn.ReLU(inplace=True)
        self.linear = Bi_Linear(self.out_planes[-1], num_classes)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) :
                init.kaiming_normal(m.weight)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(block(planes[i*2+0: i*2+3], stride))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.relu(out)

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.bn2(out)

        out = self.linear(out)
        return out


def resnet20():
    return ResNet(BasicBlock_bi, [3, 3, 3])

# def resnet32():
#     return ResNet(BasicBlock, [5, 5, 5])


# def resnet44():
#     return ResNet(BasicBlock, [7, 7, 7])


# def resnet56():
#     return ResNet(BasicBlock, [9, 9, 9])


# def resnet110():
#     return ResNet(BasicBlock, [18, 18, 18])


# def resnet1202():
#     return ResNet(BasicBlock, [200, 200, 200])


def test(net):
    import numpy as np
    total_params = 0

    for x in filter(lambda p: p.requires_grad, net.parameters()):
        total_params += np.prod(x.data.numpy().shape)
    print("Total number of params", total_params)
    print("Total layers", len(list(filter(lambda p: p.requires_grad and len(p.data.size())>1, net.parameters()))))


if __name__ == "__main__":
    for net_name in __all__:
        if net_name.startswith('resnet'):
            print(net_name)
            test(globals()[net_name]())
            print()
