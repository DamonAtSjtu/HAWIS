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
from python.torx.module_Binary.layer import crxb_Conv2d
from python.torx.module_Binary.layer import crxb_Linear

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlock_bi(nn.Module):
    expansion = 1

    def __init__(self, planes, 
                 crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise, 
                 enable_resistance_variance, resistance_variance_gamma, enable_SAF, enable_ec_SAF,
                 stride=1, option='A'):
        super(BasicBlock_bi, self).__init__()
        self.conv1 = crxb_Conv2d(planes[0], planes[1], kernel_size=3, stride=stride, padding=1,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_SAF=enable_SAF, enable_resistance_variance=enable_resistance_variance,
                                 enable_ec_SAF=enable_ec_SAF, resistance_variance_gamma=resistance_variance_gamma,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.bn1 = nn.BatchNorm2d(planes[1])
        self.conv2 = crxb_Conv2d(planes[1], planes[2], kernel_size=3, stride=1, padding=1,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, enable_resistance_variance=enable_resistance_variance,
                                 resistance_variance_gamma=resistance_variance_gamma,
                                 ir_drop=ir_drop, device=device)
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

    def forward(self, x):
        out = self.bn1(self.conv1(x))
 
        out = (self.conv2(out))
        out += self.shortcut(x)
        out = self.bn2(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, 
                crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise,
                 enable_resistance_variance, resistance_variance_gamma,enable_SAF, enable_ec_SAF,
                 num_classes=10):
        super(ResNet, self).__init__()

        #HAWIS-A
        self.out_planes = [64, 80, 80, 80, 80, 80, 80, 112, 112, 112, \
          112, 112, 112, 216, 216, 216, 216, 208, 216]

        #HAWIS-B
        # self.out_planes =[84,84,84,84,84,80,84, 168,168,192,224,188,224,\
        #      284,284, 256,248,224,224]
        
        self.in_planes = self.out_planes[0]

        self.conv1 = crxb_Conv2d(96, self.in_planes, kernel_size=3, stride=1, padding=1,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, enable_resistance_variance=enable_resistance_variance,
                                 resistance_variance_gamma=resistance_variance_gamma, ir_drop=ir_drop, device=device)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, self.out_planes[0:7], num_blocks[0], stride=1,
                    crxb_size=crxb_size, gmin=gmin, gmax=gmax, gwire=gwire, gload=gload, vdd=vdd, 
                    ir_drop=ir_drop, freq=freq, temp=temp, device=device, scaler_dw=scaler_dw, 
                    enable_noise=enable_noise, enable_resistance_variance=enable_resistance_variance,
                    resistance_variance_gamma=resistance_variance_gamma, 
                    enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF)
        self.layer2 = self._make_layer(block, self.out_planes[6:13], num_blocks[1], stride=2,
                    crxb_size=crxb_size, gmin=gmin, gmax=gmax, gwire=gwire, gload=gload, vdd=vdd, 
                    ir_drop=ir_drop, freq=freq, temp=temp, device=device, scaler_dw=scaler_dw, 
                    enable_noise=enable_noise, enable_resistance_variance=enable_resistance_variance,
                    resistance_variance_gamma=resistance_variance_gamma,
                    enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF)
        self.layer3 = self._make_layer(block, self.out_planes[12:], num_blocks[2], stride=2,
                    crxb_size=crxb_size, gmin=gmin, gmax=gmax, gwire=gwire, gload=gload, vdd=vdd, 
                    ir_drop=ir_drop, freq=freq, temp=temp, device=device, scaler_dw=scaler_dw, 
                    enable_noise=enable_noise, enable_resistance_variance=enable_resistance_variance,
                    resistance_variance_gamma=resistance_variance_gamma,
                    enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF)

        self.bn2 = nn.BatchNorm1d(self.out_planes[-1])
        self.relu = nn.ReLU(inplace=True)
        self.linear = crxb_Linear(in_features=self.out_planes[-1], out_features=num_classes, 
                               crxb_size=crxb_size, scaler_dw=scaler_dw,
                               gmax=gmax, gmin=gmin, gwire=gwire, gload=gload, freq=freq, temp=temp,
                               vdd=vdd, ir_drop=ir_drop, device=device, enable_noise=enable_noise,
                               enable_resistance_variance=enable_resistance_variance,
                               resistance_variance_gamma=resistance_variance_gamma,
                               enable_ec_SAF=enable_ec_SAF, enable_SAF=enable_SAF)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) :
                init.kaiming_normal(m.weight)


    def _make_layer(self, block, planes, num_blocks, stride,
                crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise,
                 enable_resistance_variance, resistance_variance_gamma, enable_SAF, enable_ec_SAF):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for i,stride in enumerate(strides):
            layers.append(block(planes[i*2+0: i*2+3], stride=stride,
                        crxb_size=crxb_size, gmin=gmin, gmax=gmax, gwire=gwire, gload=gload, vdd=vdd, 
                        ir_drop=ir_drop, freq=freq, temp=temp, device=device, scaler_dw=scaler_dw, 
                        enable_noise=enable_noise, enable_resistance_variance=enable_resistance_variance,
                        resistance_variance_gamma=resistance_variance_gamma,
                        enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF))
            #self.in_planes = planes * block.expansion

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


def resnet20(crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise,
                 enable_resistance_variance, resistance_variance_gamma, enable_SAF, enable_ec_SAF):
    return ResNet(BasicBlock_bi, [3, 3, 3], num_classes=10,
                    crxb_size=crxb_size, scaler_dw=scaler_dw,
                    gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                    enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                    enable_noise=enable_noise, enable_resistance_variance=enable_resistance_variance,
                    resistance_variance_gamma=resistance_variance_gamma, ir_drop=ir_drop, device=device)


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
