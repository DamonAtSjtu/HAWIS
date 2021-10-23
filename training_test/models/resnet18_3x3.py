import torch
import torch.nn as nn
import torchvision
from models.quan_conv import QuanConv as Conv
from models.quan_conv import QuanConv_first as Conv_first
from models.quan_conv import QuanLinear as QuanLinear
from models.quan_conv import QuanLinear_W_multibit as QuanLinear_W_multibit
from models.quan_conv import QuanLinear_A_multibit as QuanLinear_A_multibit


__all__ = ['ResNet', 'resnet18']


def Flag_Matrix_RGB():
    W = H = 224
    Ther_len = int(32)
    Ther_advisor = int(256/Ther_len)
    flag_matrix = torch.zeros(int(Ther_len*3), W, H )

    for i in range(Ther_len):
        flag_matrix[ int(i*3):int(i*3+3), :, :] = int( (i+0.5)*Ther_advisor )
    flag_matrix /= 255
    return flag_matrix

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

def conv7x7(in_planes, out_planes, stride=1):
    """7x7 convolution with padding"""
    return Conv(in_planes, out_planes, 7, stride=stride, padding=3, bias=False)
def conv5x5(in_planes, out_planes, stride=1):
    """7x7 convolution with padding"""
    return Conv(in_planes, out_planes, 5, stride=stride, padding=2, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return Conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

# def conv3x3_first(in_planes, out_planes, stride=1):
#     """3x3 convolution with padding"""
#     return Conv_first(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
def conv7x7_first(in_planes, out_planes, stride=1):
    """7x7 convolution with padding"""
    return Conv_first(in_planes, out_planes, kernel_size=7, stride=stride, padding=3, bias=False)

def conv5x5_first(in_planes, out_planes, stride=1):
    """5x5 convolution with padding"""
    return Conv_first(in_planes, out_planes, kernel_size=5, stride=stride, padding=2, bias=False)

def conv3x3_first(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return Conv_first(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def Bi_Linear(in_features, out_features):
    return QuanLinear(in_features, out_features, bias=False)

def Multibit_W_Linear(in_features, out_features, multibit=4):
    return QuanLinear_W_multibit(in_features, out_features, multibit=multibit, bias=False)

def Multibit_A_Linear(in_features, out_features, multibit=4):
    return QuanLinear_A_multibit(in_features, out_features, multibit=multibit, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, midplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, midplanes, stride)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = conv3x3(midplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)
        ###out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.bn2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()

        #self.planes = [64,64,64,64,64, 128,128,128,128, 256,256,256,256, 512,512,512,512]
        #HAWIS-B
        self.planes =[256, 256,256,320,320,  504,504,560,504, 1064, 1152, 1064, 1152, 1792, 1788, 1788,1780] 

        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                        bias=False)
        self.inplanes = self.planes[0]
        self.conv1_1 = conv3x3_first(int(3*32), 192, stride=2)
        self.conv1_1_r = conv1x1(int(3*32), 192, stride=1)
        self.bn1_1 = nn.BatchNorm2d(192)
        self.conv1_2 = conv3x3(192, self.inplanes, stride=2)
        self.conv1_2_r = conv1x1(192, self.inplanes, stride=1)
        self.bn1_2 = nn.BatchNorm2d(self.inplanes)
        
        self.avgpool_1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool_2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)


        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.planes[0:5], layers[0],  stride=1)
        self.layer2 = self._make_layer(block, self.planes[4:9], layers[1],  stride=2)
        self.layer3 = self._make_layer(block, self.planes[8:13], layers[2], stride=2)
        self.layer4 = self._make_layer(block, self.planes[12:], layers[3],  stride=2)

        self.fc = Bi_Linear(int(self.planes[-1]*7*7), num_classes) 

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.register_buffer(
            'img_mean', 
            torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)
        )
        self.register_buffer(
            'img_std',
            torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)
        )

        flag_matrix_RGB = Flag_Matrix_RGB()
        self.flag_matrix_RGB = flag_matrix_RGB.cuda()

        self.teacher = torchvision.models.resnet50(pretrained=True)
        for param in self.teacher.parameters():
            param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1):
        in_planes, mid_planes, out_planes = planes[0],planes[1],planes[2]

        downsample = None
        #if stride != 1 or self.inplanes != planes:
        downsample1 = nn.Sequential(
            conv1x1(in_planes, out_planes, stride=stride),
            nn.BatchNorm2d(out_planes),
            )
        layers = []
        layers.append(block(in_planes, mid_planes, out_planes, stride, downsample1))
        
        for i in range(1, blocks):
            in_planes, mid_planes, out_planes = planes[i*2+0],planes[i*2+1],planes[i*2+2]
            downsample_i = nn.Sequential(
                conv1x1(in_planes, out_planes, stride=1),
                nn.BatchNorm2d(out_planes),
            )
            layers.append(block(in_planes, mid_planes, out_planes, 1, downsample_i))

        return nn.Sequential(*layers)

    def forward(self, x):
        lesson = self.teacher(x).detach()
        '''Denormalize input images x'''
        x_3c = x*self.img_std + self.img_mean
        
        # x_3c: [N, C, W, H]
        Ther_len = int(32)
        x_3c = x_3c.repeat(1,Ther_len,1,1)
        x = (x_3c > self.flag_matrix_RGB).float()

        x = self.bn1_1( self.conv1_1(x) +self.avgpool_1(self.conv1_1_r(x)))
        x = self.bn1_2( self.conv1_2(x) + self.avgpool_2(self.conv1_2_r(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.relu(x)
        #print(x.shape)

        #x = self.avgpool(x)
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.fc(x)

        return x, lesson


def resnet18_3x3(**kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    return model


