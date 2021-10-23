import torch
import torch.nn as nn
import torchvision

from python.torx.module_Binary.layer import crxb_Conv2d
from python.torx.module_Binary.layer import crxb_Linear

### 相比于v2,最后一层全连接前不使用pooling，加宽全连接的输入数量。

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


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, midplanes, planes, 
                    crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise,
                    enable_SAF, enable_ec_SAF,enable_resistance_variance, resistance_variance_gamma,
                    stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = crxb_Conv2d(inplanes, midplanes, kernel_size=3, stride=stride, padding=1, bias=False,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_resistance_variance=enable_resistance_variance,
                                 resistance_variance_gamma=resistance_variance_gamma,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.bn1 = nn.BatchNorm2d(midplanes)
        self.conv2 = crxb_Conv2d(midplanes, planes, kernel_size=3, stride=1, padding=1, bias=False,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_resistance_variance=enable_resistance_variance,
                                 resistance_variance_gamma=resistance_variance_gamma,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)

        out = self.conv2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.bn2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers,
                crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise,
                 enable_SAF, enable_ec_SAF, enable_resistance_variance, resistance_variance_gamma,   
                    num_classes=1000):
        super(ResNet, self).__init__()
        #self.planes = [64,64,64,64,64, 128,128,128,128, 256,256,256,256, 512,512,512,512]   
        #HAWIS-B
        self.planes =[256, 256,256,320,320,  504,504,560,504, 1064, 1152, 1064, 1152, 1792, 1788, 1788,1780] 
        self.inplanes = self.planes[0]

        self.conv1_1 = crxb_Conv2d(int(3*32), int(3*32), kernel_size=7, stride=2, padding=3, bias=False,
                        crxb_size=crxb_size, scaler_dw=scaler_dw,
                        gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                        enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                        enable_noise=enable_noise, ir_drop=ir_drop, device=device)

        self.bn1_1 = nn.BatchNorm2d(int(3*32))
        
        self.conv1_2 = crxb_Conv2d(int(3*32), self.inplanes, kernel_size=3, stride=2, padding=1, bias=False,
                        crxb_size=crxb_size, scaler_dw=scaler_dw,
                        gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                        enable_resistance_variance=enable_resistance_variance,
                        resistance_variance_gamma=resistance_variance_gamma,
                        enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                        enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.conv1_2_r = crxb_Conv2d(int(3*32), self.inplanes, kernel_size=1, stride=1, padding=0, bias=False,
                        crxb_size=crxb_size, scaler_dw=scaler_dw,
                        gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                        enable_resistance_variance=enable_resistance_variance,
                        resistance_variance_gamma=resistance_variance_gamma,
                        enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                        enable_noise=enable_noise, ir_drop=ir_drop, device=device)
        self.bn1_2 = nn.BatchNorm2d(self.inplanes)
        
        self.avgpool_1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool_2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, self.planes[0:5], layers[0],  stride=1,
                    crxb_size=crxb_size, gmin=gmin, gmax=gmax, gwire=gwire, gload=gload, vdd=vdd, 
                    ir_drop=ir_drop, freq=freq, temp=temp, device=device, scaler_dw=scaler_dw,
                    enable_resistance_variance=enable_resistance_variance,
                        resistance_variance_gamma=resistance_variance_gamma, 
                    enable_noise=enable_noise,enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF)
        self.layer2 = self._make_layer(block, self.planes[4:9], layers[1],  stride=2,
                    crxb_size=crxb_size, gmin=gmin, gmax=gmax, gwire=gwire, gload=gload, vdd=vdd, 
                    ir_drop=ir_drop, freq=freq, temp=temp, device=device, scaler_dw=scaler_dw,
                    enable_resistance_variance=enable_resistance_variance,
                        resistance_variance_gamma=resistance_variance_gamma, 
                    enable_noise=enable_noise,enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF)
        self.layer3 = self._make_layer(block, self.planes[8:13], layers[2], stride=2,
                    crxb_size=crxb_size, gmin=gmin, gmax=gmax, gwire=gwire, gload=gload, vdd=vdd, 
                    ir_drop=ir_drop, freq=freq, temp=temp, device=device, scaler_dw=scaler_dw,
                    enable_resistance_variance=enable_resistance_variance,
                        resistance_variance_gamma=resistance_variance_gamma, 
                    enable_noise=enable_noise,enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF)
        self.layer4 = self._make_layer(block, self.planes[12:], layers[3],  stride=2,
                    crxb_size=crxb_size, gmin=gmin, gmax=gmax, gwire=gwire, gload=gload, vdd=vdd, 
                    ir_drop=ir_drop, freq=freq, temp=temp, device=device, scaler_dw=scaler_dw,
                    enable_resistance_variance=enable_resistance_variance,
                        resistance_variance_gamma=resistance_variance_gamma, 
                    enable_noise=enable_noise,enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF)

        self.fc = crxb_Linear(in_features=int(self.planes[-1]*7*7), out_features=num_classes, bias=False,
                               crxb_size=crxb_size, scaler_dw=scaler_dw,
                               gmax=gmax, gmin=gmin, gwire=gwire, gload=gload, freq=freq, temp=temp,
                               enable_resistance_variance=enable_resistance_variance,
                                resistance_variance_gamma=resistance_variance_gamma,
                               vdd=vdd, ir_drop=ir_drop, device=device, enable_noise=enable_noise,
                               enable_ec_SAF=enable_ec_SAF, enable_SAF=enable_SAF)

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

    def _make_layer(self, block, planes, blocks, stride,
                crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise,
                 enable_SAF, enable_ec_SAF, enable_resistance_variance, resistance_variance_gamma):
        in_planes, mid_planes, out_planes = planes[0],planes[1],planes[2]

        downsample = None
        downsample1 = nn.Sequential(
            crxb_Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_resistance_variance=enable_resistance_variance,
                                 resistance_variance_gamma=resistance_variance_gamma,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device),
            nn.BatchNorm2d(out_planes),
            )
        layers = []
        layers.append(block(in_planes, mid_planes, out_planes, stride=stride, downsample=downsample1, 
                        crxb_size=crxb_size, gmin=gmin, gmax=gmax, gwire=gwire, gload=gload, vdd=vdd, 
                        ir_drop=ir_drop, freq=freq, temp=temp, device=device, scaler_dw=scaler_dw, 
                        enable_resistance_variance=enable_resistance_variance,
                                 resistance_variance_gamma=resistance_variance_gamma,
                        enable_noise=enable_noise,enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF))
        
        for i in range(1, blocks):
            in_planes, mid_planes, out_planes = planes[i*2+0],planes[i*2+1],planes[i*2+2]
            downsample_i = nn.Sequential(
                crxb_Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False,
                                 crxb_size=crxb_size, scaler_dw=scaler_dw,
                                 gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                                 enable_resistance_variance=enable_resistance_variance,
                                 resistance_variance_gamma=resistance_variance_gamma,
                                 enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                                 enable_noise=enable_noise, ir_drop=ir_drop, device=device),
                nn.BatchNorm2d(out_planes),
            )
            layers.append(block(in_planes, mid_planes, out_planes, stride=1, downsample=downsample_i, 
                        crxb_size=crxb_size, gmin=gmin, gmax=gmax, gwire=gwire, gload=gload, vdd=vdd, 
                        ir_drop=ir_drop, freq=freq, temp=temp, device=device, scaler_dw=scaler_dw,
                        enable_resistance_variance=enable_resistance_variance,
                                 resistance_variance_gamma=resistance_variance_gamma, 
                        enable_noise=enable_noise,enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF))


        return nn.Sequential(*layers)

    def forward(self, x):
        '''Denormalize input images x'''
        x_3c = x*self.img_std + self.img_mean
        
        # x_3c: [N, C, W, H]
        Ther_len = int(32)
        x_3c = x_3c.repeat(1,Ther_len,1,1)
        x = (x_3c > self.flag_matrix_RGB).float()

        x = self.bn1_1( self.conv1_1(x) +self.avgpool_1(x))
        
        x = self.bn1_2( self.conv1_2(x) + self.avgpool_2(self.conv1_2_r(x)))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet18(crxb_size, gmin, gmax, gwire, gload, vdd, ir_drop, freq, temp, device, scaler_dw, enable_noise,
                 enable_SAF, enable_ec_SAF,enable_resistance_variance, resistance_variance_gamma):
    model = ResNet(BasicBlock, [2, 2, 2, 2],
                    crxb_size=crxb_size, scaler_dw=scaler_dw,
                    gwire=gwire, gload=gload, gmax=gmax, gmin=gmin, vdd=vdd, freq=freq, temp=temp,
                    enable_resistance_variance=enable_resistance_variance,
                    resistance_variance_gamma=resistance_variance_gamma, 
                    enable_SAF=enable_SAF, enable_ec_SAF=enable_ec_SAF,
                    enable_noise=enable_noise, ir_drop=ir_drop, device=device)
    return model


