import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.modules.batchnorm import _BatchNorm

import torchvision.models as models
from .ResNet import ResNet50
import torchvision
from functools import partial
from .global_reasoning_unit import GloRe_Unit
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class ZPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class AttentionGate(nn.Module):
    def __init__(self):
        super(AttentionGate, self).__init__()
        kernel_size = 7
        self.compress = ZPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.conv(x_compress)
        scale = torch.sigmoid_(x_out) 
        return x * scale

class TripletAttention(nn.Module):
    def __init__(self, no_spatial=False):
        super(TripletAttention, self).__init__()
        self.cw = AttentionGate()
        self.hc = AttentionGate()
        self.no_spatial=no_spatial
        if not no_spatial:
            self.hw = AttentionGate()
    def forward(self, x):
        x_perm1 = x.permute(0,2,1,3).contiguous()
        x_out1 = self.cw(x_perm1)
        x_out11 = x_out1.permute(0,2,1,3).contiguous()
        x_perm2 = x.permute(0,3,2,1).contiguous()
        x_out2 = self.hc(x_perm2)
        x_out21 = x_out2.permute(0,3,2,1).contiguous()
        if not self.no_spatial:
            x_out = self.hw(x)
            x_out = 1/3 * (x_out + x_out11 + x_out21)
        else:
            x_out = 1/2 * (x_out11 + x_out21)
        return x_out

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.residual = None
        if in_channels != out_channels:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
                nn.BatchNorm2d(out_channels))
        else:
            self.residual = None




    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.residual is not None:
            identity = self.residual(x)

        out += identity
        out = self.relu(out)

        return out


        
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )
        
    def forward(self, x):
        x = self.conv_bn(x)
        return x


class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):
        return self.reduce(x)


class conv_upsample(nn.Module):
    def __init__(self, channel):
        super(conv_upsample, self).__init__()
        self.conv = BasicConv2d(channel, channel, 1)

    def forward(self, x, target):
        if x.size()[2:] != target.size()[2:]:
            x = self.conv(F.interpolate(x, size=target.size()[2:], mode='bilinear', align_corners=True))
        return x


class DenseFusion(nn.Module):
    # Cross Refinement Unit
    def __init__(self, channel):
        super(DenseFusion, self).__init__()
        self.conv1 = conv_upsample(channel)
        self.conv2 = conv_upsample(channel)
        self.conv3 = conv_upsample(channel)
        self.conv4 = conv_upsample(channel)
        self.conv5 = conv_upsample(channel)
        self.conv6 = conv_upsample(channel)
        self.conv7 = conv_upsample(channel)
        self.conv8 = conv_upsample(channel)
        self.conv9 = conv_upsample(channel)
        self.conv10 = conv_upsample(channel)
        self.conv11 = conv_upsample(channel)
        self.conv12 = conv_upsample(channel)

        self.conv_f1 = nn.Sequential(
            BasicConv2d(2*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f2 = nn.Sequential(
            BasicConv2d(2*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f3 = nn.Sequential(
            BasicConv2d(2*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f4 = nn.Sequential(
            BasicConv2d(2*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

        self.conv_f5 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f6 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f7 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f8 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )

    def forward(self, x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4):
        
        x_sf1 = x_s1 + self.conv_f1(torch.cat((x_s1, x_e1),1))
        x_sf2 = x_s2 + self.conv_f2(torch.cat((x_s2,x_e2),1))
        x_sf3 = x_s3 + self.conv_f3(torch.cat((x_s3, x_e3), 1))
        x_sf4 = x_s4 + self.conv_f4(torch.cat((x_s4, x_e4), 1))
        '''
        x_sf1 = x_s1 + self.conv_f1(torch.cat((x_s1, x_e1,
                                               self.conv1(x_e2, x_s1),
                                               self.conv2(x_e3, x_s1),
                                               self.conv3(x_e4, x_s1)), 1))
        x_sf2 = x_s2 + self.conv_f2(torch.cat((x_s2, x_e2,
                                               self.conv4(x_e3, x_s2),
                                               self.conv5(x_e4, x_s2)), 1))
        x_sf3 = x_s3 + self.conv_f3(torch.cat((x_s3, x_e3,
                                               self.conv6(x_e4, x_s3)), 1))
        x_sf4 = x_s4 + self.conv_f4(torch.cat((x_s4, x_e4), 1))
        '''
        return x_sf1, x_sf2, x_sf3, x_sf4


class ConcatOutput(nn.Module):
    def __init__(self, channel):
        super(ConcatOutput, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2*channel, 2*channel, 3, padding=1),
            BasicConv2d(2*channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2*channel, 2*channel, 3, padding=1),
            BasicConv2d(2*channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2*channel, 2*channel, 3, padding=1),
            BasicConv2d(2*channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x1, x2, x3, x4):
        #print(self.conv_upsample1(self.upsample(x4)).shape)
        #print(x3.shape)
        #print(x2.shape)
        x3 = torch.cat((x3, self.conv_upsample1(self.upsample(x4))), 1)
        #print(x3.shape)
        x3 = self.conv_cat1(x3)
        
        #print(self.upsample(x3).shape)
        #print(x2.shape)
        x2 = torch.cat((x2, self.upsample(x3)), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1, self.upsample(x2)), 1)
       
        x1 = self.conv_cat3(x1)
        #x = self.output(x1)
        return x1
class ConcatOutput1(nn.Module):
    def __init__(self, channel):
        super(ConcatOutput1, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)

        self.conv_cat1 = nn.Sequential(
            BasicConv2d(2*channel, 2*channel, 3, padding=1),
            BasicConv2d(2*channel, channel, 1)
        )
        self.conv_cat2 = nn.Sequential(
            BasicConv2d(2*channel, 2*channel, 3, padding=1),
            BasicConv2d(2*channel, channel, 1)
        )
        self.conv_cat3 = nn.Sequential(
            BasicConv2d(2*channel, 2*channel, 3, padding=1),
            BasicConv2d(2*channel, channel, 1)
        )
        self.output = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x1, x2, x3, x4):
        x3 = torch.cat((x3, self.conv_upsample1(self.upsample(x4))), 1)
        x3 = self.conv_cat1(x3)

        x2 = torch.cat((x2, self.conv_upsample2(self.upsample(x3))), 1)
        x2 = self.conv_cat2(x2)

        x1 = torch.cat((x1, self.conv_upsample3(self.upsample(x2))), 1)
        x1 = self.conv_cat3(x1)

        x = self.output(x1)
        return x

class SCRN(nn.Module):
    # Stacked Cross Refinement Network
    def __init__(self, channel=32,stages=3,blocks=[2,3,3], n_classes=2):
        super(SCRN, self).__init__()
        self.resnet = ResNet50()
        self.reduce_s1 = Reduction(256, channel)
        self.reduce_s2 = Reduction(512, channel)
        self.reduce_s3 = Reduction(1024, channel)
        self.reduce_s4 = Reduction(2048, channel)
        self.block = BasicBlock
        self.mp = nn.Conv2d(2, 1, (1, 1), (1,1))
        self.blocks = blocks
        self.reduce_e1 = Reduction(256, channel)
        self.reduce_e2 = Reduction(512, channel)
        self.reduce_e3 = Reduction(1024, channel)
        self.reduce_e4 = Reduction(2048, channel)
        #self.reasoning2 = GloRe_Unit(1, 1)
        #self.reasoning3 = GloRe_Unit(1024,512)
        #self.RGB_encoder = self._make_encoder(in_channels=32, out_channels=32, stages=stages, blocks=blocks)
        self.RGB_decoder = self._make_decoder(in_channels=32, stages=2, n_classes=n_classes)
        #self.emau = EMAU(32, 16,3)
        #self.sknet=  SKConv(32, 8, 3, 8, 2)
        #self.reasoning1 = GloRe_Unit(256, 4)
        #self.reasoning2 = GloRe_Unit(512, 4)
        #self.reasoning3 = GloRe_Unit(1024, 4)
        #self.reasoning5 = GloRe_Unit(2048, 4)
        #self.reasoning4 = GloRe_Unit(32, 4)
        #self.triplet1=  TripletAttention(256)
        #self.triplet2=  TripletAttention(512)
        #self.triplet3=  TripletAttention(1024)
        #self.triplet4=  TripletAttention(2048)
        #self.triplet=  TripletAttention(32)
        #self.df1 = DenseFusion(channel)
        #self.df2 = DenseFusion(channel)
        #self.df3 = DenseFusion(channel)
        #self.df4 = DenseFusion(channel)
        #self.conv_f4 = nn.Sequential(
            #BasicConv2d(2*channel, channel, 3, padding=1),
            #BasicConv2d(channel, channel, 3, padding=1)
        #)
        self.output_s = ConcatOutput(channel)
        self.output_e = ConcatOutput1(channel)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        self.initialize_weights()
    def _make_stage(self, block, in_channels, out_channels, stage_blocks):
        '''创建一个阶段(stage)
            args:
                block: 基础块
                in_channels: 输入通道数
                out_channels: 输出通道数
                stage_blocks: 阶段内含有的基础块个数
        '''
        layers = nn.ModuleList()
        for i in range(stage_blocks):
            #layers.append([block(in_channels, out_channels), self.parameters[self.parameter_index+2]]) # 将每层绑定一个融合参数
            layers.append(block(in_channels, out_channels))
            in_channels = out_channels
            
        return layers
    def _make_decoder(self, in_channels, stages=2, n_classes=2):
        '''构造Decoder
            args:
                in_channels: 输入通道数
                stages: Encoder中含有阶段的个数
        '''
        unsamples = stages # unsamples: Decoder的上采样次数
        decoder = nn.ModuleList()
        
        in_channels = in_channels 
        for i in range(unsamples-1):
            unsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=in_channels//2, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(in_channels//2))
            decoder.append(unsample)
            in_channels //= 2 
        unsample = nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, out_channels=n_classes, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(n_classes))
        
        decoder.append(unsample)
        
        return decoder   
    def forward(self, x):
        size = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        #如果后面减小参数量的话，可以写成resnet(x)
        #也可以让Resnet返回四个值
        x1 = self.resnet.layer1(x)
        #x1=self.triplet1(x1)
        x2 = self.resnet.layer2(x1)
        #x2 = self.reasoning2(x2)
        #x2=self.triplet2(x2)
        x3 = self.resnet.layer3(x2)
        #x3=self.triplet3(x3)
        #x3 = self.reasoning3(x3)
        x4 = self.resnet.layer4(x3)
        #x4=self.triplet1(x4)
        #x4 = self.reasoning5(x4)
         # feature abstraction
        x_e1 = self.reduce_e1(x1)
        x_e2 = self.reduce_e2(x2)
        x_e3 = self.reduce_e3(x3)
        x_e4 = self.reduce_e4(x4)
        x_s1 = self.reduce_s1(x1)
        x_s2 = self.reduce_s2(x2)
        x_s3 = self.reduce_s3(x3)
        x_s4 = self.reduce_s4(x4)
        #print(x_s4.shape)
        #x1,x2,x3,x4 = self.df1(x1, x2, x3, x4, x_e1, x_e2, x_e3, x_e4)
        x_s1,x_s2,x_s3,x_s4 = self.df2(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1,x_s2,x_s3,x_s4 = self.df2(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1,x_s2,x_s3,x_s4 = self.df3(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1,x_s2,x_s3,x_s4 = self.df4(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        #print(x4.shape)
        #print(x3.shape)
        #print(x2.shape)
        #print(x1.shape)
        #x_s4 = self.output_m(x1, x2, x3, x4)
        #print(x4.shape)
        #print(x_s13.shape)
        #print(x_s12.shape)
        #print(x_s4.shape)
        #x_s4= self.sknet(x_s4)
        #print(x_s11.shape)
        x_s4 = self.output_s(x_s1, x_s2, x_s3, x_s4)
        #print(x_s4.shape)
        #x_s4= self.sknet(x_s4)
        x_s4 = self.reasoning4(x_s4)
        
        #x_s4 = x_s4 + self.conv_f4(torch.cat((x_s4, x_e4), 1))
        #print(x_s4.shape)
        RGB =x_s4
        #print(x_s4.shape)
        '''
        RGB_encoder = self.RGB_encoder
        for RGB_stage in RGB_encoder:
            RGB_stage_features = []
            for  RGB_layer in RGB_stage[0]:
                #print(RGB.shape)
                #RGB = RGB_layer(RGB)
                RGB_stage_features.append(RGB)
               
               
            # maxpool
            #print(RGB.shape)
            #RGB = RGB_stage[1](RGB)
            if(RGB.shape[1]==64):
                #print("s1")
                RGB1=RGB
            elif(RGB.shape[1]==128):
                RGB2=RGB
        '''
        ## RGB Decoder
        RGB_decoder = self.RGB_decoder
        
        for RGB_unsample in RGB_decoder:
          
            RGB_feature = RGB_unsample(RGB)
            RGB = RGB_feature
            #skip connection
            #print(RGB.shape)
            #if(RGB.shape[1]==64):
                #print("sss")
                #m=nn.Upsample(scale_factor=2, mode='nearest')
                #RGB1=m(RGB1)
                #RGB=RGB+RGB1
            #if(RGB.shape[1]==128):
                #print("yyy")
                #m=nn.Upsample(scale_factor=2, mode='nearest')
                #RGB1=m(RGB1)
                #RGB2=m(RGB2)
                #RGB=RGB+RGB2
            #print(RGB.shape)    
        #m=nn.Upsample(scale_factor=2, mode='nearest')
        #RGB_feature = m(RGB)
        #RGB_feature = m(RGB_feature)
        output =  self.mp(RGB)
        #print(output.shape)
       
        #output = self.output_s(x_s1, x_s2, x_s3, x_s4)
        #print(x_s4.shape)
        #x_s4= self.sknet(x_s4)
        #print(x_s4.shape)
        #x_s4 = self.reasoning4(x_s4)
       
        #x_e4 = self.reasoning4(x_e4)
        # four cross refinement units
        #x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df1(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        #x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df2(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        #x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df3(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        #x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df4(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
       
        # feature aggregation using u-net
        #pred_s = self.output_s(x_s1, x_s2, x_s3, x_s4)
        #print(pred_s.shape)
        #pred_s = self.reasoning2(pred_s)
       # self.triplet
        #pred_s = self.output_s(x_s1, x_s2, x_s3, x_s4)
        pred_e = self.output_e(x_e1, x_e2, x_e3, x_e4)
       
        #pred_s = F.upsample(output, size=size, mode='bilinear', align_corners=True)
        pred_e = F.interpolate(pred_e, size=size, mode='bilinear', align_corners=True)

        return output, pred_e

    def initialize_weights(self):
        res50 =  torchvision.models.resnet50(pretrained=True)
        self.resnet.load_state_dict(res50.state_dict(), False)
        #self.resnet.load_state_dict(torch.load('./model/resnet50.pth'),False)
