import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models
from .ResNet import ResNet50
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import mxnet as mx
from symbol_basic import *

# - - - - - - - - - - - - - - - - - - - - - - -
# GloRe Unit
def GloRe_Unit(data, settings, name, stride=(1,1)):

    # PROJ
    if type(data) is list:
        data_raw = data
        data  = mx.symbol.Concat(*[data[0], data[1]],  name=('%s_cat-input' % name))
    else:
        raise "data is list"

    num_res, num_den, num_mid = settings
    num_in = num_res + num_den

    num_state = int(2 * num_mid)
    num_node  = int(1 * num_mid)

    # reduce sampling space if is required
    if tuple(stride) == (1,1):
        x_pooled = data  # default
    else:
        x_pooled = mx.symbol.Pooling(data=data, pool_type="avg", kernel=stride, stride=stride, name=('%s_pooling' % name))

    # generate "state" for each node:
    # (N, num_in, H_sampled, W_sampled) --> (N, num_state, H_sampled, W_sampled)
    #                                   --> (N, num_state, H_sampled*W_sampled)
    x_state = BN_AC_Conv(data=x_pooled, num_filter=num_state,  kernel=(1, 1), pad=(0, 0), name=('%s_conv-state' % name))
    x_state_reshaped = mx.symbol.reshape(x_state, shape=(0, 0, -1),                       name=('%s_conv-state-reshape' % name))

    # prepare "projection" function (coordinate space -> interaction space):
    # (N, num_in, H_sampled, W_sampled) --> (N, num_node, H_sampled, W_sampled)
    #                                   --> (N, num_node, H_sampled*W_sampled)
    x_proj = BN_AC_Conv(data=x_pooled, num_filter=num_node, kernel=(1, 1), pad=(0, 0),   name=('%s_conv-proj' % name))
    x_proj_reshaped = mx.symbol.reshape(x_proj, shape=(0, 0, -1),                        name=('%s_conv-proj-reshape' % name))

    # prepare "reverse projection" function (interaction space -> coordinate space)
    # (N, num_in, H, W) --> (N, num_node, H, W)
    #                   --> (N, num_node, H*W)
    x_rproj_reshaped = x_proj_reshaped
    
    
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # Projection: coordinate space -> interaction space
    # (N, num_state, H_sampled*W_sampled) x (N, num_node, H_sampled*W_sampled)T --> (N, num_state, num_node)
    x_n_state = mx.symbol.batch_dot(lhs=x_state_reshaped, rhs=x_proj_reshaped, transpose_b=True,  name=('%s_proj' % name))

    # ------
    # Relation Reasoning
    # -> propogate information: [G * X]
    #     (N, num_state, num_node)
    #     -permute-> (N, num_node, num_state) 
    #     --conv-->  (N, num_node, num_state) 
    #     -permute-> (N, num_state, num_node) (+ shortcut)
    #     --conv-->  (N, num_state, num_node)
    x_n_rel = x_n_state
    x_n_rel = mx.symbol.transpose(data=x_n_rel, axes=(0,2,1),                           name=('%s_GCN-G-permute1' % name))
    x_n_rel = BN_AC_Conv(data=x_n_rel, num_filter=num_node,  kernel=1, pad=0, stride=1, name=('%s_GCN-G' % name))
    x_n_rel = mx.symbol.transpose(data=x_n_rel, axes=(0,2,1),                           name=('%s_GCN-G-permute2' % name))
    # -> add shortcut
    x_n_rel = mx.symbol.ElementWiseSum(*[x_n_state, x_n_rel],                           name=('%s_GCN-G_sum' % name))
    # -> update state: [H * W]
    x_n_rel = BN_AC_Conv(data=x_n_rel, num_filter=num_state, kernel=1, pad=0, stride=1, name=('%s_GCN-GHW' % name))
    # -> output
    x_n_state_new = x_n_rel

    # ------
    # Reverse Projection: interaction space -> coordinate space
    # (N, num_state, num_node) x (N, num_node, H*W) --> (N, num_state, H*W)
    #                                               --> (N, num_state, H, W)
    x_out = mx.symbol.batch_dot(lhs=x_n_state_new, rhs=x_rproj_reshaped, name=('%s_reverse-proj' % name))
    x_out = mx.symbol.reshape_like(x_out, rhs=x_state,                   name=('%s_reverse-proj-reshape' % name))

    # ------
    # expand dimension (dual path)
    x_out1 = BN_AC_Conv(data=x_out, num_filter=num_res, kernel=(1, 1), pad=(0, 0), name=('%s_conv-tail-res' % name))
    x_out2 = BN_AC_Conv(data=x_out, num_filter=num_den, kernel=(1, 1), pad=(0, 0), name=('%s_conv-tail-den' % name))

    return x_out1


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
            x = self.conv(F.upsample(x, size=target.size()[2:], mode='bilinear', align_corners=True))
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
            BasicConv2d(5*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f2 = nn.Sequential(
            BasicConv2d(4*channel, channel, 3, padding=1),
            BasicConv2d(channel, channel, 3, padding=1)
        )
        self.conv_f3 = nn.Sequential(
            BasicConv2d(3*channel, channel, 3, padding=1),
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

        x_ef1 = x_e1 + self.conv_f5(x_e1 * x_s1 *
                                    self.conv7(x_s2, x_e1) *
                                    self.conv8(x_s3, x_e1) *
                                    self.conv9(x_s4, x_e1))
        x_ef2 = x_e2 + self.conv_f6(x_e2 * x_s2 *
                                    self.conv10(x_s3, x_e2) *
                                    self.conv11(x_s4, x_e2))
        x_ef3 = x_e3 + self.conv_f7(x_e3 * x_s3 *
                                    self.conv12(x_s4, x_e3))
        x_ef4 = x_e4 + self.conv_f8(x_e4 * x_s4)

        return x_sf1, x_sf2, x_sf3, x_sf4, x_ef1, x_ef2, x_ef3, x_ef4
k_sec  = {  2: 4, \
            3: 8, \
            4: 20, \
            5: 3   }
inc_sec= {  2: 16, \
            3: 32, \
            4: 32, \
            5: 128 }

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
    def __init__(self, channel=32):
        super(SCRN, self).__init__()
        self.resnet = ResNet50()
        self.reduce_s1 = Reduction(256, channel)
        self.reduce_s2 = Reduction(512, channel)
        self.reduce_s3 = Reduction(1024, channel)
        self.reduce_s4 = Reduction(2048, channel)

        self.reduce_e1 = Reduction(256, channel)
        self.reduce_e2 = Reduction(512, channel)
        self.reduce_e3 = Reduction(1024, channel)
        self.reduce_e4 = Reduction(2048, channel)

        self.df1 = DenseFusion(channel)
        self.df2 = DenseFusion(channel)
        self.df3 = DenseFusion(channel)
        self.df4 = DenseFusion(channel)

        self.output_s = ConcatOutput(channel)
        self.output_e = ConcatOutput(channel)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                
        self.initialize_weights()

    def forward(self, x):
        size = x.size()[2:]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x1 = self.resnet.layer1(x)
        x2 = self.resnet.layer2(x1)
        x3 = self.resnet.layer3(x2)
        x4 = self.resnet.layer4(x3)
        

        # feature abstraction
        x_s1 = self.reduce_s1(x1)
        # group reasoning
        for i in range(2, k_sec[3]+1):
            
            if i in [4,7]:
                x_s1= GloRe_Unit(data=x_s1,settings=(256,int((i)*inc), 256), \
                                   name="conv4_x__%d_extra"%i, stride=(1,1))
        x_s2 = self.reduce_s2(x2)
        x_s3 = self.reduce_s3(x3)
        x_s4 = self.reduce_s4(x4)

        x_e1 = self.reduce_e1(x1)
        x_e2 = self.reduce_e2(x2)
        x_e3 = self.reduce_e3(x3)
        x_e4 = self.reduce_e4(x4)

        # four cross refinement units
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df1(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df2(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df3(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)
        x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4 = self.df4(x_s1, x_s2, x_s3, x_s4, x_e1, x_e2, x_e3, x_e4)

        # feature aggregation using u-net
        pred_s = self.output_s(x_s1, x_s2, x_s3, x_s4)
        pred_e = self.output_e(x_e1, x_e2, x_e3, x_e4)

        pred_s = F.upsample(pred_s, size=size, mode='bilinear', align_corners=True)
        pred_e = F.upsample(pred_e, size=size, mode='bilinear', align_corners=True)
        
                
        return pred_s, pred_e

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=True)
        self.resnet.load_state_dict(res50.state_dict(), False)
