# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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

class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        h = h + x
        # (n, num_state, num_node) -> (n, num_state, num_node)
        h = self.conv2(self.relu(h))
        return h


class GloRe_Unit(nn.Module):
    """
    Graph-based Global Reasoning Unit

    Parameter:
        'normalize' is not necessary if the input size is fixed
    """
    def __init__(self, num_in, num_mid, 
                 ConvNd=nn.Conv2d,
                 BatchNormNd=nn.BatchNorm2d,
                 normalize=False):
        super(GloRe_Unit, self).__init__()
        
        self.normalize = normalize
        self.num_s = int(2 * num_mid)
        self.num_n = int(1 * num_mid)
        self.triplet=  TripletAttention(num_in)
        #self.emau = EMAU(num_in,64,3)
        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        self.conv_state1 = ConvNd(num_in, 1, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        self.conv_proj1 = ConvNd(num_in, self.num_n, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.gcn1 = GCN(num_state=self.num_s+1, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04) # should be zero initialized


    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)
        x=self.triplet(x)
        #x= self.emau(x)
        #x=self.stage_1(x)
        #x=self.triplet(x)
        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        
        x_state_reshaped1 = self.conv_state1(x).view(n, 1, -1)
        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)
        x_proj_reshaped1 = self.conv_proj1(x).view(n, self.num_n, -1)
        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        #print(x_proj_reshaped.shape)
        #print(x_proj_reshaped1.shape)
        x_rproj_reshaped = x_proj_reshaped
        x_rproj_reshaped1 = x_proj_reshaped1
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_state1 = torch.matmul(x_state_reshaped1, x_proj_reshaped1.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))

        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        x_n_rel = self.gcn(x_n_state)
        x_n_rel2= x_n_rel
        #print(x_n_state.shape)
        #print(x_n_state1.shape)
        x_n_state1=torch.cat((x_n_state,x_n_state1),1)
        x_n_rel = self.gcn1(x_n_state1)
        x_n_rel= x_n_rel[:,torch.arange(x_n_rel.size(1))!=self.num_s,:] 
        x_n_rel= x_n_rel+x_n_rel2
        x_n_rel = self.gcn(x_n_rel)
        #x_n_rel = self.gcn(x_n_rel)
        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])

        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.blocker(self.conv_extend(x_state))
        #out = self.triplet(out)
        #out= self.emau(out)
        return out


class GloRe_Unit_1D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_1D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv1d,
                                            BatchNormNd=nn.BatchNorm1d,
                                            normalize=normalize)

class GloRe_Unit_2D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_2D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv2d,
                                            BatchNormNd=nn.BatchNorm2d,
                                            normalize=normalize)

class GloRe_Unit_3D(GloRe_Unit):
    def __init__(self, num_in, num_mid, normalize=False):
        """
        Set 'normalize = True' if the input size is not fixed
        """
        super(GloRe_Unit_3D, self).__init__(num_in, num_mid,
                                            ConvNd=nn.Conv3d,
                                            BatchNormNd=nn.BatchNorm3d,
                                            normalize=normalize)


if __name__ == '__main__':

    for normalize in [True, False]:
        data = torch.autograd.Variable(torch.randn(2, 32, 1))
        net = GloRe_Unit_1D(32, 16, normalize)
        print(net(data).size())

        data = torch.autograd.Variable(torch.randn(2, 32, 14, 14))
        net = GloRe_Unit_2D(32, 16, normalize)
        print(net(data).size())

        data = torch.autograd.Variable(torch.randn(2, 32, 8, 14, 14))
        net = GloRe_Unit_3D(32, 16, normalize)
        print(net(data).size())
