# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import math
from torch.nn.modules.batchnorm import _BatchNorm
import torch

import torch.nn.functional as F
import torch.nn as nn
from functools import partial

from .bn_lib.nn.modules import SynchronizedBatchNorm2d
#import settings

norm_layer = partial(SynchronizedBatchNorm2d, momentum=3e-4)
class EMAU(nn.Module):
    '''The Expectation-Maximization Attention Unit (EMAU).

    Arguments:
        c (int): The input and output channel number.
        k (int): The number of the bases.
        stage_num (int): The iteration number for EM.
    '''
    def __init__(self, c, k, stage_num=2):
        super(EMAU, self).__init__()
        self.stage_num = stage_num

        mu = torch.Tensor(1, c, k)
        mu.normal_(0, math.sqrt(2. / k))    # Init with Kaiming Norm.
        mu = self._l2norm(mu, dim=1)
        self.register_buffer('mu', mu)

        self.conv1 = nn.Conv2d(c, c, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, 1, bias=False),
            norm_layer(c))        
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, _BatchNorm):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
 

    def forward(self, x):
        idn = x
        # The first 1x1 conv
        x = self.conv1(x)

        # The EM Attention
        b, c, h, w = x.size()
        x = x.view(b, c, h*w)               # b * c * n
        mu = self.mu.repeat(b, 1, 1)        # b * c * k
        with torch.no_grad():
            for i in range(self.stage_num):
                x_t = x.permute(0, 2, 1)    # b * n * c
                z = torch.bmm(x_t, mu)      # b * n * k
                z = F.softmax(z, dim=2)     # b * n * k
                z_ = z / (1e-6 + z.sum(dim=1, keepdim=True))
                mu = torch.bmm(x, z_)       # b * c * k
                mu = self._l2norm(mu, dim=1)
                
        # !!! The moving averaging operation is writtern in train.py, which is significant.

        z_t = z.permute(0, 2, 1)            # b * k * n
        x = mu.matmul(z_t)                  # b * c * n
        x = x.view(b, c, h,w)              # b * c * h * w
        x = F.relu(x, inplace=True)

        # The second 1x1 conv
        x = self.conv2(x)
        x = x + idn
        x = F.relu(x, inplace=True)
        #print(x.shape)
        return x

    def _l2norm(self, inp, dim):
        '''Normlize the inp tensor with l2-norm.

        Returns a tensor where each sub-tensor of input along the given dim is 
        normalized such that the 2-norm of the sub-tensor is equal to 1.

        Arguments:
            inp (tensor): The input tensor.
            dim (int): The dimension to slice over to get the ssub-tensors.

        Returns:
            (tensor) The normalized tensor.
        '''
        return inp / (1e-6 + inp.norm(dim=dim, keepdim=True))

 
class GCN(nn.Module):
    """ Graph convolution unit (single layer)
    """

    def __init__(self, num_state, num_node, bias=False):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv1d(num_node+1, num_node+1, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x):
        # (n, num_state, num_node) -> (n, num_node, num_state)
        #                          -> (n, num_state, num_node)
        h = self.conv1(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1)
        #print(h.shape)
        #print(x.shape)
        h = h + x
        #print(h.shape)
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
        self.emau = EMAU(num_in,self.num_n,1)
        # reduce dim
        self.conv_state = ConvNd(num_in, self.num_s, kernel_size=1)
        # projection map
        self.conv_proj = ConvNd(num_in, self.num_n, kernel_size=1)
        self.conv_proj1 = ConvNd(num_in, 1, kernel_size=1)
        # ----------
        # reasoning via graph convolution
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        # ----------
        # extend dimension
        self.conv_extend = ConvNd(self.num_s, num_in, kernel_size=1, bias=False)

        self.blocker = BatchNormNd(num_in, eps=1e-04) # should be zero initialized


    def forward(self, x):
        '''
        :param x: (n, c, d, h, w)
        '''
        n = x.size(0)
        #print(x.shape)
        x= self.emau(x)
        #print(x.shape)
        # (n, num_in, h, w) --> (n, num_state, h, w)
        #                   --> (n, num_state, h*w)
        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        #print(x_state_reshaped.shape)
        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_proj_reshaped = self.conv_proj(x).view(n, self.num_n, -1)
        #print(x_proj_reshaped.shape)
        x_proj_reshaped1 = self.conv_proj1(x).view(n, 1, -1)
        # (n, num_in, h, w) --> (n, num_node, h, w)
        #                   --> (n, num_node, h*w)
        x_rproj_reshaped = x_proj_reshaped
        x_rproj_reshaped1 = x_proj_reshaped1

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # projection: coordinate space -> interaction space
        # (n, num_state, h*w) x (n, num_node, h*w)T --> (n, num_state, num_node)
        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_state1 = torch.matmul(x_state_reshaped, x_proj_reshaped1.permute(0, 2, 1))
        if self.normalize:
            x_n_state = x_n_state * (1. / x_state_reshaped.size(2))
            
            x_n_state = x_n_state+x_n_state1
        #print(x_n_state.shape)
        #print(x_n_state1.shape)
        x_n_state1=torch.cat((x_n_state,x_n_state1),dim=2)
        #print(x_n_state1.shape)
        # reasoning: (n, num_state, num_node) -> (n, num_state, num_node)
        
        x_n_rel = self.gcn(x_n_state1)
        #print(x_n_rel.shape)
        x_n_rel= x_n_rel[:,:,torch.arange(x_n_rel.size(2))!=16] 
        #print(x_n_rel.shape)
        #x_n_rel1 = self.gcn(x_n_state1)
        #x_n_rel = x_n_rel+x_n_rel1
        # reverse projection: interaction space -> coordinate space
        # (n, num_state, num_node) x (n, num_node, h*w) --> (n, num_state, h*w)
        x_state_reshaped = torch.matmul(x_n_rel, x_rproj_reshaped)

        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        # (n, num_state, h*w) --> (n, num_state, h, w)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        
        # -----------------
        # (n, num_state, h, w) -> (n, num_in, h, w)
        out = x + self.blocker(self.conv_extend(x_state))
        #print(out.shape)
        out=self.emau(out)
       
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
        #print(net(data).size())

        data = torch.autograd.Variable(torch.randn(2, 32, 14, 14))
        net = GloRe_Unit_2D(32, 16, normalize)
        print(net(data).size())

        data = torch.autograd.Variable(torch.randn(2, 32, 8, 14, 14))
        net = GloRe_Unit_3D(32, 16, normalize)
        #print(net(data).size())
