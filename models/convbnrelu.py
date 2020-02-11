# import chainer
# import chainer.functions as F
# import chainer.links as L


import torch.nn as nn
import torch.nn.functional as F


####
'''
soem changes made when modify the code:
1) add bias terms to conv for every chunk. 
2) delete initial weights, which is HeNormal() in chainer

'''
####

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride=1, pad=0, bias=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride=1, padding=pad, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        # )
        # self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride=1, padding=pad, bias=bias),
        # self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)

        return F.relu(h)
