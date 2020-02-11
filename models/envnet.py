"""
 Implementation of EnvNet [Tokozume and Harada, 2017]
 opt.fs = 16000
 opt.inputLength = 24014

"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# import chainer
# import chainer.functions as F
# import chainer.links as L


from models.convbnrelu import ConvBNReLU


class EnvNet(nn.Module):
    def __init__(self, n_classes):
        super(EnvNet, self).__init__()
        self.conv1 = ConvBNReLU(1, 40, (1, 8))
        self.conv2 = ConvBNReLU(40, 40, (1, 8))
        self.conv3 = ConvBNReLU(1, 50, (8, 13))
        self.conv4 = ConvBNReLU(50, 50, (1, 5))
        self.pool1 = nn.MaxPool2d((1, 160))
        self.pool2 = nn.MaxPool2d(3)
        self.pool3 = nn.MaxPool2d((1, 3))
        self.dropout = nn.Dropout(0.5)
        self.fc5 = nn.Linear(50 * 11 * 14, 4096)
        self.fc6 = nn.Linear(4096, 4096)
        self.fc7 = nn.Linear(4096, n_classes)
        
        # self.train = True

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.pool1(h)
        h = torch.transpose(h,1, 2)


        h = self.conv3(h)
        h = self.pool2(h)
        h = self.conv4(h)
        h = self.pool3(h)
        h = h.view(h.shape[0], -1)

        h = self.dropout(F.relu(self.fc5(h))) 
        h = self.dropout(F.relu(self.fc6(h)))

        return self.fc7(h)



