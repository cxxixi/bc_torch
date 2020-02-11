"""
 Implementation of EnvNet-v2 (ours)
 opt.fs = 44100
 opt.inputLength = 66650

"""


from models.convbnrelu import ConvBNReLU
import torch.nn as nn
import torch.nn.functional as F


class EnvNetv2(nn.Module):
    def __init__(self, n_classes):
        super(EnvNetv2, self).__init__()
        self.conv1 = ConvBNReLU(1, 32, (1, 64), stride=(1, 2))
        self.conv2 = ConvBNReLU(32, 64, (1, 16), stride=(1, 2))
        self.conv3 = ConvBNReLU(1, 32, (8, 8))
        self.conv4 = ConvBNReLU(32, 32, (8, 8))
        self.conv5 = ConvBNReLU(32, 64, (1, 4))
        self.conv6 = ConvBNReLU(64, 64, (1, 4))
        self.conv7 = ConvBNReLU(64, 128, (1, 2))
        self.conv8 = ConvBNReLU(128, 128, (1, 2))
        self.conv9 = ConvBNReLU(128, 256, (1, 2))
        self.conv10 = ConvBNReLU(256, 256, (1, 2))
        self.pool1 = nn.MaxPool2d((1, 64))
        self.pool2 = nn.MaxPool2d((5, 3))
        self.pool3 = nn.MaxPool2d((1, 2))
        self.dropout = nn.Dropout(0.5)

        self.fc11 = nn.Linear(256 * 10 * 8, 4096)
        self.fc12 = nn.Linear(4096, 4096)
        self.fc13 = nn.Linear(4096, n_classes)
        
        self.train = True

    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.pool1(h)
        h = h.transpose(1, 2)

        h = self.conv3(h)
        h = self.conv4(h)
        h = self.pool2(h)
        h = self.conv5(h)
        h = self.conv6(h)
        h = self.pool3(h)
        h = self.conv7(h)
        h = self.conv8(h)
        h = self.pool3(h)
        h = self.conv9(h)
        h = self.conv10(h)
        h = self.pool3(h)
        h = h.view(h.shape[0], -1)

        h = self.dropout(F.relu(self.fc11(h)))
        h = self.dropout(F.relu(self.fc12(h)))

        return self.fc13(h)
