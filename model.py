import torch.nn as nn
import torch.nn.functional as F
import torch

def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=False))
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)
    
class ConvEncoder(nn.Module):
    def __init__(self, c_in=3, out_dim=1024, conv_dim=64, bn=True, norm=True):
        super(ConvEncoder, self).__init__()
        self.conv1 = conv(c_in, conv_dim, 4, bn=bn)
        self.conv2 = conv(conv_dim, conv_dim * 2, 4, bn=bn)
        self.conv3 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1, bn=bn)
        self.conv4 = conv(conv_dim * 2, conv_dim * 2, 3, 1, 1, bn=bn)

        self.conv5 = conv(conv_dim * 2, out_dim, 8, 1, 0, bn=bn)
        self.norm = norm

    def forward(self, x):
        out = F.relu(self.conv1(x))  # (?, 64, 16, 16)
        out = F.relu(self.conv2(out))  # (?, 128, 8, 8)
        out = F.relu(self.conv3(out))  # ( " )
        out = F.relu(self.conv4(out))  # ( " )

        out = self.conv5(out)
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            norm_x = norm_x + (norm_x == 0).float()
            out = out / norm_x
        return [out]

class ConvDecoder(nn.Module):
    def __init__(self, c_in=1024, c_out=3, conv_dim=64, bn=True, norm=True):
        super(ConvDecoder, self).__init__()
        self.deconv0 = deconv(c_in, conv_dim * 2, 8, 1, 0)
        self.deconv1 = deconv(conv_dim * 2, conv_dim, 4)
        self.deconv2 = deconv(conv_dim, c_out, 4, bn=False)
        self.norm = norm

    def forward(self, x):
        out = F.relu(self.deconv0(x))
        out = F.relu(self.deconv1(out))
        out = F.tanh(self.deconv2(out))
        if self.norm:
            norm_x = torch.norm(out, dim=1, keepdim=True)
            norm_x = norm_x + (norm_x == 0).float()
            out = out / norm_x
        return [out]


class Dense_Net(nn.Module):
    """Generator for transfering from svhn to mnist"""
    def __init__(self, input_dim=28*28, out_dim=20, norm=True):
        super(Dense_Net, self).__init__()
        mid_num = 4096
        self.fc1 = nn.Linear(input_dim, mid_num)
        self.fc2 = nn.Linear(mid_num, mid_num)
        self.fc3 = nn.Linear(mid_num, out_dim)
        self.dropout = nn.Dropout(p=0.2)
        self.norm = norm

    def forward(self, x):
        out1 = F.relu(self.fc1(x))
        out2 = F.relu(self.fc2(out1))
        out3 = self.fc3(out2)
        if self.norm:
            norm_x = torch.norm(out3, dim=1, keepdim=True)
            norm_x = norm_x + (norm_x == 0).float()
            out3 = out3 / norm_x
        return [out1, out3]
