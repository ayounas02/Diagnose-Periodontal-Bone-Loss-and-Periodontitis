import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
import numpy as np
from constants import *

class DoubleConvolutional(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolutional, self).__init__()
        self.conv = nn.Sequential(
            ##  First Conv
            nn.Conv2d(in_channels, out_channels, 3, 1, 1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            ##  Second Conv
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Segmentation(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Segmentation, self).__init__()

        self.up_sampling = nn.ModuleList()
        self.down_sampling = nn.ModuleList()
        self.max_pool = nn.MaxPool2d(kernel_size=2 , stride=2)
        self.num_filters = [64, 128, 256, 512]


        ##down sampling
        for feature in self.num_filters:
            self.down_sampling.append(DoubleConvolutional(in_channels,feature))
            in_channels = feature

        ## upsampling
        for feature in reversed(self.num_filters):
            self.up_sampling.append(nn.ConvTranspose2d(feature*2,feature,kernel_size=2, stride=2))
            self.up_sampling.append(DoubleConvolutional(feature*2, feature))

        self.bottleneck = DoubleConvolutional(self.num_filters[-1],self.num_filters[-1]*2)
        self.final_conv = nn.Conv2d(self.num_filters[0], out_channels, kernel_size=1)


    def forward(self, x):

        skip_connections = []

        for down in self.down_sampling:
            x  = down(x)
            skip_connections.append(x)
            x = self.max_pool(x)


        x = self.bottleneck(x)
        skip_connections = skip_connections[ :: -1]

        for index in range(0, len(self.up_sampling), 2):
            x = self.up_sampling[index](x)
            skip_connection = skip_connections[index//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            ### concatenation part
            concat_skip = torch.cat((skip_connection , x), dim = 1)
            x =  self.up_sampling[index+1](concat_skip)

        return self.final_conv(x)
