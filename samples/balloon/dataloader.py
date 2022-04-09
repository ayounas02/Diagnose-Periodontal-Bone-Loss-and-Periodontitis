import torch
import os
import pandas as pd
from PIL import Image

import numpy as np
import torch.nn.functional as F
import constants as const
from torch.utils.data import Dataset
import cv2


class teethDataset(torch.utils.data.Dataset):
    def __init__(self, xrays, masks, transformation=None):
        self.xrays = xrays
        self.masks = masks
        self.transformation = transformation


    def __len__(self):
        #return 2
        return len(self.xrays)


    def __getitem__(self, index):
        xray_scan = self.xrays[index]
        if self.masks is not None:
            xray_mask = self.masks[index]
        # img = cv2.imread(xray_mask, cv2.IMREAD_GRAYSCALE)
        # cv2.imwrite(xray_mask, img)

        if self.transformation and self.masks is not None:
            x, y = self.transformation(Image.open(xray_scan), Image.open(xray_mask))
        else:
            x, y = self.transformation(Image.open(xray_scan),Image.open(xray_scan))




        return x, y


















