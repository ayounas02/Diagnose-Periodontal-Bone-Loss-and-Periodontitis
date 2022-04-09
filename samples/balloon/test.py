from train import  load_checkpoint, Compose, correct_pixel

import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import albumentations as A
# from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import constants as const
from constants import *
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from model import Segmentation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import DataLoader
import random
import numpy as np
from dataloader import teethDataset
from constants import *
import cv2
import matplotlib.pyplot as plt
#from imutils import contours

def extract_edges(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    canny = cv2.Canny(blurred, 120, 255, 1)

    return canny


def tensor_to_image(predicted_image, path):
    img = torch.reshape(predicted_image, (const.width, const.height))
    img = img.detach().numpy()
    img = correct_pixel(img)
    #cv2.imwrite(path, img)
    fig = plt.imshow(img)
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.show()


def predict_output(model, img_path):
    model.eval()
    transform = Compose([transforms.Resize((const.width, const.height)), transforms.ToTensor()])

    train_ds = teethDataset(
        xrays=[img_path],
        masks=None,
        transformation=transform,
    )

    test_loader = DataLoader(
        train_ds,
        batch_size=1,
        num_workers=0,
        pin_memory=False,
    )
    with torch.no_grad():
        for x, y in test_loader:
            # if x[0].shape[0] != 1:
            #     x1 = x[0][1]
            #     x1 = torch.unsqueeze(x1, 0)
            #     x1 = torch.unsqueeze(x1, 0)
            preds = torch.sigmoid(model(x))
            preds = torch.where(preds > 0.5, 1.0, 0.0)
    return preds


def predict_masks(path):

    perio_model = Segmentation(in_channels=1, out_channels=1).to(device)
    cej_model = Segmentation(in_channels=1, out_channels=1).to(device)

    ## define optimizer
    perio_optimizer = optim.Adam(perio_model.parameters(), lr=learning_rate)
    cej_optimizer = optim.Adam(cej_model.parameters(), lr=learning_rate)

    load_checkpoint(torch.load(PERIOD_MODEL_LOAD_PATH, map_location=torch.device('cpu')), perio_model, perio_optimizer)
    load_checkpoint(torch.load(CEJ_MODEL_LOAD_PATH, map_location=torch.device('cpu')), cej_model, cej_optimizer)

    predicted_perio = predict_output(perio_model, path)
    predicted_cej = predict_output(cej_model, path)

    preds_p = torch.where(predicted_perio > 0.5, 1.0, 0.0)
    preds_c = torch.where(predicted_cej > 0.5, 1.0, 0.0)

    preds_p = torch.reshape(preds_p, (const.width, const.height))
    preds_c = torch.reshape(preds_c, (const.width, const.height))

    preds_p = preds_p.detach().numpy()
    preds_c = preds_c.detach().numpy()

    return preds_p, preds_c

# path = 'segments/sshrey_CEJ/v0.1/1.png'
# # perio_path = input('Enter perio model_path')
# # cej_path = input('Enter cej model path')
#
# perio_path = PERIOD_MODEL_LOAD_PATH
# cej_path = CEJ_MODEL_LOAD_PATH
#
# perio_model = Segmentation(in_channels=1, out_channels=1).to(device)
# cej_model = Segmentation(in_channels=1, out_channels=1).to(device)
#
#
# ## define optimizer
# perio_optimizer = optim.Adam(perio_model.parameters(), lr=learning_rate)
# cej_optimizer = optim.Adam(cej_model.parameters(), lr=learning_rate)
#
# load_checkpoint(torch.load(PERIOD_MODEL_LOAD_PATH, map_location=torch.device('cpu')), perio_model, perio_optimizer)
# load_checkpoint(torch.load(CEJ_MODEL_LOAD_PATH, map_location=torch.device('cpu')), cej_model, cej_optimizer)
#
# predicted_perio = predict_output(perio_model, path)
# predicted_cej = predict_output(cej_model, path)
#
# preds_p = torch.where(predicted_perio > 0.5, 1.0, 0.0)
# preds_c = torch.where(predicted_cej > 0.5, 1.0, 0.0)
#
# tensor_to_image(predicted_perio, 'pred_perio.png')
# tensor_to_image(predicted_cej, 'pred_cej.png')
#
# canny_p = extract_edges('pred_perio.png')
# canny_c = extract_edges('pred_cej.png')
#
# fig = plt.imshow(canny_p)
# fig1 = plt.imshow(canny_c)
# plt.axis('off')
# fig.axes.get_xaxis().set_visible(False)
# fig.axes.get_yaxis().set_visible(False)
# plt.savefig('asda.png', bbox_inches='tight', pad_inches=0)
# plt.show()
#
# plt.savefig('res.png')
# plt.close()



