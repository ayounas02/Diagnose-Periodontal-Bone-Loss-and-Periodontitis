
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
from sklearn.model_selection import train_test_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from PIL import Image
print(device)
import cv2
from torch.utils.data import DataLoader
import random
import numpy as np
from dataloader import teethDataset
import matplotlib.image as mpimg
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img), t(mask)

        return img, mask

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = torch.flatten(y_true)
    y_pred_f = torch.flatten(y_pred)
    y_pred_f = torch.where(y_pred_f > 0.5, 1.0, 0.0)
    intersection = torch.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def get_teeth_data(path):
    xrays = []
    masks = []

    for subdir, dirs, files in os.walk(path):
        for file in files:
            if subdir.__contains__('v0.1')  and not subdir.__contains__('masks') and not file.__contains__('label') and not file.__contains__('mask'):
                xrays.append(os.path.join(subdir, file))
            if subdir.__contains__('masks') and file.endswith('.png'):
                masks.append(os.path.join(subdir, file))

    return xrays, masks




def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            y = torch.reshape(y, (1, 1, const.width, const.height))
            preds = torch.sigmoid(model(x))
            preds = torch.where(preds > 0.5, 1.0, 0.0)
            y = torch.where(y > 0.5, 1.0, 0.0)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                    (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct / num_pixels * 100:.2f}"
    )
    print(f"Dice score: {dice_score / len(loader)}")
    model.train()

    return dice_score/len(loader), num_correct/num_pixels*100



def correct_pixel(img):
    for i in range(const.width):
        for j in range(const.height):
            if img[i][j] > 0.5:
                img[i][j] = 1
            else:
                img[i][j] = 0

    return img

def plot_graphs(train_arr, test_arr, filename):
    length = len(train_arr)
    t_epochs = []

    for i in range(length):
        t_epochs.append(i)
    plt.plot(t_epochs, train_arr)
    plt.plot(t_epochs, test_arr)
    ylable = filename.replace('.png', '')
    plt.ylabel(ylable)
    plt.xlabel('epoch')
    plt.legend(["train"+ylable, "test"+ylable])
    plt.savefig(filename)
    plt.close()


#we just store target * predicted images
def output_images(xray, predicted_image, y, epoch, batch_idx, results_path):
    w = 20
    h = 20
    fig = plt.figure(figsize=(20, 20))
    columns = 3
    rows = 1
    img = torch.reshape(y, (const.width, const.height))
    img = img.detach().numpy()
    fig.add_subplot(rows, columns, 1)
    plt.imshow(img)

    img = torch.reshape(xray, (const.width, const.height))
    img = img.detach().numpy()



    #plt.imshow(ds, cmap=plt.cm.bone)
    for i in range(2, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
        #plt.savefig('img.png')
        img = torch.reshape(predicted_image, (const.width, const.height))
        img = img.detach().numpy()
        img = correct_pixel(img)

    plt.savefig(results_path +str(epoch)+'_'+str(batch_idx)+'_result.png')
    plt.close()



def get_loaders(path, batch_size, num_workers=0, pin_memory=False, ):
    transform = Compose([transforms.Resize((const.width, const.height)), transforms.ToTensor()])
    X,y = get_teeth_data(path)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=69)

    train_ds = teethDataset(
        xrays = X_train,
        masks=y_train,
        transformation=transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        #shuffle=True,
    )

    val_ds = teethDataset(
        xrays=X_test,
        masks=y_test,
        transformation=transform,
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

### load checkpoint
def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])


def train_fn(train_loader, model, optimizer, loss_fn, epoch, results_path):
    loop = tqdm(train_loader, leave=True)
    mean_loss = None
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(device), y.to((device))
        #x = torch.squeeze(x, 1)
        #x = x.transpose(2, 3)
        predicted_image = model(x)

        y = torch.where(y > 0.5, 1.0, 0.0)

        loss = loss_fn(predicted_image.float(), torch.reshape(y.float(), (BATCH_SIZE, out_channels,  const.width, const.height))) + dice_coef_loss(torch.reshape(y.float(), (BATCH_SIZE, out_channels,  const.width, const.height)), predicted_image.float())
        print(loss)
        output_images(x, predicted_image, torch.reshape(y.float(), (1, 1,  const.width, const.height)), epoch, batch_idx, results_path)
        mean_loss = loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    return mean_loss


def train_models(model, optimizer, train_loader, val_loader, loss_fn, current_model, checkpoint_path, results_path ):
    loss_values = []
    dice_scores = []
    accuracies = []

    val_dice_scores = []
    val_accuracies = []

    model.train()
    for epoch in range(NUM_EPOCHS):
        loss = train_fn(train_loader, model, optimizer, loss_fn, epoch, results_path)
        loss_values.append(loss)

        if epoch > 1:
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, checkpoint_path + str(epoch) + ".pth.tar")

        # check train accuracy
        dice_s, accuracy = check_accuracy(train_loader, model, device=device)
        dice_scores.append(dice_s)
        accuracies.append(float(accuracy))

        # check val accuracy

        test_dice_s, test_accuracy = check_accuracy(val_loader, model, device=device)
        val_dice_scores.append(test_dice_s)
        val_accuracies.append(test_accuracy)

        plot_graphs(dice_scores, val_dice_scores, current_model+ '_dice_scores.png')
        plot_graphs(accuracies, val_accuracies, current_model+ '_accuracy.png')



def main():


    perio_model = Segmentation(in_channels=1, out_channels=1).to(device)
    cej_model = Segmentation(in_channels=1, out_channels=1).to(device)

    ## define loss
    perio_loss_fn = nn.BCEWithLogitsLoss()
    cej_loss_fn = nn.BCEWithLogitsLoss()


    ## define optimizer
    perio_optimizer = optim.Adam(perio_model.parameters(), lr=learning_rate)
    cej_optimizer = optim.Adam(cej_model.parameters(), lr=learning_rate)

    # load train & test dataset ->val_loader means test dataset

    perio_train_loader, perio_val_loader = get_loaders(
        './segments/sshrey_Perio1/',
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    cej_train_loader, cej_val_loader = get_loaders(
        './segments/sshrey_CEJ/',
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    ## we can resume training from last checkppoint by setting LOAD_MODEL= True & updating filename herr
    if PERIO_LOAD:
        load_checkpoint(torch.load(PERIOD_MODEL_LOAD_PATH), perio_model, perio_optimizer)

    if CEJ_LOAD:
        load_checkpoint(torch.load(CEJ_MODEL_LOAD_PATH), cej_model, cej_optimizer)

    train_models(perio_model, perio_optimizer, perio_train_loader, perio_val_loader, perio_loss_fn, 'perio', PERIO_SAVED_DIR,PERIO_RESULT_DIR )
    print('##################### CEJ MODEL ##########################################')
    train_models(cej_model, cej_optimizer, cej_train_loader, cej_val_loader, cej_loss_fn, 'cej', CEJ_SAVED_DIR, CEJ_RESULT_DIR)





if __name__ == "__main__":
    main()
