import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

import os
import numpy as np
import torch
from torch import optim
import csv

def iou_coeff(y_true, y_pred, smooth=1e-6):
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)

    intersection = (y_true * y_pred).sum()
    total = (y_true + y_pred).sum()
    union = total - intersection

    return (intersection + smooth) / (union + smooth)

def jaccard_distance_loss(y_true, y_pred, smooth=1e-6):
    return 1 - iou_coeff(y_true, y_pred, smooth=smooth)

def power_jaccard_loss(y_true, y_pred, p=2, smooth=1e-6):
    """
    From https://www.scitepress.org/Papers/2021/103040/103040.pdf
    """
    y_true = y_true.float().view(-1)
    y_pred = y_pred.float().view(-1)

    intersection = (y_true * y_pred).sum()
    total = (torch.pow(y_true, p) + torch.pow(y_pred, p)).sum()
    union = total - intersection

    jaccard = (intersection + smooth) / (union + smooth)

    return 1 - jaccard

def binary_focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Adapted from: https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/base/functional.py#L286
    """
    epsilon = 1e-7
    y_pred = torch.clamp(y_pred, epsilon, 1.0 - epsilon)

    loss_1 = - y_true * (alpha * torch.pow((1 - y_pred), gamma) * torch.log(y_pred))
    loss_0 = - (1 - y_true) * ((1 - alpha) * torch.pow(y_pred, gamma) * torch.log(1 - y_pred))
    loss = loss_0 + loss_1

    return loss.mean()


def binary_focal_jaccard_loss(y_true, y_pred):
    return power_jaccard_loss(y_true, y_pred) + binary_focal_loss(y_true, y_pred)

class Trainer:
    def __init__(self, config, train_loader, valid_loader):
        """
        Initializes the Trainer with configuration, data loaders, and other parameters.

        Args:
            config (object): Configuration object with model parameters like learning rate and epochs.
            train_loader (DataLoader): DataLoader for the training dataset.
            valid_loader (DataLoader): DataLoader for the validation dataset.
        """

        self.train_loader = train_loader
        self.valid_loader = valid_loader

        # Get cpu, gpu or mps device for training.
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        print(f"Using {self.device} device")

        self.criterion = power_jaccard_loss

        self.lr = config.lr
        self.num_epochs = config.num_epochs
        self.model_path = config.model_path

    def train_and_evaluate(self, model, model_name):
        model = model.to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        best_jaccard = 0.0
        for epoch in range(self.num_epochs):
            model.train()
            for images, masks in self.train_loader:
                images, masks = images.to(self.device), masks.to(self.device)

                logits_masks = model(images)
                pred_masks = torch.sigmoid(logits_masks)

                # print("Outputs range:", pred_masks.min(), pred_masks.max())
                # print("Masks range:", masks.min(), masks.max())

                # plt.imshow(images[0].cpu().permute(1, 2, 0).numpy())
                # plt.axis('off')
                # plt.title("Image")
                # plt.show()

                # plt.imshow(masks[0].cpu().permute(1, 2, 0).numpy())
                # plt.axis('off')
                # plt.title("Mask")
                # plt.show()

                # plt.imshow(pred_masks[0].cpu().permute(1, 2, 0).detach().numpy())
                # plt.axis('off')
                # plt.title("Output")
                # plt.show()

                loss = self.criterion(pred_masks, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # print(f"Epoch [{epoch + 1}/{self.num_epochs}], Training loss: {losses:.4f}")

            # Validation
            model.eval()

            val_jaccards = []
            with torch.no_grad():
                for images, masks in self.valid_loader:
                    images, masks = images.to(self.device), masks.to(self.device)

                    logits_masks = model(images)
                    pred_masks = torch.sigmoid(logits_masks)

                    jaccard = iou_coeff(pred_masks.cpu(), masks.cpu())
                    val_jaccards.append(jaccard)

                median_jaccard = np.median(val_jaccards)

                print(f"Epoch [{epoch + 1}/{self.num_epochs}], Median Jaccard: {median_jaccard:.4f}")

                # Save the best model
                if median_jaccard > best_jaccard:
                    best_jaccard = median_jaccard
                    torch.save(model.state_dict(), os.path.join(self.model_path, f"{model_name}_best.pth"))

        return best_jaccard
