import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from expert_dataset_old import ExpertDataset
from models.cilrs import CILRS
import torch.nn as nn
import numpy as np


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    pass


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    losses = np.empty([1])

    # switch to train mode
    model.train()
    optimizer = optim.SGD(model.parameters(), lr = 0)
    loss_fn = nn.L1Loss()

    for i, (img, speed, target, mask) in enumerate(dataloader):
        print("i = ", i)

        output, pred_speed = model(img, speed)
        command_mask = output * mask

        branch_loss = loss_fn(command_mask, target) * 4
        speed_loss = loss_fn(pred_speed, speed)
        loss = branch_loss + speed_loss

        np.append(losses, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(losses)
    return np.mean(losses)


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    pass


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "../dataset"
    val_root = "../dataset"
    model = CILRS()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 3
    batch_size = 20
    save_path = "cilrs_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
    torch.save(model, save_path)
    print(train_losses)
    print(val_losses)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
