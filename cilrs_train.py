import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from expert_dataset import ExpertDataset
from models.cilrs import CILRS
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    losses = []

    model.eval()

    loss_fn = nn.L1Loss()

    for i, (img, speed, target, mask) in enumerate(dataloader):

        output, pred_speed = model(img, speed)
        command_mask = output * mask

        branch_loss = loss_fn(command_mask, target) * 4
        speed_loss = loss_fn(pred_speed, speed)
        loss = branch_loss + speed_loss
        losses += [loss.item()]

    return sum(losses) / len(losses)


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    losses = []

    # switch to train mode
    model.train()
    # optimizer = optim.SGD(model.parameters(), lr=0.1)
    optimizer = optim.Adam(model.parameters())
    loss_fn = nn.L1Loss()

    for i, (img, speed, target, mask) in enumerate(dataloader):

        output, pred_speed = model(img, speed)
        command_mask = output * mask

        branch_loss = loss_fn(command_mask, target) * 4
        speed_loss = loss_fn(pred_speed, speed)
        loss = branch_loss + speed_loss

        losses += [loss.item()]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return sum(losses) / len(losses)


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    X_train = range(len(train_loss))
    X_val = range(len(val_loss))

    plt.close()

    plt.scatter(X_train, train_loss, color="b")
    plt.plot(X_train, train_loss, color="b")

    plt.scatter(X_val, val_loss, color="g")
    plt.plot(X_val, val_loss, color="g")

    plt.show()


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "../dataset/train"
    val_root = "../dataset/val"
    model = CILRS()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 3
    batch_size = 5
    save_path = "cilrs_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    print("started")
    for i in range(num_epochs):
        print("running epoch ", i + 1, "/", num_epochs)
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
    print("done")
    torch.save(model, save_path)
    print(train_losses)
    print(val_losses)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
