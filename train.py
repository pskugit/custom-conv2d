import numpy as np
import torch
from models.fcn import VGGNet, FCNmy, cfg, ranges
import torch.nn as nn
import torch.optim as optim
from models.customconv import MyConv2d, Conv2dXY
import os
import torch
import math
import time
import datetime
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
import cv2
from torch.utils.data import Dataset, DataLoader
from data.circles_semseg import CirclesSemseg


parser = argparse.ArgumentParser(description='Train xy experiment')
parser.add_argument("--v", action="store_true", help="set to have verbose outputs")
parser.add_argument("--dataroot", default="/home/skudlik/xyexp/circle_4cls/circles/",
                    help="Folder containing my circles")
parser.add_argument("--labelroot", default="/home/skudlik/xyexp/circle_4cls/circles_labels/",
                    help="Folder containing labels")
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 2]')
parser.add_argument('--epochs', type=int, default=2, help='Epochs [default: 10]')
parser.add_argument('--lr', type=int, default=1e-3, help='learning rate [default: 0.001]')
parser.add_argument('--n_class', type=int, default=4, help='Point Number [default: 3]')
parser.add_argument("--xyconv", action="store_true", help="")
parser.add_argument('--exp_dir', type=str, default="/home/skudlik/xyexp/experiments/", help='Log path [default: None]')
parser.add_argument('--exp_name', type=str, default="semseg_xy_4cls_less", help='Log path [default: None]')

args = parser.parse_args()
print(args)

logging.basicConfig(level=logging.INFO)

'''CREATE DIR'''
timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
experiment_dir = Path(args.exp_dir)
experiment_dir.mkdir(exist_ok=True)
if args.exp_name is None:
    experiment_dir = experiment_dir.joinpath(timestr)
else:
    experiment_dir = experiment_dir.joinpath(args.exp_name)
experiment_dir.mkdir(exist_ok=True)
checkpoints_dir = experiment_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)
log_dir = experiment_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True)

# make dataset and loader
train_dataset = CirclesSemseg(args.dataroot, args.labelroot, "train")
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
val_dataset = CirclesSemseg(args.dataroot, args.labelroot, "val")
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
num_train_batches = len(train_loader)
num_val_batches = len(val_loader)

# load model
vgg_model = VGGNet(model='vgg9', xyconv=args.xyconv, requires_grad=True).cuda()
model = FCNmy(pretrained_net=vgg_model, n_class=args.n_class).cuda()

# optimizer and criterion
optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
criterion = nn.CrossEntropyLoss()

trainlosses = []
vallosses = []
for epoch in range(args.epochs):
    loss_sum = 0
    loss_sum_100 = 0
    print("Starting Epoch: [%d/%d]" % (epoch, args.epochs))
    model.train()
    epoch_starttime = time.time()

    for batch_id, (img, label) in enumerate(train_loader):
        img, label = img.cuda(), label.cuda().long()

        optimizer.zero_grad()
        logits = model(img)

        loss = criterion(logits, label)
        trainlosses.append(loss.item())
        loss.backward()
        optimizer.step()

        loss_sum += loss
        loss_sum_100 += loss
        if (batch_id+1)%100 == 0:
            print("batch no. {}, loss {}".format(batch_id+1, loss_sum_100/100))
            #trainlosses.append(loss_sum_100/100)
            loss_sum_100 = 0

    print('Epoch %d Training mean loss: %f' % (epoch, loss_sum / num_train_batches))
    print("Epoch %d Train time %d seconds:" %(epoch, time.time()-epoch_starttime))

    with torch.no_grad():
        loss_sum = 0
        for batch_id, (img, label) in enumerate(val_loader):
            img, label = img.cuda(), label.cuda().long()
            logits = model(img)
            loss = criterion(logits, label)
            loss_sum += loss

    print('Epoch %d Validation mean loss: %f' % (epoch, loss_sum / num_val_batches))
    vallosses.extend([loss_sum / num_val_batches]*num_train_batches)

    savepath = str(checkpoints_dir) + '/model.pth'
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(state, savepath)

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.plot(trainlosses)
    ax.plot(vallosses)
    plt.show()
    fig.savefig(str(checkpoints_dir) + '/losses.png')
    np.save(str(checkpoints_dir) + '/trainlosses.npy', trainlosses)
    np.save(str(checkpoints_dir) + '/vallosses.npy', vallosses)

