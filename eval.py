import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from models.fcn import VGGNet, FCNmy, cfg, ranges
from data.circles_semseg import CirclesSemseg

def label_to_img(label):
    cls_to_color = {
        1: (1,1,0),
        2: (0,0,1),
        3: (1,0,0)
    }
    img = np.zeros((*label.shape, 3))
    for i in range(1,4):
        img[np.where(label==i)] = cls_to_color[i]
    return img

parser = argparse.ArgumentParser(description='Evaluation')
parser.add_argument("--dataroot", default="/home/skudlik/xyexp/circle_4cls/circles/",
                    help="Folder containing my circles")
parser.add_argument("--labelroot", default="/home/skudlik/xyexp/circle_4cls/circles_labels/",
                    help="Folder containing labels")
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 2]')
parser.add_argument('--epochs', type=int, default=2, help='Epochs [default: 10]')
parser.add_argument('--n_class', type=int, default=4, help='Point Number [default: 3]')
parser.add_argument("--xyconv", action="store_true", help="")
parser.add_argument('--exp_dir', type=str, default="/home/skudlik/xyexp/experiments/", help='Log path [default: None]')
parser.add_argument('--exp_name', type=str, default="semseg_4cls", help='Log path [default: None]')
parser.add_argument('--eval_mode', type=str, default="quantitative", help='qualitative or quantitative eval [default: qualitative]')

args = parser.parse_args()
print(args)

experiment_dir = Path(args.exp_dir)
experiment_dir = experiment_dir.joinpath(args.exp_name)
checkpoints_dir = experiment_dir.joinpath('checkpoints/')
loadpath = str(checkpoints_dir) + '/model.pth'

# make dataset and loader
val_dataset = CirclesSemseg(args.dataroot, args.labelroot, "val")
val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
num_val_batches = len(val_loader)

# load model
vgg_model = VGGNet(model='vgg9', xyconv=args.xyconv, requires_grad=True).cuda()
model = FCNmy(pretrained_net=vgg_model, n_class=args.n_class).cuda()
checkpoint = torch.load(loadpath)
model.load_state_dict(checkpoint['model_state_dict'])

criterion = nn.CrossEntropyLoss()


num_classes = val_dataset.get_num_classes()

if args.eval_mode == "quantitative":
    # get metrics
    cls_correct = np.zeros(num_classes)
    cls_exist = np.zeros(num_classes)

    count = len(val_loader)
    with torch.no_grad():        
        for batch_id, (img, label) in enumerate(val_loader):
            #print("batch_id", batch_id)
            if batch_id >= count:
                break
            img, label = img.cuda(), label.cuda().long()
            logits = model(img)
            pred_class = logits.cpu().data.max(1)[1]

            label = label.cpu().numpy()
            pred_class = pred_class.numpy()
            for cls in range(num_classes):
                exist = np.sum(label == cls)
                overlap = pred_class[np.where(label == cls)] == cls
                correct = np.sum(overlap)
                cls_correct[cls] += correct
                cls_exist[cls] += exist
                # print("\tclass",cls, end="\t")
                #print("%d/%d"%(correct, exist))
        with np.errstate(divide='ignore'):
            result = np.nan_to_num(cls_correct/cls_exist)
            print("c", result)

        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.bar(range(num_classes),result, color=[(0,0,0),(1,1,0),(0,0,1),(1,0,0)])
        ax.set_xticks(range(num_classes))
        ax.set_ylabel('Recall')
        xlabels = ['class 0\n(background)', 'class 1\n(yellow-right)', 'class 2\n(yellow-left)', 'class 3\n(red-bottom)']
        xlabels = [xl+"\n\n"+str("%.5f" % recall) for xl, recall in zip(xlabels, result)]
        ax.set_xticklabels(xlabels)

        plt.show()


if args.eval_mode == "qualitative":
    count = 1
    # load 'count' batches and show results
    with torch.no_grad():
        for batch_id, (img, label) in enumerate(val_loader):
            if batch_id > count:
                break
            img, label = img.cuda(), label.cuda().long()
            logits = model(img)
            pred_class = logits.cpu().data.max(1)[1]
            loss = criterion(logits, label)
            print("loss",loss)
            for i in range(args.batch_size):
                plt.imshow(img[i].permute(1, 2, 0).cpu().numpy())
                plt.title("Input")
                plt.show()
                plt.imshow(label_to_img(label[i].cpu().numpy()))
                plt.title("Label")
                plt.show()
                plt.imshow(label_to_img(pred_class[i].numpy()))
                plt.title("Prediction")
                plt.show()


