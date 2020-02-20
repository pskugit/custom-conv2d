import os
import cv2
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision.transforms import ToTensor
from torch.utils.data import Dataset, DataLoader


class CirclesSemseg(Dataset):
    """
    :param
    split: string
        can be 'train', 'val' or 'test'
    data_root: string
        root path to the datasets main folder
    label_root: string
        root path to the label folder

    Returns an image and a segmentation map
    """
    def __init__(self, data_root, label_root, split, get_paths=False):
        super().__init__()
        print("data_root",data_root)
        print("label_root",label_root)
        self.data_root = data_root
        self.label_root = label_root
        self.file_list = []
        self.num_classes = 3
        self.get_paths = get_paths
        self.split = split
        self.file_list = os.listdir(data_root+"/"+split)
        self.label_list = os.listdir(label_root+"/"+split)
        assert len(self.label_list) == len(self.file_list)
        self.img_size = np.array(cv2.imread(self.data_root+"/"+self.split+"/"+self.file_list[0])).shape[1]


    def __getitem__(self, index):
        img_name = self.file_list[index]
        rgb_img = cv2.imread(self.data_root+"/"+self.split+"/"+img_name)
        rgb_img = ToTensor()(rgb_img)

        label_name = img_name.split("_")[0]+"_label.npy"
        label = np.load(self.label_root+"/"+self.split+"/"+label_name)
        return rgb_img, label

    def __len__(self):
        return len(self.file_list)

    def get_label_r_weigths(self):
        return np.array([1,1,1])   # TODO: set proper weights

    def get_num_classes(self):
        """Get the number of classes for this dataset"""
        return 4

    def get_index_fullpaths(self, idx):
        pass

    def crop_rgb(self, img):
        return img


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load a Circle Semseg Dataset')
    parser.add_argument("--v", action="store_true", help="set to have verbose outputs")
    parser.add_argument("--dataroot", default="/home/skudlik/xyexp/circle_small/circles/", help="Folder containing my circles")
    parser.add_argument("--labelroot", default="/home/skudlik/xyexp/circle_small/circles_labels/", help="Folder containing labels")
    parser.add_argument("--split", default="train", help="Split")
    args = parser.parse_args()
    print(args)

    logging.basicConfig(level=logging.INFO)
    batch_size = 2
    # make dataset and loader
    dataset = CirclesSemseg(args.dataroot, args.labelroot, args.split)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    train_iter = iter(train_loader)
    for i, data in enumerate(train_loader):
        if i >= 1:
            break
    img, label = data
    print("img",img.shape, type(img))
    print(img)
    print("label",label.shape, type(label))
    for i in range(batch_size):
        plt.imshow(img[i].permute(1,2,0).numpy())
        plt.show()
        plt.imshow(label[i].numpy())
        plt.show()
