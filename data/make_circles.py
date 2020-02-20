import cv2
import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt

from skimage import img_as_ubyte
from itertools import product, count


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

def generate_unit_vectors(n):
    'Generates matrix NxN of unit length vectors'
    phi = np.random.uniform(0, 2*np.pi, (n, n))
    v = np.stack((np.cos(phi), np.sin(phi)), axis=-1)
    return v

# quintic interpolation
def qz(t):
    return t * t * t * (t * (t * 6 - 15) + 10)
# cubic interpolation
def cz(t):
    return -2 * t * t * t + 3 * t * t

def generate_2D_perlin_noise(size, ns):
    nc = int(size / ns)  # number of nodes
    grid_size = int(size / ns + 1)  # number of points in grid
    # generate grid of vectors
    v = generate_unit_vectors(grid_size)
    # generate some constans in advance
    ad, ar = np.arange(ns), np.arange(-ns, 0, 1)
    # vectors from each of the 4 nearest nodes to a point in the NSxNS patch
    vd = np.zeros((ns, ns, 4, 1, 2))
    for (l1, l2), c in zip(product((ad, ar), repeat=2), count()):
        vd[:, :, c, 0] = np.stack(np.meshgrid(l2, l1, indexing='xy'), axis=2)
    # interpolation coefficients
    d = qz(np.stack((np.zeros((ns, ns, 2)),
                     np.stack(np.meshgrid(ad, ad, indexing='ij'), axis=2)),
           axis=2) / ns)
    d[:, :, 0] = 1 - d[:, :, 1]
    # make copy and reshape for convenience
    d0 = d[..., 0].copy().reshape(ns, ns, 1, 2)
    d1 = d[..., 1].copy().reshape(ns, ns, 2, 1)
    # make an empy matrix
    m = np.zeros((size, size))
    # reshape for convenience
    t = m.reshape(nc, ns, nc, ns)
    # calculate values for a NSxNS patch at a time
    for i, j in product(np.arange(nc), repeat=2):  # loop through the grid
        # get four node vectors
        av = v[i:i+2, j:j+2].reshape(4, 2, 1)
        # 'vector from node to point' dot 'node vector'
        at = np.matmul(vd, av).reshape(ns, ns, 2, 2)
        # horizontal and vertical interpolation
        t[i, :, j, :] = np.matmul(np.matmul(d0, at), d1).reshape(ns, ns)
    return m

### SET PARAMETERS HERE

savepath = "/home/skudlik/xyexp/circle_4cls/"
split = "val"
show = False
num = 5000 if split == "train" else 2500
num = 5 if show else num

for i in tqdm.tqdm(range(num)):
    img_size = 128
    # noise
    intensity = 0.5
    r = generate_2D_perlin_noise(img_size, int(img_size/4))
    r = (r+abs(np.min(r))) / (abs(np.min(r))+np.max(r))
    g = generate_2D_perlin_noise(img_size, int(img_size/4))
    g = (g+abs(np.min(g))) / (abs(np.min(g))+np.max(g))
    b = generate_2D_perlin_noise(img_size, int(img_size/4))
    b = (b+abs(np.min(b))) / (abs(np.min(b))+np.max(b))
    img = np.stack([r,g,b], axis=2) * intensity
    label = np.zeros((img_size,img_size))
    num_circles = 3
    hasright = False
    hasleft = False
    for c in range(num_circles):
        color = (1,1,0) if random.random() < 0.5 else (1,0,0)     # color is yellow or red
        radius = int(img_size/12 + (img_size/8)*random.random())
        center_x = int(radius + random.random() * ((img_size//2)-radius))
        side = int(2* random.random())
        center_x = int((radius + (img_size//2*side)) + random.random() * ((img_size//2)-2*radius))
        center_y = int(radius + random.random() * (img_size-radius))
        cv2.circle(img, (center_x,center_y), radius, color, thickness=-1, lineType=8, shift=0)
        circle_on_zero = cv2.circle(np.zeros((img_size,img_size)), (center_x,center_y), radius, 1, thickness=-1, lineType=8, shift=0)

        class_ = 0
        if color == (1, 1, 0):   # if color is yellow
            if center_x > img_size // 2:    # and on right side
                class_ = 1
            else:                           # and on left side
                class_ = 2
        if color == (1, 0, 0):   # if color is red
            if center_y > img_size // 2:    # and in top half
                class_ = 3
        label[circle_on_zero.astype(bool)] = class_

    if show:
        plt.imshow(img)
        plt.show()

        plt.imshow(label_to_img(label))
        plt.show()
    else:
        np.save(savepath+"circles_labels/"+split+"/"+str(i)+"_label.npy", label)
        cv2.imwrite(savepath+"circles/"+split+"/"+str(i)+"_img.png", img_as_ubyte(img))

