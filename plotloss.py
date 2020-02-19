import numpy as np
import torch
from pathlib import Path
import matplotlib.pyplot as plt

experiment_dir = Path("/home/skudlik/xyexp/experiments/")
exp_name = "semseg_4cls"
experiment_dir = experiment_dir.joinpath(exp_name)
checkpoints_dir = experiment_dir.joinpath('checkpoints/')
trainlosses_normal = np.load(str(checkpoints_dir) + '/trainlosses.npy', allow_pickle=True)
vallosses_normal = np.load(str(checkpoints_dir) + '/vallosses.npy',  allow_pickle=True)

experiment_dir = Path("/home/skudlik/xyexp/experiments/")
exp_name = "semseg_xy_4cls"
experiment_dir = experiment_dir.joinpath(exp_name)
checkpoints_dir = experiment_dir.joinpath('checkpoints/')
trainlosses_xy = np.load(str(checkpoints_dir) + '/trainlosses.npy', allow_pickle=True)
vallosses_xy = np.load(str(checkpoints_dir) + '/vallosses.npy',  allow_pickle=True)



fig, ax = plt.subplots(nrows=1, ncols=1)
start = 500

mavg_N = 40
trainlosses_normal_mavg = np.convolve(trainlosses_normal, np.ones((mavg_N,))/mavg_N, mode='valid')
print("done")
vallosses_normal_mavg = np.convolve(vallosses_normal, np.ones((mavg_N,))/mavg_N, mode='valid')
print("done")
trainlosses_xy_mavg = np.convolve(trainlosses_xy, np.ones((mavg_N,))/mavg_N, mode='valid')
print("done")
vallosses_xy_mavg = np.convolve(vallosses_xy, np.ones((mavg_N,))/mavg_N, mode='valid')
print("done")
ax.plot(trainlosses_normal_mavg[start:])
ax.plot(vallosses_normal_mavg[start:])
ax.plot(trainlosses_xy_mavg[start:])
ax.plot(vallosses_xy_mavg[start:])

#ax.plot(trainlosses_normal[start:])
#ax.plot(vallosses_normal[start:])
#ax.plot(trainlosses_xy[start:])
#ax.plot(vallosses_xy[start:])

plt.show()