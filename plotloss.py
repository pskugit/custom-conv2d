import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# define vanilla experiment
experiment_dir = Path("/home/skudlik/xyexp/experiments/")
exp_name = "semseg_4cls"
experiment_dir = experiment_dir.joinpath(exp_name)
checkpoints_dir = experiment_dir.joinpath('checkpoints/')
trainlosses_normal = np.load(str(checkpoints_dir) + '/trainlosses.npy', allow_pickle=True)
vallosses_normal = np.load(str(checkpoints_dir) + '/vallosses.npy',  allow_pickle=True)

# define coordinate experiment
experiment_dir = Path("/home/skudlik/xyexp/experiments/")
exp_name = "semseg_xy_4cls_less"
experiment_dir = experiment_dir.joinpath(exp_name)
checkpoints_dir = experiment_dir.joinpath('checkpoints/')
trainlosses_xy = np.load(str(checkpoints_dir) + '/trainlosses.npy', allow_pickle=True)
vallosses_xy = np.load(str(checkpoints_dir) + '/vallosses.npy',  allow_pickle=True)

start = 500     # cut first n values
mavg_N = 40     # moving average
trainlosses_normal_mavg = np.convolve(trainlosses_normal, np.ones((mavg_N,))/mavg_N, mode='valid')
trainlosses_xy_mavg = np.convolve(trainlosses_xy, np.ones((mavg_N,))/mavg_N, mode='valid')

# plot zoomed losses
fig, ax = plt.subplots(nrows=1, ncols=1)
ax.plot(trainlosses_normal_mavg[start:], label='Train Loss ClassicConv')
ax.plot(vallosses_normal[start+mavg_N:-mavg_N], label='Val Loss ClassicConv')
ax.plot(trainlosses_xy_mavg[start:], label='Train Loss CoordConv')
ax.plot(vallosses_xy[start+mavg_N:-mavg_N], label='Val Loss CoordCornv')
plt.legend(loc="upper right")
plt.show()

# plot full losses
fig, ax = plt.subplots(nrows=1, ncols=1)
start = 0
ax.plot(trainlosses_normal_mavg[start:], label='Train Loss ClassicConv')
ax.plot(vallosses_normal[start+mavg_N:-mavg_N], label='Val Loss ClassicConv')
ax.plot(trainlosses_xy_mavg[start:], label='Train Loss CoordConv')
ax.plot(vallosses_xy[start+mavg_N:-mavg_N], label='Val Loss CoordCornv')
plt.legend(loc="upper right")
plt.show()