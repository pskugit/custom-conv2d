import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules import Conv2d
from torch.nn.modules.conv import _ConvNd, init
from torch.nn.modules.utils import _single, _pair, _triple

class MyConv2d(torch.nn.modules.conv._ConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(MyConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def conv2d_forward(self, input, weight):
        return myconv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)


class Conv2dXY(torch.nn.modules.conv._ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, xy_bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2dXY, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        if xy_bias:
            self.xy_bias = Parameter(torch.Tensor(out_channels, in_channels, 2))
            self.init_xy_bias()
        else:
            self.xy_bias = None

    def conv2d_forward(self, input, weight):
        return conv2d_xy(input, weight, self.bias, self.xy_bias, self.stride,
                         self.padding, self.dilation)

    def forward(self, input):
        return self.conv2d_forward(input, self.weight)

    def init_xy_bias(self):
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.xy_bias, -bound, bound)


# Clean Convolution
def myconv2d(input, weight, bias=None, stride=(1,1), padding=(0,0), dilation=(1,1), groups=1):
    if dilation != (1, 1):
        raise NotImplementedError("conv2d_xy function only supports dilation of size 1")
    # pad image and get parameter sizes
    pad = [padding[0], padding[0], padding[1], padding[1]]
    input = F.pad(input=input, pad=pad, mode='constant', value=0)
    dh, dw = stride
    out_channels, in_channels, kh, kw = weight.shape
    batch_size = input.shape[0]
    # unfold input
    patches = input.unfold(2, kh, dh).unfold(3, kw, dw)
    h_windows = patches.shape[2]
    w_windows = patches.shape[3]
    patches = patches.expand(out_channels, *patches.shape)
    patches = patches.permute(1, 3, 4, 0, 2, 5, 6)
    patches = patches.contiguous()
    # use our filter and sum over the channels
    patches = patches * weight
    patches = patches.sum(-1)
    patches = patches.sum(-1)
    patches = patches.sum(-1)
    # add bias
    if bias is not None:
        bias = bias.expand(batch_size, h_windows, w_windows, out_channels)
        patches = patches + bias
    patches = patches.permute(0, 3, 1, 2)
    return patches


# xy Convolution
def conv2d_xy_0(input, weight, bias=None, xy_bias=None, stride=(1,1), padding=(1,1), dilation=(1,1)):
    if dilation != (1, 1):
        raise NotImplementedError("conv2d_xy function only supports dilation of size 1")
    # pad image and get parameter sizes
    pad = [padding[0], padding[0], padding[1], padding[1]]
    input = F.pad(input=input, pad=pad, mode='constant', value=0)
    dh, dw = stride
    out_channels, in_channels, kh, kw = weight.shape
    batch_size = input.shape[0]
    # unforld input according to filter
    patches = input.unfold(2, kh, dh).unfold(3, kw, dw)
    h_windows = patches.shape[2]
    w_windows = patches.shape[3]

    # make y and x "inputs"
    xy_map = get_xy_map(h_windows, w_windows)
    xy_map = xy_map.expand(batch_size, out_channels, in_channels, h_windows, w_windows, 2)
    xy_map = xy_map.permute(0,3,4,1,2,5)
    xy_map = xy_map * xy_bias
    xy_map = xy_map.sum(-1)
    xy_map = xy_map.to(xy_bias.device)

    # tile patches to get out_channels
    patches = patches.expand(out_channels, *patches.shape)

    # make patches *-compatible with weights
    patches = patches.permute(1, 3, 4, 0, 2, 5, 6)
    patches = patches.contiguous()

    # Now we can use our filters/weights
    patches = patches * weight

    # sum over kh, kw and in_channels
    patches = patches.sum(-1)
    patches = patches.sum(-1)

    # add xy bias to every filter weight matrix
    patches = patches + xy_map
    patches = patches.sum(-1)

    # add bias to every filter
    if bias is not None:
        bias_exp = bias.expand(batch_size, h_windows, w_windows, out_channels)
        patches = patches + bias_exp

    patches = patches.permute(0, 3, 1, 2)
    return patches.float()

# xy Convolution
def conv2d_xy(input, weight, bias=None, xy_bias=None, stride=(1,1), padding=(1,1), dilation=(1,1)):
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, in_channels, kh, kw = weight.shape
    out_h = int((in_h - kh + 2 * padding[0]) / stride[0] + 1)
    out_w = int((in_w - kw + 2 * padding[1]) / stride[1] + 1)

    unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
    inp_unf = unfold(input).double()
    w_ = weight.view(weight.size(0), -1).t().double()

    if bias is None:
        out_unf = inp_unf.transpose(1, 2).matmul(w_).transpose(1, 2)
    else:
        out_unf = (inp_unf.transpose(1, 2).matmul(w_) + bias).transpose(1, 2)

    if xy_bias is not None:
        xy_map = get_xy_map(out_h, out_w).cuda()
        xy_map = xy_map.permute(2, 0, 1)
        xy_map = xy_map.repeat(in_channels, 1, 1)
        xy_map = xy_map.view(in_channels * 2, -1).float()
        xy_bias = xy_bias.view(xy_bias.size(0), -1).t().float()
        xy_out = xy_map.transpose(0, 1).matmul(xy_bias).transpose(0, 1)
        out_unf += xy_out

    out = out_unf.view(batch_size, out_channels, out_h, out_w)
    return out.float()


def get_xy_map(out_h, out_w):
    y_map = torch.arange(out_h).float() / (out_h - 1)
    x_map = torch.arange(out_w).float() / (out_w - 1)
    x_map = x_map.expand(out_h, *x_map.shape)
    y_map = y_map.expand(out_w, *y_map.shape).transpose(1, 0)
    xy_map = torch.stack((x_map, y_map), dim=2)
    return xy_map