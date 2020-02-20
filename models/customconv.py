import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules.conv import _ConvNd, init
from torch.nn.modules.utils import _single, _pair, _triple

class MyConv2d(torch.nn.modules.conv._ConvNd):
    """
    Implements a standard convolution layer that can be used as a regular module
    """
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
    """
    Implements a convolution layer with coordinate bias that can be used as a regular module.
    The coordinate bias is defined as a learnable Parameter and intialized the same way as the other kernel weights.
    If xy_bias=False it behaves the same as a standard convolution.
    """
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
            self.xy_bias = Parameter(torch.Tensor(out_channels, 2))
            self.init_xy_bias()
            print("Added an coordinate_bias with shape:",self.xy_bias.shape)
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
    """
    Function to process an input with a standard convolution
    """
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, in_channels, kh, kw = weight.shape
    out_h = int((in_h - kh + 2 * padding[0]) / stride[0] + 1)
    out_w = int((in_w - kw + 2 * padding[1]) / stride[1] + 1)
    unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
    inp_unf = unfold(input)
    w_ = weight.view(weight.size(0), -1).t()
    if bias is None:
        out_unf = inp_unf.transpose(1, 2).matmul(w_).transpose(1, 2)
    else:
        out_unf = (inp_unf.transpose(1, 2).matmul(w_) + bias).transpose(1, 2)
    out = out_unf.view(batch_size, out_channels, out_h, out_w)
    return out.float()


# xy Convolution
def conv2d_xy(input, weight, bias=None, xy_bias=None, stride=(1, 1), padding=(1, 1), dilation=(1, 1)):
    """
    Function to process an input with a coordinate convolution
    Uses (out_channel, 2) as xy_bias
    """
    # calculate parameters
    batch_size, in_channels, in_h, in_w = input.shape
    out_channels, in_channels, kh, kw = weight.shape
    out_h = int((in_h - kh + 2 * padding[0]) / stride[0] + 1)
    out_w = int((in_w - kw + 2 * padding[1]) / stride[1] + 1)
    # unfold input according to kernel
    unfold = torch.nn.Unfold(kernel_size=(kh, kw), dilation=dilation, padding=padding, stride=stride)
    inp_unf = unfold(input).double()
    w_ = weight.view(weight.size(0), -1).t().double()
    # add regular bias
    if bias is None:
        out_unf = inp_unf.transpose(1, 2).matmul(w_).transpose(1, 2)
    else:
        out_unf = (inp_unf.transpose(1, 2).matmul(w_) + bias).transpose(1, 2)
    # add coordinate bias
    if xy_bias is not None:
        xy_map = get_xy_map(out_h, out_w).to(xy_bias.device)
        xy_map = xy_map.permute(2, 0, 1)
        xy_map = xy_map.view(2, -1).float()
        xy_b_ = xy_bias.view(xy_bias.size(0), -1).t().float()
        xy_out = xy_map.transpose(0, 1).matmul(xy_b_).transpose(0, 1)
        out_unf += xy_out
    # reshape output
    out = out_unf.view(batch_size, out_channels, out_h, out_w)
    return out.float()

def get_xy_map(out_h, out_w):
    """
    created coordinate map with the given shape.
    Returns map of shape (out_h, out_w, 2)
    """
    y_map = torch.arange(out_h).float() / (out_h - 1)
    x_map = torch.arange(out_w).float() / (out_w - 1)
    x_map = x_map.expand(out_h, *x_map.shape)
    y_map = y_map.expand(out_w, *y_map.shape).transpose(1, 0)
    xy_map = torch.stack((x_map, y_map), dim=2)
    return xy_map