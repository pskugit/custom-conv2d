# custom-conv2d
> A study for a custom convolution layer in which the x and y components of an image pixel are added to the kernel inputs.

## Prerequisites

The code is based on pytorch and numpy.
Dataset creation uses opencv-pyhon and sckikit-image.
Visualizations use matplotlib.pyplot.

```
$ pip install torch
$ pip install numpy 
$ pip install pathlib
$ pip install argparse
$ pip install matplotlib
$ pip install opencv-pyhon
$ pip install sckikit-image
```

## Intro 

One of the reasons for the success of modern CNN architectures in image processing is their ability to be translation invariant. Meaning, that they are able to locate objects or relevant shapes within an image independent of where exactly in the image it is located. 
While this property is certainly wanted in many cases, one may also see tasks where it would be beneficial to have the position of a certain detevtion influence the networks predicition. This possibly includes many tasks where the camera position is fixed and the position of elements within the scene carries semantic information. 
In this experiment, I introduce a dataset that may be interpreted as a very loose abstraction for the "lane detection" task in the field of autonomous driving. I will then explain an extension to the vanilla convolutional layer that shows significant improvements in terms of learning speed and quality by adding coordinate information as input to the convolutional kernel. 

## The Dataset

![](/images/data.png?raw=true "Dataset")

The Dataset contains images of colored circles on a perlin noise background. The circles belong to three distinct classes (1,2,3) plus a background class 0.

**Yellow Circles:** A yellow circle belongs to class 1 if its center lays in the right half of the image. A yellow circle with its center in the left part of the image is part of class 2.
A yellow circle may be thought of as left and right lane markings. While both share the same visiual characteristics, a right lane marking can (per definition) never occur in the left part of an image and vice versa. 

**Red circles:** A red circle is part of class 3 only if its center is in the bottom half of the image. Otherwise it shall simply be treated as background (class 0).
Red circles thus may be interpreted as cars or street in an autonomous driving setting. As even though the sky may sometimes appear to share street features, we can be confident that the street will only ever be found in the bottom half of the image. 


## Extending the Convolution

In their 2018 paper Lui st al. identified what they called "An intriguing failing of convolutional neural networks and the CoordConv solution" (https://arxiv.org/pdf/1807.03247.pdf). Expanding on their works which were mostly based on coordinate regression, this repository provides a PyTorch implementation of a slightly more efficient approach with mathematically similiar properties. Lui et al.'s approach is based on concatenating two more channels to the convolution input which contain hard coded values of x and y coordinates. These channels are then treated just like the other input channels and convolved with the same sized filter kernels. 
This repositories approach first calculates the output size of the convolutional layer and then constructs similiar coordinate channels whose entries are the relative x and y positions of the respective filter kernel center. Opposed to using same sized kernel parameters to convolve the coordinate maps, we will only use a single value per coodinate channel and Filter. As such we can think of the new parameters as a Coordinate Bias for each Convolutional Filter. In settings with standard 3x3 Filterkernels this new operation reduces the additional parameters by roughly 90% (increased benefit with increased filter kernel size).

![](/images/cc.png?raw=true "Convolution with coordinate bias")

The Figure above shows the process of adding the Coordinate Bias. The left part is the standard convolutional operation over a 7x7x3 (H_in,W_in,in_channels) input image with TWO 3x3 Kernel, stride=3, padding=1. This produces a resulting featuremap of size 3x3x2 (H_out,W_out,num_filters). 
The right part shows the constructed 3x3x2 Coordinate maps, where one contains the relative y and the other the relative x components. These Coordinate Maps are multiplied with the learned Coordinate Bias values, resulting in a feature map that has the same dimensions as the one from the standard convolution path. Both outputs will finally be summed.


## Evaluation (Quantitative)

For the experiment, a standard Fully Convolutional Network Architecture with skip-connections was used. The encoder side is a small Network derived from VGG11 (then called VGG9), and the decoder is a matching deconvolution path. 
For the vanilla Network, the regular torch.nn.Con2d layer has been used, while for the second nework, a Coordinate Bias was added to all convolution layers in the encoding path. The decoding path remained the same for both networks.

![](/images/eval.png?raw=true "Qualitative evaluation")
_right: zoomed version_


Both networks were trained on the Dataset with 5000 images for 2 epochs, each. The Validation loss was calculated after each epoch (thus two times). As can be seen in the loss curves, the network with coordinate bias learns much quicker and plateaus on a significantly lower loss compared to the vanilla network. 


## Evaluation (Qualitative)

Vanilla Network Predictions (Input, Label, Pred)          |  Coordinate Network Predictions (Input, Label, Pred)
:------------------------------------:|:------------------------------------:
![](/images/vanilla.png?raw=true "")  |  ![](/images/cc_img.png?raw=true "") 

As for the qualitative evaluation, one can easiliy recognize that the vannilla network indeed learns that yellow circles belong to either class 1 or class 2, however it struggles to assign the pixels correctly with respect to their x-position. 
The same can been seen for the red circle, for which the network seems usure if it belongs to the background class (like red circles in the top half of the image) or the actual class 3 (like red circles in the bottom half should). 

Contrastly, the network with the coordinate bias seems to utilize the additional inputs to recreate the labels almost perfectly after training for the same number of epochs. 


## Open Questions

* Most modern achitectures do not use a bias in their conv2d layers as such will have a gradient of zero after being passed through a BatchNorm layer. It should be investigated if the same happens to the Coordinate Bias.
* Clearly the Coordinate enriched networks outperforms the vanilla version. Might this performance increase, however, only be caused by having additional parameters?
* Further it is also to be researche if and how the intriduced approach differs from the 2018 approach of Lui et al.


### Running the scrips

-- detailed instructions follow --
