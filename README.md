# custom-conv2d
A study for a custom convolution layer in which the x and y components of an image pixel are added to the kernel inputs.

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

![](/images/cc.png?raw=true "Dataset")




