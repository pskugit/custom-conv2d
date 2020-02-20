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

