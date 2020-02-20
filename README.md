# custom-conv2d
A study for a custom convolution layer in which the x and y components of an image pixel are added to the kernel inputs.

One of the reasons for the success of modern CNN architectures in image processing is their ability to be translation invariant. Meaning, that they are able to locate objects or relevant shapes within an image independent of where exactly in the image it is located. 
While this property is certainly wanted in many cases, one may also see tasks where it would be beneficial to have the position of a certain detevtion influence the networks predicition. This possibly includes many tasks where the camera position is fixed and the position of elements within the scene carries semantic information. 
In this experiment, I introduce a dataset that may be interpreted as a very loose abstraction for the "lane detection" task in the field of autonomous driving. I will then explain an extension to the vanilla convolutional layer that shows significant improvements in terms of learning speed and quality by adding coordinate information as input to the convolutional kernel. 

## The Dataset

