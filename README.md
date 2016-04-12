# Face or Place Example
Deep learning example of faces vs places binary classification in Torch.

## Requirements
1) Up-to-date Torch installation with cutorch, nn, cudnn, cunn, matio, graphicsmagic, image etc.

2) Cuda 7.5 with CUDNN v4 was used, but older version would also work

3) Download data from the sources described below

4) Obtain imagenet pretrained models (optional)

## Obtaining the example training data
### 1) Faces
One of the suitable dataset of faces-in-the-wild can be obtain from the following source:
http://vis-www.cs.umass.edu/lfw/

The dataset contains 13233 images of 5681 people from the web oout of which 1680 people are pictured more than once. The images are resized and cropped to 250x250 pixels. The images corresponding to each person are stored in a seperate folder. For the purpose of this example you will need to flatten the directory hierarchy and can do so as follows:

```
#!bash
find images/ -type f -print0 | xargs -0 mv -t flattened/
```

### 2) Places
One of the suitable dataset of places can be found here:
http://places.csail.mit.edu/

The training and validation set of this dataset is very large, more than a million images. We have instead used the testing set of these data, which is around 40k images, out of which the code used up to 14k so as not to cause imbalance in the classes.

## Obtaining pretrained models
Imagenet pretrained models can be obtained from the Visual Geometry Group website: http://www.vlfeat.org/matconvnet/pretrained/

It is not necessary to use pretrained models and these networks can usually be trained from scratch. However, loading a pretrained model speeds-up the training and usually leads to higher performance. The imagenet pretrained models are especcially usefull, due to the fact that they are trained on a huge dataset (1.2 million images), with a very large number of classes (1000), which leads to generic convolutional features that can be used in many other tasks.

In this example we have implemented the following models:

1) VGG-F (fast), VGG-M (medium) and VGG-S (slow) models with Local Response Normalisation (LRN), also referred to as Same Response Normalisation. They are modified versions of AlexNet and were used in the "Return of the devil in the details". They are challower and smaller that GoogleNet or the classic versions of VGG net.

2) VGG-A, VGG-B, VGG-D, VGG-E, without any normalisation (neighter LRN nor BN). These are very deep models with a small receptive field (3x3) and are documented to preserve details more that AlexNet (receptive field 11x11).
VGG-D corresponds to "vgg-verydeep-16", but can only fit in the memory of a Titan X. As we had a GTX 980Ti, we could not run this example with VGG-D. We were also unable to find a pretrained version of VGG-A or VGG-B, but it is possible to initialise them from their larger counterparts.

To run this example ou should download the pretrained models and point to them by setting the self.loadPretrained variable of the options class or setting it 'none' is no pretrained model is to be loaded.

## How to run example
You can run training and validation as follows:

```
#!bash
th main.lua
```

But first you need to fix the paths and put the data in the right place. To do that you need to edit the ./options/opts.lua file accordingly. More specifically:

1) You need to put the data into a folder with the same name as the name of the dataset, specified in self.dataset variable of the options class.

2) The images should be saved in seperate folder per class.

3) You need to specify the location of the dataset by setting the self.defaultDir variable in the options class.

After one or more epochs and when the accuracy has reached the desired level, you can run the evaluation demo on the test images as follows:

```
#!bash
qlua evaluation.lua
```

Please remember to specify the model that is going to be loaded within the script.

## Performance
As the task in question is extremely simple, the example reaches 97.5% validation set accuracy after only one sweep of the data (epoch). This is using one of the shallowest and simplest models, namely the VGG-F with LRN.

## Code Structure
The code is mostly written from scratch, it is heavily influenced by the structure of the Facebook exmples, but it is substantially different and simpler. We have followed best practices to the extent that this was practical. It is also written in a generic a manner, meaning that it can work as is with any number of categories.
