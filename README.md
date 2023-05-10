# Image Classification Project

## Dataset

The dataset used for this project is the Open Images Dataset, which can be accessed [here](https://storage.googleapis.com/openimages/web/visualizer/index.html?type=detection). It contains a wide range of images across various categories.

## Tasks

### Preparation
To begin with, the dataset is split into a 70/30 train/test split to ensure a separate evaluation set. This split allows for training the model on the training set and evaluating its performance on unseen data in the test set.

### Training VGG19 from Scratch
The first task involves training a VGG19 network from scratch. The VGG19 model is initialized with random weights, and the network is trained on the training set. After training, the accuracy of the model is estimated using the test set.

### Transfer Learning
In the next task, transfer learning is applied using an ImageNet pre-trained VGG19 network. The weights of the pre-trained network are loaded, and the last layer of the network is replaced with a new softmax layer for the three chosen classes. The model is then fine-tuned on the training set, and its accuracy is estimated using the test set. The differences in loss and accuracy between the plain (from scratch) and pre-trained networks are analyzed over the first 10 epochs.

### Data Cleansing
In this task, "bad" images from the dataset are removed to improve the quality of the training data. The specific criteria for identifying bad images may vary, but common approaches include removing images with low resolution, high noise, or incorrect labeling. The number of bad images removed and the results obtained are discussed.

### Data Augmentation
To further enhance the model's performance, data augmentation techniques are applied to the training set. The following augmentations are utilized:
- Random flip: Randomly flips the images horizontally or vertically.
- Random contrast: Adjusts the contrast of the images randomly.
- Random translation: Shifts the images horizontally or vertically by a random amount.
 
The VGG19 model is trained again on the augmented training set, and the results are discussed.

### Modifying VGG19 Architecture
In this task, the VGG19 architecture is modified by adding layers after the block4_conv4 layer (25, 25, 512). The modifications include:
- Adding an inception layer with dimensionality reduction. The number of output filters is set to 512, and the dimensionality reduction is performed using 1x1 convolutional layers with user-defined values.
- Adding a convolutional layer with a kernel size of 1x1, 1024 filters, valid padding, stride 1, and a leaky ReLU activation.
- Adding another convolutional layer with a kernel size of 3x3, 1024 filters, same padding, stride 1, and a ReLU activation.

The conv2 layers and the layers before them are frozen during training.

### Testing with Custom Images
To assess the model's performance, a few custom images are tested, and the results are presented.

### Answering Questions
Finally, several questions are addressed:
- What accuracy can be achieved? What is the accuracy of the train vs. test set?
- On what infrastructure was the model trained? What is the inference time?
- What are the number of parameters of the model?
- Which categories are most likely to be confused by the algorithm? The results are presented in a confusion matrix.
