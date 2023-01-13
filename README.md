# FashionNet

A 2 layer convolutional neural network implemented with PyTorch trained on FashionMNIST dataset.

# The Architecture of FashionNet:
## 1. First Layer:

* A 2d convolution:
  * With a kernel size of 3 x 3.
  * A padding of 1
  * A stride of 1
  * And 24 filters.
* Batch normalization
* Dropout:
  *  With 0.5 probabilty of dropping a neuron.
* Activation Function:
  * LeakyReLU with negative slope of 0.1
  * The weights are initialized using kaiming normal.
  * The biases are initialized to a constant of 0.

## 2. Second Layer:

* A 2d convolution:
  * With a kernel size of 3 x 3.
  * A padding of 1
  * A stride of 2
  * And 64 filters.
* Batch normalization
* Dropout:
  *  With 0.5 probabilty of dropping a neuron.
* Activation Function:
  * LeakyReLU with negative slope of 0.1
  * The weights are initialized using kaiming normal.
  * The biases are initialized to a constant of 0.

## 3. Fully Connected Layer:
* The feature map is flattened before feeding it to this layer
* This layer uses the default initialization for the weights and biases.

# Training:

## 1. Optimizer:
* AdamW.
* With a learning rate of 3e-3.
* And a weight decay of 1e-5.
* With amsgrad.

## 2. Criterion (Loss function):
* Cross entropy loss.

## 3. Epochs:
* The model was trained for 60 epochs

## 4. Learning rate scheduler:
* StepLR.
* With a step size of 25.
* And a gamma of 0.1


# Evaluation:
* Accuracy:

| Train | Validation | Test |
|-------|-------------|------|
| 92.22%| 89.35%      | 90.30%|

* Precision, recall, and f1-score:

| classes | precision | recall | f1-score | support |
|---------|-----------|--------|----------|---------|
| T-shirt/top       | 0.87      | 0.87   | 0.87     | 408     |
| Trouser       | 0.99      | 0.97   | 0.98     | 395     |
| Pullover       | 0.86      | 0.87   | 0.87     | 383     |
| Dress      | 0.91      | 0.92   | 0.91     | 386     |
| Coat       | 0.84      | 0.88   | 0.86     | 402     |
| Sandal       | 0.98      | 0.96   | 0.97     | 408     |
| Shirt       | 0.76      | 0.70   | 0.73     | 406     |
| Sneaker       | 0.92      | 0.94   | 0.93     | 389     |
| Bag       | 0.98      | 0.98   | 0.98     | 391     |
| Ankle boot       | 0.94      | 0.93   | 0.94     | 432     |


# Dataset:
The dataset used for training this CNN is Fashion MNIST dataset.
The dataset was downloaded using torchvision.
* The dataset consists of images of clothings.
* There are 10 different classes in the dataset.
* A transform is applied on the dataset to make sure that all images are of the same size which is 28 x 28

