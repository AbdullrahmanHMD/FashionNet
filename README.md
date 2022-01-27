# FashionNet
A 2 layer convolutional neural network implemented with PyTorch trained on FashionMNIST dataset.

## The Architecture of FashionNet:
### 1. First Layer:

* A 2d convolution:
  * With a kernel size of 3 x 3.
  * A padding of 1
  * A stride of 1
  * And 24 filters.
* Batch normalization
* Dropout:
  *  With 0.5 probabilty of dropping a neuron.
* Activation Function:
  * ReLU
  * The weights are initialized using kaiming normal.
  * The biases are initialized to a constant of 0.

### 2. Second Layer:

* A 2d convolution:
  * With a kernel size of 3 x 3.
  * A padding of 1
  * A stride of 2
  * And 64 filters.
* Batch normalization
* Dropout:
  *  With 0.25 probabilty of dropping a neuron.
* Activation Function:
  * ReLU
  * The weights are initialized using kaiming normal.
  * The biases are initialized to a constant of 0.

### 3. Fully Connected Layer:
* The feature map is flattened before feeding it to this layer
* This layer uses the default initialization for the weights and biases.

## Dataset:
The dataset used for training this CNN is Fashion MNIST dataset.
The dataset was downloaded using torchvision.

## Results:
This CNN acheived an validation accuracy of 89.24%
