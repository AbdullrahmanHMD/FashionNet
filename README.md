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

## Training:

### 1. Optimizer:
* Adam.
* With a learning rate of 1e-3.
* And a weight decay of 5e-4.
* With amsgrad.

### 2. Criterion (Loss function):
* Cross entropy loss.

### 3. Epochs:
* The model was trained for 30 epochs
* In each epoch the model parameters are saved so they can be used for evaluation.

## Evaluation:
* The evaluation is done by loading the saved parameters from training and generating predictions.
* A softmax is applied on the predictions to get the predicted label.
* The predicted labels are then, used to evaluate the accuracy of the model by combaring them with actual labels and counting how many predicted labels were correct.

## Dataset:
The dataset used for training this CNN is Fashion MNIST dataset.
The dataset was downloaded using torchvision.
* The dataset consists of images of clothings.
* There are 10 different classes in the dataset.
* A transform is applied on the dataset to make sure that all images are of the same size which is 28 x 28

## Results:
This CNN acheived an validation accuracy of 89.24%
