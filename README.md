## Convolutional Neural Networks Tutorial
### Women in Machine Learning and Data Science, LinkedIn, 18 January 2018

Code to accompany a talk on Convolutional neural networks. It provides
- implementations of a basic CNN, VGG-like CNN, and Residual Network in pyTorch
- pre-trained model of each type
- script to train each model type on CIFAR10
- script to evaluate a trained model on CIFAR10

Models will automatically be trained on a GPU if one is available.

*Please Note: Slides will be posted after the talk*

### Usage

**Installation**

1. Install [Anaconda or Miniconda](https://conda.io/docs/user-guide/install/index.html)
2. Run the following commands to install the relevant libraries in an anaconda environment. `source activate env_name` activates the environment.

```bash
conda env create -n cnn_tutorial -f environment.yml
source activate cnn_tutorial
```

**Create data and model folders**
```bash
mkdir data
mkdir models
```

**To train a basic CNN with default settings**
```bash
python main.py
```

This will take approximately 1 minute per epoch to train on a laptop, CPU only.

**To train a VGG like net using the default settings**
```bash
python main.py --nettype 1
```

This will take approximately 6 - 8 minutes per epoch to train on a laptop, CPU only. It takes < 1 minute per epoch on a good GPU.

**To train a Residual Network**
```bash
python main.py --nettype 2
```

This will take approximately 12 - 14 minutes per epoch to train on a laptop, CPU only. It takes < 1 minute per epoch on a good GPU.

**Use the command line options to change**
- Batch size
- Learning rate
- Number of training epochs
- Network type
- Dropout rate
- Number of initial filters

**To evaluate a trained model on the training and test dataset**
```bash
python run_trained_models.py --nettype x --model_path path/to/saved_model
```
x = 0 (basic CNN), 1 (VGG like net), 2 (Residual network)

Please note: The `nettype` must match the saved model. Additionally for the VGG Like net and the Residual Network the number of filters and dropout percentage must match between the initialized and saved model. If you are using a Residual network the number of residual blocks must also match.

Training and testing the models with the default settings will ensure the architectures match.

Alternatively load one of the trained models provided in the `models/` folder. For example,

```bash
python run_trained_models.py --nettype 1 --model_path models/VGGlike_30.pth
```

### Results after 30 epochs:
1. Basic CNN:
  - Train: 70%
  - Test: 63%
2. VGG like net
  - Train: 87%
  - Test: 77%
3. ResNet
  - Train: 91%
  - Test: 82%

**Training curves**

![Training data](train.png#center)

![Test data](test.png#center)

### Sources

- [Pytorch CIFAR10 tutorial](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py)
- [VGG paper](https://arxiv.org/pdf/1409.1556.pdf)
- [ResNet paper](https://arxiv.org/abs/1512.03385)
