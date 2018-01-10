## Convolutional Neural Networks Tutorial
### Women in Machine Learning and Data Science, LinkedIn, 18 January 2018


### Usage

**Installation**

1. Install [Anaconda or Miniconda](https://conda.io/docs/user-guide/install/index.html)
2. Run the following commands

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

**To train a VGG like net**
```bash
python main.py --nettype 1
```

**To train a Residual Network**
```bash
python main.py --nettype 2
```

### Results after 20 epochs:
1. Basic CNN:
  - Train: 55%
  - Test: 55%
2. VGG like net
  - Train: 85%
  - Test: 80%
3. ResNet
  - Train:
  - Test:

### Sources

- [Pytorch CIFAR10 tutorial](https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/cifar10_tutorial.py)
- [VGG paper](https://arxiv.org/pdf/1409.1556.pdf)
- [ResNet paper](https://arxiv.org/abs/1512.03385)
