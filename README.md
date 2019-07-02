# AIDL_2019_project

## Introduction

In this project 

In the second part of the project, we splitted the group individually and each member of the group had to implement a siamese network with two VGG 

## Development

https://pjreddie.com/darknet/yolo/

## Architecture



## Hyperparameters

In order to test the training efficiency of the networks and compare between (...), several hyperparameters were tunned and its effect 

**Learning rate:** The learning rate determines the step size towards the minimum of the loss function during training, which ultimately dictates how fast the network learns. Here, a larger learning rate was selected when running experiments where the VGGs were not pretrained, and smaller learning rate was used in experiments where pretrained VGGs were used.

**Weight decay:** This parameter prevents weights from exploding, that is from having values that are too large. It does so by multiplying the weights by a number smaller than 1. It is used as a regularizer to avoid overfitting, as weights with large values are more penalized than small weights. Several values for the weight decay were tested to see how it affected generalization.

**Momentum:** Momentum is used in SGD algorithms and it is an improvement to classical SGD, as it helps the network converge faster by accelerating the gradients toward the minimum and diminishing noisy effects. Here this parameter was used when SGD was selected as optimizer, but not when using Adam.

**Data augmentation:** This parameter helps to improve generalization, as it adds noise and other small perturbations to the images in the training set. Here, data augmentation was implemented by adding random horizontal flips and random rotations to the images.

**Batch size:** This is the number of images that are loaded in the GPU and fed into the network in each iteration. Since there were no problems with running out of GPU memory, the batch size was left fixed for all experiments.

In addition to the forementioned hyperparameters, some experiments were tested using pretrained networks and some training the network from scratch.

The following table shows the parameter values used in the experiments

|Learning Rate|  Weight Decay  | Momentum | Batch Size |  Dropout  | Data Augmentation |  Pretrained  |
|-------------|-------------|----------|------------|-----------|-------------------|--------------|
| 1e-3 / 5e-4 | 5e-3 / 5e-4 |      0.9 |         16 | 0.5 / 0.6 | True / False      | True / False |

## Results



|                            | Optimizer | Data Augmentation | Pretrained | Dropout | train / val accuracy | test accuracy |
|----------------------------|-----------|-------------------|------------|---------|----------------------|---------------|
| Siamese + decision network | SGD       | False             | True       | Default | 0.956 / 0.717        |               |
| Siamese + decision network | SGD       | True              | True       | Default | 1.0 / 0.711          |               |
| Siamese + decision network | Adam      | True              | True       | Default | 0.508 / 0.528        |         0.509 |
| Siamese + decision network | SGD       | False             | False      | Default | 0.841 / 0.621        |         0.659 |
| Siamese + decision network | SGD       | True              | True       | 0.6     | 0.768 / 0.729        |         0.839 |
| Cosine similarity          | SGD       | False             | True       | Default |                      |               |
| Cosine similarity          | Adam      | False             | False      | Default |                      |               |


## Conclusions
