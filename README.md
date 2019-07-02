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

#### Siamese + Decision network

![experiment_1_loss]
![experiment_1_acc]

   <sup>*Figure 1: Train and validation loss per epoch (no data aug)*</sup>         *Figure 2: Train and validation accuracy per epoch (no data aug)*

Figures 1 and 2 show the training and validation curves for loss (left) and accuracy (right) for a siamese + decision network, using SGD as optimizer, a learning rate of 5e-4 and weight decay of 5e-4, with no data augmentation (full hyperparameter description in table below). The model overfits quickly to the training data (blue line), shown by a really high training accuracy (close to 1) and an increase in validation loss

![experiment_2_loss]*Figure 3: Train and validation loss per epoch (data aug)*
![experiment_2_acc]*Figure 4: Train and validation accuracy per epoch (data aug)*

Figures 3 and 4 show the training and validation curves for a siamese + decision network using a SGD optimizer, with learning rate of 5e-4, weight decay of 5e-3 and data augmentation. Although not as quickly as in the case with no data augmentation, the model still overfits

![experiment_3_loss]*Figure 5: Train and validation loss per epoch (Adam optimizer)*
![experiment_3_acc]*Figure 6: Train and validation accuracy per epoch (Adam optimizer)*


|                            | Optimizer | Learning Rate | Weight Decay | Data Augmentation | Pretrained | Dropout | val accuracy | test accuracy |
|----------------------------|-----------|---------------|--------------|-------------------|------------|---------|--------------|---------------|
| Siamese + decision network | SGD       | 5e-4          | 5e-4         | False             | True       | Default |        0.717 |               |
| Siamese + decision network | SGD       | 5e-4          | 5e-3         | True              | True       | Default |        0.711 |               |
| Siamese + decision network | Adam      | 5e-4          | 5e-4         | True              | True       | Default |        0.528 |         0.509 |
| Siamese + decision network | SGD       | 1e-3          | 5e-4         | False             | False      | Default |        0.621 |         0.659 |
| Siamese + decision network | SGD       | 5e-4          | 5e-4         | True              | True       | 0.6     |        0.729 |         0.839 |
| Cosine similarity          | SGD       | 5e-4          | 5e-4         | False             | True       | Default |              |               |
| Cosine similarity          | Adam      | 1e-3          | 5e-4         | False             | False      | Default |              |               |




## Discussion and conclusion


[experiment_1_loss]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_SGD_loss_2.png "loss decision SGD no data augmentation"
[experiment_1_acc]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_SGD_accuracy_2.png "accuracy decision SGD no data augmentation"
[experiment_2_loss]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_SGD_loss_1.png "loss decision SGD with data augmentation"
[experiment_2_acc]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_SGD_accuracy_1.png "accuracy decision SGD with data augmentation"
[experiment_3_loss]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_Adam_loss.png "loss decision Adam with data augmentation"
[experiment_3_acc]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_Adam_accuracy.png "accuracy decision Adam with data augmentation"
[experiment_4_loss]
[experiment_4_acc]
[experiment_5_loss]
[experiment_5_acc]
[experiment_6_loss]
[experiment_6_acc]
