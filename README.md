# Face detection and recognition

Artificial intelligence with Deep Learning, Universitat Politècnica de Catalunya

Spring 2019

## Introduction

In this project we implemented a face detection and recognition neural networks The project was divided in two parts: a face detection part, in which given any picture with people (...) the network is able to detect the faces 

In the first part we tested and compared two facial detection algorithms using two different datasets, 
In the second part of the project, we splitted the group individually and each member of the group had to implement a siamese network with two VGG 

## Development

### Face detection

In the first part of the project we 

| Dataset | Num. of photos | Num. of faces |             Split              |
|---------|----------------|---------------|--------------------------------|
| FDDB    |           2840 |          4834 | 40% train / 10% val / 50% test |
| Wider   |          32203 |        383203 | 40% train / 10% val / 50% test |

The two algorithms 

### Face recognition

![alt][siamese_network]

## Architecture

The two networks implemented here share a common core architecture consisting of a siamese with two VGGs that share weights. Most of the experiments were executed with VGGs pretrained with imagenet, although in some experiments the networks were trained from scratch.

In the first network, the feature vectors outputted by the two VGGs are concatenated and then fed into a decision network consisting of a linear layer and a ReLU, which determines whether the two input images are from the same person or not. To train the network we use cross entropy loss. The accuracy is determined by the number of images guessed correctly out of the total number of images.

![alt][siamese_decision]

The code that implements this network can be found here: https://github.com/foca13/AIDL_2019_project/blob/master/Final_Project_siamesa_v2_sbd.ipynb

In the second network, the cosine similarity between the feature vectors outputted by the two VGGs is calculated. The network is trained by decreasing the angle between feature vectors of images that correspond to the same identity (thus increasing similarity) and increasing the angle between vectors of images that correspond to different identities (thus decreasing similarity). To determine the validation and test accuracy, we select the lowest similarity between images of the same identity (lower bound) and the highest similarity between images of different identities (upper bound). Starting from the lower bound, we select a threshod and we gradually increase the value until we reach the upper bound (steps of 0.001). From this range of values we then select the threshold that yields the highest accuracy.

![alt][siamese_cosine]

The code that implements this network can be found here: 
https://github.com/foca13/AIDL_2019_project/blob/master/Final_project_siamese_cosine_sim_network.ipynb

## Hyperparameters

In order to test the training efficiency of the networks and aim to find the best model, several hyperparameters were tunned and their effects on the model were studied.

**Learning rate:** The learning rate determines the step size towards the minimum of the loss function during training, which ultimately dictates how fast the network learns. Here, a larger learning rate was selected when running experiments where the VGGs were not pretrained, and smaller learning rate was used in experiments where pretrained VGGs were used.

**Weight decay:** This parameter prevents weights from exploding, that is from having values that are too large. It does so by multiplying the weights by a number smaller than 1. It is used as a regularizer to avoid overfitting, as weights with large values are more penalized than small weights. Several values for the weight decay were tested to see how it affected generalization.

**Momentum:** Momentum is used in SGD algorithms and it is an improvement to classical SGD, as it helps the network converge faster by accelerating the gradients toward the minimum and diminishing noisy effects. Here this parameter was used when SGD was selected as optimizer, but not when using Adam.

**Data augmentation:** This parameter helps to improve generalization, as it adds noise and other small perturbations to the images in the training set. Here, data augmentation was implemented by adding random horizontal flips and random rotations of up to +/- 20 degrees to the images.

**Batch size:** This is the number of images that are loaded in the GPU and fed into the network in each iteration. Since there were no problems with running out of GPU memory, the batch size was left fixed for all experiments.

In addition to the forementioned hyperparameters, some experiments were tested using pretrained networks and some training the network from scratch.

The following table shows the parameter values used in the experiments

|Learning Rate|  Weight Decay  | Momentum | Batch Size |  Dropout  | Data Augmentation |  Pretrained  |
|-------------|-------------|----------|------------|-----------|-------------------|--------------|
| 1e-3 / 5e-4 | 5e-3 / 5e-4 |      0.9 |         16 | 0.5 / 0.6 | True / False      | True / False |

## Results and discussion

### Siamese + Decision network

![alt][experiment_1_loss]
![alt][experiment_1_acc]

Figures 1 and 2 show the training and validation curves for loss (left) and accuracy (right) for a siamese + decision network, using SGD as optimizer, a learning rate of 5e-4 and weight decay of 5e-4, with no data augmentation (full hyperparameter description in table below). The model overfits quickly to the training data (blue line), shown by a really high training accuracy (close to 1) and an increase in validation loss. Despite the relatively high validation and test accuracies, the model is not training properly. The next experiment was ran with data augmentation and a higher weight decay to try to improve generalization.

![alt][experiment_2_loss]
![alt][experiment_2_acc]

Figures 3 and 4 show the training and validation curves for a siamese + decision network using a SGD optimizer, with learning rate of 5e-4, weight decay of 5e-3 and data augmentation. Although not as quickly as in the case with no data augmentation and lower weight decay, the model still overfits. This could be explained by the relatively small size of the dataset used and the similarity between many of the identities in the training and validation splits.

![alt][experiment_3_loss]
![alt][experiment_3_acc]

Figures 3 and 4 show the training and validation curves for a siamese + decision network using Adam optimizer, with learning rate of 5e-4, weight decay of 5e-4 and data augmentation. The seemingly random fluctuations in training and validation loss and accuracy suggest that the model didn't learn properly. This is supported by the low validation and test accuracy in this model. Since Adam is a well known and used optimizing algorithm, these problems were problably due to a bad implementation of the optimizer in our model.

![alt][experiment_4_loss]
![alt][experiment_4_acc]

Figures 5 and 6 show the training and validation curves for a siamse + decision network trained from scratch, with SGD optimizer, learning rate of 1e-3, weight decay of 5e-4 and without data augmentation. Since the model is not pretrained, it takes longer to learn, as shown by the much slower and progressive decrease of the loss function and increase of the accuracy function with respect to the previous models. Nonetheless, the model still shows overfitting behavior, and the increasing training accuracy suggest that the model would reach a training accuracy of 1 if it ran for more epochs.

![alt][experiment_5_loss]
![alt][experiment_5_acc]

Figures 7 and 8 show training and validation curves for a siamese + decision network with SGD optimizer, learning rate of 5e-4, weight decay of 5e-4 and an increased droupout of 0.6. This model was able to generalize better than the previous ones, as the overall accuracy in test and validation increased and the validation loss decreased. Another thing to notice is that the training accuracy didn't increase to 1

### Siamese cosine similarity

![alt][experiment_7_loss]
![alt][experiment_8_loss]

![alt][experiment_table]

## Conclusion


[experiment_1_loss]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_SGD_loss_2.png "loss decision SGD no data augmentation"
[experiment_1_acc]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_SGD_accuracy_2.png "accuracy decision SGD no data augmentation"
[experiment_2_loss]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_SGD_loss_1.png "loss decision SGD with data augmentation"
[experiment_2_acc]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_SGD_accuracy_1.png "accuracy decision SGD with data augmentation"
[experiment_3_loss]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_Adam_loss.png "loss decision Adam with data augmentation"
[experiment_3_acc]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_network_Adam_val.png "accuracy decision Adam with data augmentation"
[experiment_4_loss]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_pretrained_false_loss.png "loss decision not pretrained"
[experiment_4_acc]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_pretrained_false_val.png "accuracy decision not pretrained"
[experiment_5_loss]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_dropout_loss.png "loss decision with dropout"
[experiment_5_acc]: https://github.com/foca13/AIDL_2019_project/blob/master/results/Decision_dropout_accuracy.png "accuracy decision without dropout"
[experiment_6_loss]: 
[experiment_6_acc]:
[experiment_7_loss]: https://github.com/foca13/AIDL_2019_project/blob/master/results/loss_SGD_cosine.png "loss cosine similarity SGD"
[experiment_8_loss]: https://github.com/foca13/AIDL_2019_project/blob/master/results/loss_adam_cosine.png "loss cosine similarity Adam"
[experiment_table]: https://github.com/foca13/AIDL_2019_project/blob/master/resources/Table1.png
[siamese_network]: https://github.com/foca13/AIDL_2019_project/blob/master/resources/siamese_diagram.jpeg "siamese network"
[siamese_decision]: https://github.com/foca13/AIDL_2019_project/blob/master/resources/siamese_decision_diagram.png "siamese + decision network"
[siamese_cosine]: https://github.com/foca13/AIDL_2019_project/blob/master/resources/siamese_cosine_diagram.png "siamese cosine similarity"
