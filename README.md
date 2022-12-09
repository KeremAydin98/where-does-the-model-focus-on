# where-does-the-model-focus-on

## Introduction

With the development of convolutional neural networks AI improved itself even better than humans in image classification. Convolutional layers learns how to differ the images by extracting features from low-level to high-level. But where does the model focus on before predicting the class of an image and is it the same as humans? That's why I built a model that generates the class activation maps of a given image so that we can see where the model focuses.

## Dataset 

The dataset contains images of 10 different fruits from kaggle:

https://www.kaggle.com/datasets/karimabdulnabi/fruit-classification10-class

## Generating class activation maps

To generate class activation maps, we need a working image classification model first. The model I have built can be seen below:

```
self.model = nn.Sequential(convBlock(3,64),
                                   convBlock(64,64),
                                   convBlock(64,64),
                                   convBlock(64,64),
                                   convBlock(64, 64),
                                   nn.Flatten(),
                                   nn.Linear(1024,256),
                                   nn.Dropout(0.2),
                                   nn.ReLU(inplace=True),
                                   nn.Linear(256, id2int))
```

Convblocks contain a convolutional layer, relu function, batch normalization layer, max pooling and dropout layer. After training the image classification model, we need to extract the activations of last convolutional layer's output, so that we can see the high level features from the model.

```
img2fmap = nn.Sequential(*(list(model.model[:4]) + list(model.model[4][:2])))
```

Only the channels that are responsible for predicting the correct class will have a high gradient, therefore after extracting these feature maps which are corresponded to that class and upsampling them to our image size, we can examine the place the model focuses on to make a prediction.

The whole process:

![Screenshot 2022-12-09 134735](https://user-images.githubusercontent.com/77073029/206685458-77dee036-dd39-46bf-80ce-0e26a5ff74c7.png)


This can also be used when our model performs badly, since it may focus on the wrong objects in a picture. If that's the case, we can either use object detection or cropping to get a better result.


## Some of the results

We can see where the model focuses when predicting certain fruits. For instance, we can see that the model focuses on the stalk while predicting a cherry.

<p align="center">
  <img src="https://user-images.githubusercontent.com/77073029/206567145-604ff126-8048-4796-9373-a203594d35ce.png" />
</p>


