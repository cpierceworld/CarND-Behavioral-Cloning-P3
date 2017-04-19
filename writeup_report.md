# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[cnn_diagram]: ./cnn_diagram.png "Model Visualization"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
(*NOTE*: Was developed using Keras version 2.0.3)

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is essentially the NVIDIA architecture mentioned [here](https://arxiv.org/abs/1604.07316).

It is a convolution neural network with 4 convolution layers consisting of 2 5x5 kernels and 2 3x3 kernels with depths between 24 and 64 (model.py lines 53, 55, 57, 59) and 3 fully connect layers (lines 64, 66, 68)

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 52). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 54, 56, 58, 63, 65, 67).  The percentage of dropped values depends on the size of the layer (with large layers dropping 40%, and the smallest layer dropping 5%) 

Also equal amounts of training data from both the "simple track" and "challenge track" were used in the training set (which look very different so it couldn't "memorize the track")

The model was trained and validated on different data sets to ensure that the model was not overfitting (The "validation_split" of the Keras "fit" method, line 77). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 72).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and multiple passes at know parts of the track with poor performance.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
I started with  a convolution neural network model similar to the NVIDIA architecture since that was shown to work well for this exact problem.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that during training I rapidly obtained a low mean squared error on the training set but the mean squared error on the validation would hit a low and then start rising. This implied that the model was overfitting. 

To combat the overfitting, I first tried decreasing the depth of the convolution layers so that there were not enough degrees of freedom to over fit, but this didn't help as it just made the training data stop learning early on.   I then tried adding more data from the "advanced track", but this just made it worse (but I think this was just do to an imbalance between the amount of "simple track" training data vs. the amount of "advanced track" training data).  

Next I added Dropout between all the layers.  I attempted to add a lot of dropout to layers with a lot of nodes (the convolution layers) and a small amount dropout between layers with small amount of nodes (the latter fully connected layers)   This helped a lot, not only with keeping the validation loss similar to the training loss, but allowing the training to continue to improve for many epochs.

I then added more training data from the "advanced track" to the point where there was as much or more samples from the "advanced track" as the simple track.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, specifically where the contrast between the background and foreground was low (approaching a lake or the off-road dirt area).  To fix this, I gathered more samples at the specific locations that the care went off the track, including "recovery" scenarios from the edges in those locations.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 50-69) consisted of a convolution neural network with the following layers and layer sizes:

| Layer                 | Description                                   | 
|:---------------------:|:---------------------------------------------:| 
| Input                 | 160x320x3 RGB image                           | 
| Image Cropping        | 110x320x3 Cropped RGB image                   |
| Image Normalization   | "Lambda" layer (normalize -0.5 to 0.5)         |
| Convolution 5x5       | 2x2 stride, ReLU, outputs 55x160x24           |
| Dropout               | 40% dropped out                               |
| Convolution 5x5       | 2x2 stride, ReLU, outputs 28x80x36            |
| Dropout               | 40% dropped out                               |
| Convolution 5x5       | 2x2 stride, ReLU, outputs 14x40x48            |
| Dropout               | 40% dropped out                               |
| Convolution 3x3       | 1x1 stride, ReLU, outputs 14x40x64            |
| Convolution 3x3       | 2x1 stride, ReLU, outputs 7x40x64             |
| Dropout               | 40% dropped out                               |
| Fully connected       | input = 17920, ReLU, output = 100             |
| Dropout               | 20% dropped out                               |
| Fully connected       | input = 100, ReLU, output = 25                |
| Dropout               | 20% dropped out                               |
| Fully connected       | input = 25, ReLU, output = 10                 |
| Dropout               | 10% dropped out                               |
| Fully connected       | input = 10, output = 1                        |

Here is a visualization of the architecture:

![alt text][cnn_diagram]

#### 3. Creation of the Training Set & Training Process

I used the training data provided by Udacty for the "simple track" to start with, I supplemented that with 2 laps on the simple track in both directions. I then gathered samples from the "challenge track" by completing 2 laps in each direction.  I then went back and gathered samples only on the parts of the track(s) where the trained model failed (i.e. where the car went off track).

To augment the data sat, I also flipped images and angles which would double the amount of training data and balance out any "left turn" bias from training on a counterclockwise track.

I also made an attempt to use the Left/Right camera images while adding/subtracting to the steering angle.  After playing around with this for a while, I gave up as the Left/Right camera only seemed to make things worse.

During training, I decreased the brightness of every 10th sample image in order to get more "in the shadows" samples to help with the "challenge track".  This helped somewhat, though the model still fails to finish the "challenge track".

In the end, I had over 30,000 samples to train with. I randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was around 20 since at that point both the training and validation loss seemed to get any better.  I used an adam optimizer so that manually training the learning rate wasn't necessary.
