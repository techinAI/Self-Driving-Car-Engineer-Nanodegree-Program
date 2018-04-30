# Behavioral Cloning


The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/nvidia-architecture.png "Model Visualization"
[image2]: ./examples/center.png "Center"
[image3]: ./examples/recovery_left.png "Recovery Image Left"
[image4]: ./examples/recovery_right.png "Recovery Image Right"
[image5]: ./examples/uncropped.jpg "Uncropped"
[image6]: ./examples/cropped.jpg "Cropped"
[image7]: ./examples/unflipped.jpg "Unflipped"
[image8]: ./examples/flipped.jpg "Flipped"
[image9]: ./examples/left.jpg "Left"
[image10]: ./examples/middle.jpg "Middle"
[image11]: ./examples/right.jpg "Right"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.mp4 a video recording of your vehicle driving autonomously at least one lap around the track
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model uses the NVIDIA architecture of a deep neural network for autonomous driving. It consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64. 

The model includes RELU layers to introduce nonlinearity, the images are cropped and the data is normalized in the model using a Keras lambda layer. 

#### 2. Attempts to reduce overfitting in the model

The data includes multi-camera images and adjusted steering angles. In addition each images is flipped in order help the network generalize better. Also, the data includes several laps of counter-clockwise driving.

In addition, I added a max pooling layer and a dropout layer with a keep probability of 50% after the last 3x3 convolution layer.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, counter-clockwise driving and driving smoothly arround curves.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use the provided NVIDIA architecture, the suggested image augmentation and strategies for data collection.

I basically concentrated collecting as much data as possible, rather than tweaking the network.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with a normalization layer, three 5x5 convolution layers, two 3x3 convolution layers and 3 Full-Connected Layers + Output Layer.

Here is a visualization of the architecture (taken from the NVIDIA website):

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to navigate back to the center. These images show what a recovery looks like:

![alt text][image3]
![alt text][image4]

I also recorded two laps of counter-clockwise driving and one lap of smooth curve driving. In addition, I did some extra driving in problematic sections / curves of Track1.


To augment the data sat, I also flipped images and angles. For example, here is an image that has then been flipped:

![alt text][image7]
![alt text][image8]

In order to reduce distraction for the neural network and concentrate only on the road, I cropped 75px from the top and 25px from the bottom of each pixel:

![alt text][image5]
![alt text][image6]

Furthermore, I simulated the vehicle being in different positions, somewhat further off the center line by using all three provided camera images (left, middle and right):

![alt text][image9]
![alt text][image10]
![alt text][image11]

After the collection process, I had 11049 number of data points. I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.


