# **Traffic Sign Recognition** 

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/label_distribution.png "hist"
[image2]: ./img/before.png "before"
[image3]: ./img/after.png "after"
[image4]: ./new_data/new0.png "Traffic Sign 1"
[image5]: ./new_data/new1.png "Traffic Sign 2"
[image6]: ./new_data/new2.png "Traffic Sign 3"
[image7]: ./new_data/new3.png "Traffic Sign 4"
[image8]: ./new_data/new4.png "Traffic Sign 5"
[image9]: ./img/feature_map.png "feature"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/yang1fan2/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799.
* The size of the validation set is 4,410.
* The size of test set is 12,630.
* The shape of a traffic sign image is 32 * 32.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the data distribution for different labels. We can see that for a few labels, the training data is so small.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I didn't convert the RGB image to gray scale. I rescale the pixel values for all RGB channels using suggested method: (pixel - 128)/128. The following images show the traffic sign before and after preprocessing. 

![alt text][image2]

![alt text][image3]

I also tried to augment the training data by randomly flipping the images both vertically and horizontally using tensorflow API:
`tf.image.random_flip_left_right` and `tf.image.random_flip_up_down`.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x32 	|
| RELU					|												|
| Max pooling	 2x2    	| 1x1 stride, same padding outputs 32x32x32 				|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 32x32x64						|
| RELU |
| Max pooling	 2x2    	| 1x1 stride, same padding outputs 32x32x64 				|
| Flatten |
| Fully connected		| outputs 1024    									|
| Dropout | probability = 0.5 |
| Fully connected | outputs 43|
| Softmax				| |
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained my model with Adam optimizer with learning rate 1e-4 for 20 epochs. And the batch size is 100. Moreover, I initialize the weight parameters using Xavier initializer.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.6%
* validation set accuracy of 93.3%
* test set accuracy of 91.9%
It is calculated in my python notebook (i.e. Traffic_Sign_Classifier.ipynb).

In the beginning, I used Lenet5 but both training and validation accuracy is low. So the Lenet5 is underfitting the data (with validation accuracy ~87%). Then I took a look at Mnist dataset and found several similiarities:
- The traininig data sizes are at the same level (60k vs 35k)
- Both tasks are image classifications problem (classify 10 labels vs 43 labels).

So I borrowed a well-known design from tensorflow-model [repository](https://github.com/tensorflow/models/tree/master/official/mnist).
There are a few major changes:
- Increase the model size (e.g. kernel size, filter size, fully connected layer's dimensions)
- Add dropout
- Change the padding method to `SAME`

In the end I also explored different initializier method (e.g. xavier and glorot uniform) and the dropout rate.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

All the images have good qualities, so it should be easy to classify them.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Right-of-way at the next intersection      		| Right-of-way at the next intersection   									| 
| Speed limit (70km/h)     			| Speed limit (70km/h) 										|
| Road work					| Road work											|
| Speed limit (30km/h)      		| Speed limit (30km/h)					 				|
| General caution			| General caution      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the bottom of the Ipython notebook.

The top five soft max probabilities of second image were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .928         			| Speed limit (70km/h)   									| 
| .069     				| Speed limit (50km/h) 										|
| .003					| Speed limit (20km/h)											|
| ...	      			| ...					 				|

The top five soft max probabilities of last image were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .979         			| General caution  									| 
| .02     				| Traffic signals 										|
| ...	      			| ...					 				|

I didn't mention the rest is because the max probability is almost 100%.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
Please find the following images. The different filters allow the model to detect different parts of the traffic sign. Some filters detect the triangle, while some detect the sign in the middle.
![alt text][image9]
