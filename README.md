# **Traffic Sign Recognition** 

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

[image1]: ./Writeup-supported-pictures/dataset-visualization.jpg "Visualization"
[image2]: ./Writeup-supported-pictures/graysclae.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./Web-Traffic/1.jpg "Traffic Sign 1"
[image5]: ./Web-Traffic/2.jpg "Traffic Sign 2"
[image6]: ./Web-Traffic/3.jpg "Traffic Sign 3"
[image7]: ./Web-Traffic/4.jpg "Traffic Sign 4"
[image8]: ./Web-Traffic/5.jpg "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 samples
* The size of the validation set is 4410 samples
* The size of test set is 12630 samples
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

![alt text][image1]

I randomly selected an image for every unique class ID and use this image to display an example of every class as shown below. 

The bar graph shows the training dataset distribution of each unique class. Each bar represents one class ID and how many samples are in the training dataset for each class ID. This graph is very helpful when training a network since it shows which class is underrepresented and could potentially need more data.

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale since it is a lot faster to process a 1 channel image compare to a 3 channel image especially with the amount of images the model has to process to be trained. 

Then I normalized the image data because I want all my data to have zero mean and equal variance to get a well conditioned problem which help the optimizer get the solution and train a lot faster. normalizing the data also helps the weights and biases to initialize at a good values

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I used a Convolutional Neural Networks to classify different traffic signs. The network takes an image size of 32x32x1 as the input and output the probabilty of each 43 classes (the output is the probabilty of each of the 43 possible traffic signs.)

I used the standard LeNet-5 architecture with the addition of Dropout after the third and forth layer. Dropout reduces ouverfitting by randomly dropping activations from the network. This makes sure that the network never rely on any given activation to be present.

My final model consisted of the following layers:

| Layer         			|     Description	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Input         			| 32x32x3 RGB image   							| 
| Pre-Processing  			| 32x32x1 Grayscale image   					|
| Convolution 5x5     		| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU						| Activation Function							|
| Max pooling	      		| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU						| Activation Function							|
| Max pooling	      		| 2x2 stride,  outputs 5x5x16 				    |
| Flatten		      		| Input 5x5x16,  outputs 400				    |
| Fully connected			| Input 400,  outputs 120       				|
| RELU						| Activation Function							|
| Dropout					| 45% keep probability							|
| Fully connected			| Input 120,  outputs 84       			    	|
| RELU						| Activation Function							|
| Dropout					| 45% keep probability							|
| Fully connected (logits)	| Input 84,  outputs 43       			    	|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used a batch size of 128 and 30 epochs. I tried to use a smaller batch size but it didn't seem to improve the validation accuracy so I decided to keep it at 128. It also seems like the validation accuracy started to saturate at around 22 epochs but I decide to go with 30 epoches since it gave me a 1.5% of a better accuracy.

I also used the Adam optimizer because it uses the moving average of the parameters (momentum) which enables the optimizer to use a larger step size allowing the optimizer to coverage faster without fine tuning of the step size. This is a great advantage over the traditional  gradient decent. However, the main downside of the Adam optimizer that it requires more computational power. I decided to choose a learning rate of 0.001 without any fine tuning and let the Adam optimizer to converge

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.5%
* validation set accuracy of 96.5% 
* test set accuracy of 94.3%

I started with the LeNet-5 architecture which we learned in the LeNet Lab. I thought this would be a good architecture to start with as a baseline. I used a 2 convolution, 2 max pooling, and 3 fully connected. I modified the input to 32x32x3 and the output to 43 (number of classes). Training this architecture gave me a validation accuracy of 89%. I converted all training, validation, and test data to grayscale and normalized all the data. By pre-processing the images first, I was about to pump my validation accuracy from 89% to 91%

I then tried to tune the hyperparameters. Tuning the batch size and learning rate didn't seem to help my network get a better accuracy or I decide to leave them at 128 for batch size and 0.001 for learning rate. Increasing the number of the Epochs seemed to improve the network prediction accuracy. I started with 50 epochs but then realized  that the accuracy started to saturate at around 22 epochs so I decided to use 30 epoches. Tuning those hyperparameters allowed me to increase my validation accuracy to 92.5%

I then revisited the network architecture to see what modification I can do to improve my accuracy. I decided to add a dropout to my network since my network was overfitting which means the model won't generalize well to a new data points. I added two dropouts, after the third and fourth layer. In the beginning it seemed seem to improve my accuracy but later I was able to tune the keep probability to 0.45 which then improve my network accuracy to 96.5% 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I picked those images specifically because I was trying to pick some images which are very well represented in the training data and others which are underrepresented to experience  ow well my model will predict those signs

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No entry     			| No entry 										|
| Roundabout mandatory	| Roundabout mandatory							|
| Yield					| Yield     									|
| 100 km/h	      		| 100 km/h  					 				|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. Class ID 40 which is Roundabout mandatory sign wasn't classified correctly due to the fact that this class ID is underrepresented compared to the rest of the classes. One was to go around this to improve the model detection is by either collect more training data or argument the data to get more example of the underpresented classes.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 15th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.99), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop sign   									| 
| .000002  				| Right-of-way at the next intersection sign	|
| .0000006				| Turn right ahead sign							|
| .0000004     			| No vehicles sign				 				|
| .0000001			    | Turn left ahead sign 							|

For the second image, the model is relatively sure that this is a No entry sign (probability of 0.99), and the image does contain a No entry sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| No entry sign									| 
| .000007  				| Stop sign                         			|
| .000006				| Turn left ahead sign							|
| .0000002     			| Turn right ahead sign			 				|
| .00000003			    | Beware of ice/snow sign						|

For the third image, the model is relatively sure that this is a Priority road sign (probability of 0.99), and the image doesn't contain a Priority road sign, in fact it is a Roundabout mandatory sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Priority road sign							| 
| .001  				| Roundabout mandatory sign         			|
| .000007				| Right-of-way at the next intersection sign	|
| .0000007     			| Speed limit(30km/h) sign		 				|
| .0000001			    | Speed limit(100km/h) sign						|

For the Forth image, the model is relatively sure that this is a Yield sign (probability of 1.00), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Yield sign        							| 
| .00000000000000008	| No vehicles sign                  			|
| .0000000000000000001	| Turn left ahead sign                       	|
| .00000000000000000001	| No passing sign        		 				|
| .000000000000000000005| Priority road sign    						|

For the fifth and final image, the model is relatively sure that this is a Yield sign (probability of 1.00), and the image does contain a Yield sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Speed limit(100km/h) sign						| 
| .00008            	| Speed limit(80km/h) sign            			|
| .000000004        	| Speed limit(120km/h) sign                  	|
| .0000000001	        | Speed limit(50km/h) sign 		 				|
| .0000000000007        | Speed limit(30km/h) sign 						|

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


