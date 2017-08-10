[//]: # (Image References)

[image1]: ./examples/visualization.png "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/tests/bycicles-cross.jpg "Bycicle Cross"
[image5]: ./examples/tests/general-caution.jpg "General Caution"
[image6]: ./examples/tests/speed-limit-50.jpg "Speed Limit 50"
[image7]: ./examples/tests/speed-limit-70.jpg "Speed Limit 70"
[image8]: ./examples/tests/yield.jpg	 "Yield"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used NumPy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is 32 * 32 * 3.
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data distribution for training and testing set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocess.

As a first step, I decided to convert the images to grayscale because when I tuning parameters, it takes me lots of time to train the module. Reducing 3 channels to 1 could save some time on training.

I decided to generate additional data because use LeNet with dropouts couldn't get good result.

So I randomly add rotate,  translate and shear to images. Except original training sets, I added 7 additional transformed images for each image in original training set.


#### 2. Model Architecture

My final model is as same as LeNet:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 grayscale image |
| Convolution 5x5   | 1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      | 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	  | 1 stride, valid padding, output 10x10x16 |
| RELU | |
| Max pooling	      | 2x2 stride,  outputs 5x5x16 				|
| Flatten | 400 outputs
| Fully connected		| input: 400, output 120 |
| RELU | |
| Fully connected	  | input: 120, output 84  |
|	RELU |												|
|	Fully connected   |	input: 84, output 43   |
| Softmax	| |



#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Adam Optimizer with following hyper parameters:
 * Learning rate: 0.005
 * Epochs: 50
 * Batch size: 512
 * dropout rate before softmax: 0.75

I heard from forum that Adam Optimizer can adjust learning rate adaptively.
#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.942.
* validation set accuracy of 0.934.
* test set accuracy of 0.9075.

I first tried to increase complexity like have more neurons on hidden layer. But after several experiment without improvement, I noticed that the network is actually complex enough to over fit. I was expect a accuracy decrease on validation set for overfitting, but no improvement for validation set accuracy is also an overfitting. So I changed the model back to LeNet5 and tried to add some dropouts.

But dropouts also decrease the training set accuracy. After browsing the forum for solution, I decided to augment training set.

After Augmentation, I got a pretty good result.


### Test a Model on New Images


Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8]

They all nice pictures but some of them contains transparent stamps and some of then contains banner for image provider. Those useless information might increase difficulty for correctly classify them.


Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Bycicle Cross   | Go straight or right |
| General Caution	| General caution |
| Speed Limit 50 | Speed Limit 50 |
| Speed Limit 70 | Speed Limit 20 |
| Yield | Road work |


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 40%.


The code for making predictions on my final model is located in the 37th cell of the Ipython notebook.

For all the images, the model is actually pretty confident on the result. Except 4th, all 4 others' softmax output are 1.0. And the 4th one has a possibility 9.999. I guess this is because the 0.75 keep rate on dropouts.
