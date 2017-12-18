# TorchMNIST

## Introduction:

The exercise entailed classifying the MNIST dataset using the demo code provided for Torch. It was my first time using Torch and I found it easier to grasp than my initial experience with Theano. I tried to compare the performance, specfically accuracy versus training epochs, for the three algorithms in the demo code namely Convolutional Nerual Networks (convnets), a 2-layer fully connected Perceptron (mlp) and Logisitic Regression. Then I made architectural changes to the convnet and plotted the performance difference.

## The Data Set:

MNIST is a dataset of handwritten digits. The version provided to us for the exercise consisted of 32X32 images for each digit. The training set consists of 60,000 images and the test set is 10,000 images. In the code there is the option to use the reduced data set i.e., 2000 training images and 1000 test images or the full data set. I started with the reduced data set to make an initial comparison as it took much less time and eventually moved to the full data set when making changes to the convnet.

## List of Experiments on the smaller set:

1.	My intial run was to just let the default convet in the code run so that I had some idea of the performance benchmark and a reasonable value for the number of epochs to run the code for. The optimzation algorithm was Stochastic Gradient Descent and the learning rate was 0.05. I did not modify the the learning algorithm or learning rate for any of the experiments. The optimzation criterion for all experiments is negative log likelihood. On the output layer for all the experiments a LogSoftMax function is applied. The network architecture is as follow:
	 input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> output
  	(1): nn.SpatialConvolutionMM(1 -> 32, 5x5)
  	(2): nn.Tanh
  	(3): nn.SpatialMaxPooling(3x3, 3,3)
  	(4): nn.SpatialConvolutionMM(32 -> 64, 5x5)
  	(5): nn.Tanh
  	(6): nn.SpatialMaxPooling(2x2, 2,2)
  	(7): nn.Reshape(256)
  	(8): nn.Linear(256 -> 200)
  	(9): nn.Tanh
	(10): nn.Linear(200 -> 10)
	The batch size was 10. The performance plot on training and test set is as follows:
  
	All the graphs plot the mean class accuracu against the training epochs. The test accuracy is around 95%. I decided to limit subsequent experiments to 30 epochs. 

2.	Next I ran an MLP using the same training parameters. The accuracy plots are as follow:

		












	











	The test accuracy is around 88 percent. 

3.	My last experiment on the smaller data set was the logistic regression with the same training parameters. The plots are as follows:
	
	











	











	The test accuracy is around 85.75 percent.

## List of Experiments on the Full Data Set:

1.	There were two things I modified with the default convnet for this experiment. I increased the batch size to 100 and added dropout at the input of the two fully connected layers.  Smaller batches tend to reduce overfitting and increase performance but each training iteration takes more time so the conventional wisdom is to select the smallest batch size that your machine can handle effeciently in terms of time. Dropout is a simply method to regularize the network to improve performance and reduce overfitting. I did not train the dropout probability and instead used the value of 0.5
	The network architecture is as follows:

	input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output
  	(1): nn.SpatialConvolutionMM(1 -> 32, 5x5)
  	(2): nn.Tanh
  	(3): nn.SpatialMaxPooling(3x3, 3,3)
  	(4): nn.Dropout(0.500000)
  	(5): nn.SpatialConvolutionMM(32 -> 64, 5x5)
  	(6): nn.Tanh
  	(7): nn.SpatialMaxPooling(2x2, 2,2)
  	(8): nn.Dropout(0.500000)
  	(9): nn.Reshape(256)
  	(10): nn.Linear(256 -> 200)
  	(11): nn.Tanh
  	(12): nn.Linear(200 -> 10)
	
	The plots for the training and test accuracy are as follow:

	   












	












	The test accuracy is around 97.96 percent. The literature suggests that when dropout is we should train for longer as their is more performance gain and the curve does not flatten as quickly. However to mantain uniformity I kept to the 30 epochs. The batch size of 100 took too long on my machine. So for the subsequent experiments I used a batch size of 1000. 

2.	In my second experiment I kept the architecture as is and only modified the batch size to 1000. This experiment also served as a benchmark for the later experiments where I kept the same batch size.
	The plots for the training and test accuracy are as follows:
	











	












	The training accuracy was around 96.2 percent and with the larger batch the graph is smoother but it takes more epochs to increase the accuracy. 

3.	In this experiment I replached the hyperbolic tangent  (tanh) units with the Rectified Linear Units (ReLU). ReLU have been shown to decrease training time in the literature and they work well in conjuction with dropout.

	The architecture for the network is as follows:

	input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output
  	(1): nn.SpatialConvolutionMM(1 -> 32, 5x5)
  	(2): nn.ReLU
  	(3): nn.SpatialMaxPooling(3x3, 3,3)
  	(4): nn.SpatialConvolutionMM(32 -> 64, 5x5)
  	(5): nn.ReLU
  	(6): nn.SpatialMaxPooling(2x2, 2,2)
  	(7): nn.Reshape(256)
  	(8): nn.Dropout(0.500000)
  	(9): nn.Linear(256 -> 200)
  	(10): nn.ReLU
  	(11): nn.Dropout(0.500000)
  	(12): nn.Linear(200 -> 10) 
	
	The plots for training and test accuracy are as follows:
	  







	










	
	 









	The test accuracy is around 97.96 percent. One observation from the plot is that roughly it takes  half the number of epochs to reach the 95 percent mark compared to the archtiecture using tanh units. 

4.	Next I modified the maxpooling layer. The default was a 3*3 in the first convolution layer and a 2*2 in the second convolution layers. Pooling is way of downsampling, thereby reducing dimensionality and reducing computation for the subsequent layers. If we pool over a larger area in the lower layers so reduce too much information. On the other hand not pooling at all may lead to too many computations and even overfitting.  For this experiment I used 3*3 pooling for both layers. This also reduced the number of units in the fully connected layers.
	The network architecture is as follows:

	input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output
	(1): nn.SpatialConvolutionMM(1 -> 32, 5x5)
	(2): nn.ReLU
  	(3): nn.SpatialMaxPooling(3x3, 3,3)
  	(4): nn.SpatialConvolutionMM(32 -> 64, 5x5)
  	(5): nn.ReLU
  	(6): nn.SpatialMaxPooling(3x3, 3,3)
  	(7): nn.Reshape(64)
  	(8): nn.Dropout(0.500000)
  	(9): nn.Linear(64 -> 200)
  	(10): nn.ReLU
  	(11): nn.Dropout(0.500000)
  	(12): nn.Linear(200 -> 10)

	The plots for the experiment are as follows:

	








	











	We can observe that the accuracy dropped to about 95%. One advantage though of the lesser number of units in the fully connected layer was that individual training iterations were a bit quicker.

5.	In the last experiment for convnets I used 2x2 maxpooling in both the convolutions. This resulted in the network having more units in the fully connected layer and the training iterations took longer. The network is as follows:
	input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> (11) -> (12) -> output]
	(1): nn.SpatialConvolutionMM(1 -> 32, 5x5)
  	(2): nn.ReLU
  	(3): nn.SpatialMaxPooling(2x2, 2,2)
  	(4): nn.SpatialConvolutionMM(32 -> 64, 5x5)
  	(5): nn.ReLU
  	(6): nn.SpatialMaxPooling(2x2, 2,2)
  	(7): nn.Reshape(1600)
  	(8): nn.Dropout(0.500000)
  	(9): nn.Linear(1600 -> 200)
  	(10): nn.ReLU
  	(11): nn.Dropout(0.500000)
	(12): nn.Linear(200 -> 10)
	
	The training and test accuracy graphs are as follows:
	 










	










	The accuracy on the test set is 97.8 percent



6.	In this experiment I tried the mlp on the full data set with the default parameters. The network is as follows:
	input -> (1) -> (2) -> (3) -> (4) -> output]
  	(1): nn.Reshape(1024)
  	(2): nn.Linear(1024 -> 2048)
  	(3): nn.Tanh
  	(4): nn.Linear(2048 -> 10)
	
	As in the convnet there is a logsoftmax non-linearity at the end. The accuracy plots are as showm below:

	

















	











	The test accuracy is 95.6 percent. However we observe that the performance is improving at the 30 epoch mark. I did another experiment to observe the performance for 40 epochs. The test accuracy curve is follows:
	











	
	The accuracy is 96.2 percent and we observe that the performance curve tends to flattens after the 35 epoch mark.
7. 	In the last experiment I ran the logistic regression on the full data set. The network is as follows:
	input -> (1) -> (2) -> output
  	(1): nn.Reshape(1024)
  	(2): nn.Linear(1024 -> 10)
	
	There is a logSoftmax non linearity at the output. The accuracy plots are as follows:

	







	











	The accuracy achieved in 92.17 percent

## Discussion:

We observe that on the full data set if we constrain ourselves to 30 epochs with a batch size of 1000, the best performance is achieved for the convnet with ReLU units, dropout and 2x2 maxpooling after both convolution layers. There are other parameters and configurations that I could have modified. One of the most important is the learning rate and idea of adding momentum to it. 

There is an increasing trend to optimize all the network parameters including the number of units in each layer using sophisticated algorithms such as Bayesian Optimization techniques.  
