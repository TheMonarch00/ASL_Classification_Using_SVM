# ASL_Classification_Using_SVM
Classifies the American Sign Language alphabets using support vector machines
Sign-Language Recognition using Support Vector Machines 
 

I. Introduction: 

Learning Sign-Language gestures is a challenging task which needs lots of practice and experience. As people are not willing to spare such time and effort in learning sign-language, people with hearing and speaking disabilities are not able to communicate freely with people outside their community. In this paper, we have used Support Vector Machines (SVM) to train a Machine Learning model which lets us understand the American Sign Language (ASL) alphabets. This model allows us to understand the Deaf and Mute making them comfortable even outside their community. 

Before training the model, the image data set was preprocessed. Then the data was split into train and test data in a ratio of 7: 3. Then the training data was fit into Support Vector Classification (SVC). Hyperparameter tuning was done before fitting the model for better accuracy. This model takes image as an input from the user and classifies it into ASL alphabet and gives the output. 

II. Related Work: 

Various types of techniques can be used to implement the recognition and classification of images using Machine Learning. Various research papers used different combinations of pre-processing and feature extraction techniques, like using MATLAB as the main programming language and using Scale Invariant Feature Transform (SIFT), Histogram of Gradients (HoG) in order to extract the features from the image and then using Support Vector Machines (SVM) in order to classify these into various alphabets [1]. Augmented Reality is also used in Sign-Language Recognition [2].  

Xbox Kinect camera is also used to retrieve the 3D image and translate it into English words [3]. In [4], the author took the hand kinematics from data glove to classify the gestures made by hand. Artificial Neural Networks is also an increasing Machine Learning field which can be used to extract various features from the image as done in [5]. Spatio temporal graph kernels are used to process 3D images and classify them as alphabets [6]. In [7], Recurring Neural Networks (RNN) with k-Nearest Neighbour (kNN) method was proposed for recognition 26 alphabets.  

[8] proposed hand sign language recognition using Single Shot Detector (SSD), 2D Convolutional Neural Network (2DCNN), 3D Convolutional Neural Network (3DCNN), and Long Short-Term Memory (LSTM) from RGB input videos. [9] This study proposes a new real-time sign recognition system based on a wearable sensory glove, which has 17 sensors with 65 channels. In [10], two stream Convolution Neural Network (2CNN) is used for sign-language recognition. 

Hyperparameter tuning helps in obtaining good accuracy. In [11], the author provided the significance of hyperparameter tuning. In [12], the authors reviewed different methods of Sign-language recognition and gave a detailed report. In [13], Depth images are used for sign-language recognition. [14] This paper discusses the problems faced by hearing impaired individuals in India and the use of technology to make their interaction easier. In [15], selfie camera is used as input device and the gestures are recognized. 

---------------------------------------------------------------------------------------------------- 

III. Proposed Methodology 

The proposed model in this paper consists of three basic processing stages. 

Image Preprocessing 

Model Training 

Predicting Output 

 

DATA COLLECTION 

	Our dataset comprises of 26 folders for 26 English alphabets. The data was taken from Kaggle. Instead of using whole dataset, we have used only 40 images per alphabet due to limited system specifications. 

IMAGE PREPROCESSING 

	Before splitting the data, images had been resized to 300x300x3 so that all images were of same size and execution takes optimal time. Target values which were characters were label encoded. The image processing was handled using skimage. Scikit-image (skimage) is a collection of algorithms for image processing. 

	After resizing the images data, it was flattened to a one-dimension as 3D can’t be used as inputs. Once the images were flattened, they should be converted into numpy arrays in order to perform operations on the data. Numpy is the fundamental package for scientific computing which enables scientific and mathematical operations. 

	Once the preprocessing is done, the data can be shown in a data frame for better readability. 
SPLTTING THE DATA 

	Once the data preprocessing was done, it was split into training data and testing data using train_test_split() function available in sklearn.model_selection library. Scikit-learn is a free software machine learning library for the Python programming language. The preprocessed flat data was passed through the function along with specified ratio (7:3) and was stratified with respect to target values. Stratifying helps to split all the alphabets evenly. 

HYPERPARAMETER TUNING 

	A hyperparameter is a parameter whose value is used to control the learning process. The values of hyperparameters greatly affect the learning process of the machine. With a good hyperparameter tuning, we can achieve better model with higher accuracy. 

	In our model, we used GridSearchCV and best_params_  from sklearn for hyperparameter tuning. GridSearchCV takes lists of parameters and searches for the best combination of hyperparameters. With a big dataset, this process takes a lot of time as it fits the data into different hyperparameter combinations and returns the best combination. 

TRAINING THE MODEL 

	Once we were done with hyperparameter tuning, we fit the training data into the model. We used Support Vector Classification (SVC) class which is one of Support Vector Machines classes as our main objective was to classify the ASL alphabets. 

	Support vector machines (SVMs) are a set of learning methods used for classification, regression and outlier's detection. The SVC is well capable of performing multi-class classification. Our model should be able to perform classification on 26 classes. After hyperparameter tuning, we found that ‘rbf’ was the best kernel for our model. 

	Radial Basis Function is one of the popular kernels and more accurate than linear kernel in most of the cases. Training/ Building the model takes some time for execution but it takes less time compared to GridSearchCV. 

TESTING THE MODEL 

	Once the model was built, we had to test it with the testing data to know how accurate our model was. For this, we used predict() function. Once we got the predicted values, we had to compare them with the original target values of testing data. For evaluating the model, we used metrics library from sklearn.  

	The sklearn.metrics module implements several losses, score, and utility functions to measure classification performance. In sklearn.metrics, we had used two methods for evaluation. 

confusion_matrix(): This method returns a matrix which shows the information of true positives, true negatives, false positives and false negatives. 

accuracy_score(): This method returns the accuracy of our model ranging from 0 to 1. Our model got an accuracy score of 0.99679 which implies that our model is 99.679% accurate. 

 

SAVING THE MODEL 

	We used pickle library to save our model. We saved our model as a pickle file. Pickling is the process of serializing the data into a file and saving it. Saving the model prevents training a model every time we want to use it.  Once the model is pickled and saved, we have to unpickle/load it in order to use it to predict a new image. 

Classifying NEW IMAGE 

	Once we have our model saved and loaded, we can use it to classify new images. The new image should also be preprocessed before giving it as input to the model. Once the input image is preprocessed, we can use predict() function in the model to predict what the alphabet is.  
