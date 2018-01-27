# Indoor-Positioning-Via-Wifi-Fingerprinting
Predicting peoples' location (Building, Floor and Coordinates) from WAPs signal information using classification and regression machine learning algorithms (K-NN, Gradient boosting, and Random Forest)

# About the datasets
Source od the datasets: http://archive.ics.uci.edu/ml/datasets/UJIIndoorLoc

Two datasets, training and validation, contain information about 520 WAPs signal strengths, three buildings and their floors, coordinates of users who logged in, space where the users logged in and the relative position (inside or outside the room), user ID, phone ID, and timestamp. 
The data was collected at Jaume I University.
More information about the dataset can be found on the source link provided above. 

# Objective
My objective is to build models that predict the location (building, floor and coordinates) from the WAPs signal strengths of a user who connects to the internet at Jaume I University. 

# Procedure
I have cleaned the data, put the attributes into proper data types, subsetted dataset for each buiding, examined each column, and normalized WAPs rows in order to build models. I have used different algorithms (KNN, Random Forest, and Gradient Boosting) to predict the building, floor and coordinates of a logged in user. 

The first step was to remove columns and rows where WAP signal strength has only one value, meaning that the WAP was not detected. Also, the columns were converted into appropriate data types. 
Original dataset used WAP signal range from -104 to 0 (-104 being the lowest signal and 0 the highest), and value 100 when WAP was not detected. I have changed the values so that 0 represents the value when WAP was not detected and the highest signal is 104.  

The figure below shows the distribution of WAPs singal strength for each dataset. 
![distribution together](https://user-images.githubusercontent.com/32273216/35473323-f09f8c72-037e-11e8-82f6-2060a360c51b.jpg)

The following figure shows the distribution of WAPs signal strength on the same histogram in order to see the comparison clearly.

![03 - waps distribution training and test](https://user-images.githubusercontent.com/32273216/35473334-36cf6f5a-037f-11e8-922f-c559dd2756bc.png)

Number of WAPs detected per building for training and validation set can be seen in the figure below.

![number of waps both buildings](https://user-images.githubusercontent.com/32273216/35473388-232fe15e-0380-11e8-9242-60fddb9eaa27.jpg)

The coordinates in the dataset are given in World Geodetic System form as Longitude and Latitude. I have converted the to absolute values starting from 0. 

Log In locations of users from trainining and validation sets are shown in the figure below.

![06 - log in locations](https://user-images.githubusercontent.com/32273216/35473432-edf81dc0-0380-11e8-9bb4-f95b8305ffd2.png)

The following figure shows the log in locations of users at each floor in Building 1 from the training set. 

![07 - log in locations building 1](https://user-images.githubusercontent.com/32273216/35473449-3b879598-0381-11e8-9951-5ae6b8dde1c9.png)

I have checked the locations were signal were in good, medium and bad range. For example, the following figure shows the locations at which users had signal higher than 60 (out of 104).

![good signal longlat](https://user-images.githubusercontent.com/32273216/35473500-d58e332c-0381-11e8-82e7-f1b008340055.png)

It can be seen that least amount of high WAP signals was recorded in the middle building. Later in the modeling process this will cause lower accuracy of location predictions for this building. 

# Results (Machine Learning Models)

I started with building the model to predict the building at which user had connected to the internet. 
The first model had low accuracy so I decided to normalize WAPs rows put them in range from 0 to 1) and it drastically improved the model's performance. 

## Building prediction (Classification with KNN model)

With normalized WAP rows, the achieved accuracy is 100% for buildings' prediciton with KNN model. 
The confusion matrix and performance of the model can be seen below:
### Confusion Matrix:      0    1    2
###                   0  536   0    0
###                   1   0   307   0
###                 2   0    0   268
### Accuracy = 1
### Kappa = 1

## Floor prediction (Classification with KNN, Random Forest, and Gradient Boosting models)

For the floor prediction I have used the normalized WAPs data. 

### Building 1 Floor prediction
### Building 2 Floor prediction
### Building 3 Floor prediction


