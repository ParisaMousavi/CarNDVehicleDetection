# Vehicle Detection
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)


In this project, your goal is to write a software pipeline to detect vehicles in a video (start with the test_video.mp4 and later implement on full project_video.mp4), but the main output or product we want you to create is a detailed writeup of the project.  Check out the [writeup template](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup.  

# Histogram of Oriented Gradients (HOG)


## Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

> Explanation given for methods used to extract HOG features, including which color space was chosen, which HOG parameters (orientations, pixels_per_cell, cells_per_block), and why.

**Resubmission 1**

In this resubmission I have transferred the following functions to an .py file to have shorter and readable code in jupyter notebook.

- bin_spatialI
- convert_color
- color_hist
- get_hog_features

The name of python file is `features.py` and it is imported in Jupyter notebook.

**Histogram of Oriented Gradient**
I have a function for extracting the Histogram of Oriented Gradient with the following header, which has been called in “Extracting the Histogram of Oriented Gradient (scikit-image HOG)” cell.
`def get_hog_features(myimg, orient, pix_per_cell, cell_per_block,vis=False, feature_vec=True):`

The next figure is the visualization of the output of the `get_hog_features` function.
![Vidualization of the get_hog_features function](https://github.com/ParisaMousavi/CarNDVehicleDetection/blob/master/pics/figure1.png)

The main function which is used to extract the image HOG is hog function as in training codes. The `get_hog_feature` function has been used in two different part of this project.

- Under “Extract Dataset's features” cell: via calling `extract_features` function from training materials. 

```python
car_features = extract_features(cars, cspace = color_space,
spatial_size = spatial_size,
hist_bins = hist_bins,
hist_range = hist_range)

notcar_features = extract_features(notcars, cspace = color_space,
spatial_size = spatial_size,
hist_bins = hist_bins,
hist_range = hist_range)
```

- Under “Function for processing each frame” cell: via calling  `find_cars` function.
```python
predicated_windows = find_cars(image, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
```

As we can see in the two previous code snippets we have variables which determine which feature must be considered for training and for vehicle finding as well. These variables have been defined under “Global values” cell.

Variable name | Value | Description
------------ | ----------------- | -------------
spatial_feat | True or False | True means spatial feature is considered in feature of image for training and vehicle detection. False means this feature must not be considered in both parts.
hist_feat | True or False | True means Histogram of Color is considered in feature of image for training and vehicle detection. False means this feature must not be considered in both parts.
hog_feat | True or False | True means Histogram of Oriented Gradient is considered in feature of image for training and vehicle detection. False means this feature must not be considered in both parts.

### About orient, pix_per_cell, cell_per_block variables
I have tested the “Extracting the Histogram of Oriented Gradient (scikit-image HOG)” code snippet not only for 8 pixels per cell but also for 4 and 2 pixels per cell and all of them with 2 cells per block.

![orient=9, pix_per_cell=2, cell_per_block=2](https://github.com/ParisaMousavi/CarNDVehicleDetection/blob/master/pics/figure2.png)
*orient=9, pix_per_cell=2, cell_per_block=2*

For example, this figure at the left side is a sample of pixel per cell 2 and cell per block 2.
The pattern of vehicle is clearer than the previous one, but the performance is worse than pix_per_cell=8, cell_per_block=2.
Therefore, I have developed my project with orient = 9, pix_per_cell=8 and cell_per_block=2 which are defined and assigned under “Global values” section in Jupyter Notebook.
### About color space variable
I have also tested all the color spaces which have been defined in project as shown in following figure.

![enter image description here](https://github.com/ParisaMousavi/CarNDVehicleDetection/blob/master/pics/figure3.png)|![enter image description here](https://github.com/ParisaMousavi/CarNDVehicleDetection/blob/master/pics/figure4.png)
------------|------------
Figure 3: RGB color space | Figure 4: HSV


![enter image description here](https://github.com/ParisaMousavi/CarNDVehicleDetection/blob/master/pics/figure5.png)|![enter image description here](https://github.com/ParisaMousavi/CarNDVehicleDetection/blob/master/pics/figure6.png)
------------|------------
Figure 5: LUV | Figure 6: HLS


![enter image description here](https://github.com/ParisaMousavi/CarNDVehicleDetection/blob/master/pics/figure7.png)|![enter image description here](https://github.com/ParisaMousavi/CarNDVehicleDetection/blob/master/pics/figure8.png)
------------|------------
Figure 7: YUV | Figure 8: YCrCb

But I have decided to use `YCrCb` color space.

<![endif]-->

## Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

> The HOG features extracted from the training data have been used to train a classifier, could be SVM, Decision Tree or other. Features should be scaled to zero mean and unit variance before training the classifier.

In the following figure we have the flowchart of my training and test Linear SVC Classifier mechanism.

FLOCHART PLACE HOLDER

The training code is started from cell “Global values”, where I have defined the variables and values for training the model as in following.

```python
color_space = 'YCrCb'
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = "ALL" 
spatial_size = (16, 16) 
hist_bins = 128   
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
hist_range = (0, 256)
```

In “Load Dataset” cell I have defined two loops to make two lists of vehicle and non-vehicle image names for training and testing the linear SVC model as shown in the following code snippet.

```python
images = glob.iglob('./all/vehicles/**/*.png',recursive=True)
cars = []
notcars = []
for image in images:
    cars.append(image)
images = glob.glob('./all/non-vehicles/**/*.png',recursive=True)
for image in images:
    notcars.append(image)
```
After preparing the vehicle and non-vehicle lists, I extract the feature/s of the car and non-car images with extract_features function in cell “Extract Dataset's features” and create an array stack of the extracted features to pass to `StandardScaler().fit()` function as developed in “Train Linear SVC(support vector classifier) Classifier” cell.
`StandardScaler().fit()` fits the scale per-column and I apply the calculated scaler to the created array stack as shown below.

```python
    # Scale the feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)      
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
   ```

After transforming the data, they must be split into two Train and Test dataset and I have developed a random logic to calculate a random percentage for splitting the dataset as shown in the following code snippet.

```python
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split( scaled_X, y, test_size=0.2, random_state=rand_state)
```
As next step the model is trained and as I have explained in the yellow box. The optimization in this step is the `C` parameter of the `LinearSVC` function.

**Resubmission 1**

Another point that I have considered in this resubmission is the fine tuning of the C parameter of the `LinearSVC` classifiert as following:
`svc = LinearSVC(C=0.0005)`.

```python
svc = LinearSVC(C=0.0005)
svc.fit(X_train, y_train)
```

Although it wasn’t requested in the project to save the model but I have developed a cell in Jupyter notebook to save the mode as I have explained in the bellow yellow box.

**Resubmission 1**

Another new logic that I have developed in code is saving the variables and model because it was not time efficient for me to run the whole code each time that I wanted to develop the steps after training the mode like developing find_cars function. This logic is developed in “Save model” cell.

```python
obj = {
"svc": svc,
"scaler": X_scaler,
"orient" : orient,
"pix_per_cell": pix_per_cell,
"cell_per_block": cell_per_block,
"spatial_size": spatial_size,
"hist_bins": hist_bins,
"color_space": color_space,
"hog_channel": hog_channel,
"spatial_feat": spatial_feat,
"hist_feat": hist_feat,
"hog_feat": hog_feat,
"hist_range":hist_range
}
pickle.dump(obj, open('svc_pickle.p', 'wb'))
```

And in “Read model” cell there is a logic for reading/reloading the saved variables and model.

# Sliding Window Search

## Describe how (and identify where in your code) you implemented a sliding window search. How did you decide what scales to search and how much to overlap windows?

> A sliding window approach has been implemented, where overlapping tiles in each test image are classified as vehicle or non-vehicle. Some justification has been given for the particular implementation chosen.

The sliding, scaling and overlapping are done in `find_cars` function in “Using LinearSVC Classifier to detect vehicles” cell. I’ll explain about each one in the following.

**Overlapping**

As it has been mentioned in training materials instead of the overlapping we can use the following code line.

```python
cells_per_step =  2  _#Instead of overlap, define how many cells to step
```

I have trained the code with different values. In case of better value, I didn’t have efficient running time. Therefore, I decided to have 2 cells per step overlapping because the result was satisfactory, and the result wasn’t long.


<iframe width="560" height="315" src="https://www.youtube.com/embed/-AbUpO2lEOM" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExMjIyOTk1NTUsMTQ3NDI1MDE4MywtNj
Q2MjYzMzQzLDQ5NzU2Njk4Nyw4MDU0MjUzODgsOTY3MTUwMTgx
LDYzOTM3ODIzLC0zNzc3MDI2OCwyMTM4MDEwNTk5LDYzNDQyOD
MyMywtMTY5MjYyNzU3MSwtMTE4MjIyMTc1OSwtMTIxNzEwNjA2
MCwtODI3MDIxNDAwLDEyMTg5Njk5MTgsLTgzMzU3NzU2NywtMz
k4OTcyMzMxXX0=
-->