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


[this free course](https://www.udacity.com/course/writing-readmes--ud777).
<iframe width="560" height="315" src="https://www.youtube.com/embed/-AbUpO2lEOM" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>
<!--stackedit_data:
eyJoaXN0b3J5IjpbLTExODIyMjE3NTksLTEyMTcxMDYwNjAsLT
gyNzAyMTQwMCwxMjE4OTY5OTE4LC04MzM1Nzc1NjcsLTM5ODk3
MjMzMV19
-->