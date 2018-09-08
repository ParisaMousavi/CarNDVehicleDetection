import cv2
import numpy as np
from skimage.feature import hog

def bin_spatialI(img, size=(32, 32)):
    small_img = cv2.resize(img, size)
    #to convert this to a one dimensional feature vector
    feature_vec = small_img.ravel()
    return feature_vec


def convert_color(image, color_space='BGR2YCrCb'): #ndarray
    if color_space != 'RGB':
        if color_space == 'HSV':
            image_result = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            image_result = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            image_result = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            image_result = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'BGR2YCrCb':
            image_result = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        elif color_space == 'YCrCb':
            image_result = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        elif color_space == 'GRAY':
            image_result = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        elif color_space == 'RGB2YCrCb':
            image_result = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else: image_result = np.copy(image)
    return image_result

def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins , range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins , range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins , range=bins_range)
    
    
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    
    # Return the individual histograms, bin_centers and feature vector
    return channel1_hist,channel2_hist,channel3_hist,hist_features


'''
Extracting the Histogram of Oriented Gradient
'''
def get_hog_features(myimg, orient, pix_per_cell, 
                     cell_per_block,vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(myimg, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell,pix_per_cell),
                                  cells_per_block=(cell_per_block,cell_per_block), 
                                  transform_sqrt=False, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(myimg, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=False, 
                       visualise=vis, feature_vector=feature_vec)
        return features
    
    
    
    
    
    
    
    
