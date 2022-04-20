import cv2  
import numpy as np 

def binary_array(array, thresh, value=0):
  
    if value == 0:
        # Create an array of ones with the same shape and type as
        # the input 2D array.
        binary = np.ones_like(array)

    else:
        # Creates an array of zeros with the same shape and type as
        # the input 2D array.
        binary = np.zeros_like(array)
        value = 1

  
    binary[(array >= thresh[0]) & (array <= thresh[1])] = value

    return binary
  
  
  
def blur_gaussian(channel, ksize=3):
 
    return cv2.GaussianBlur(channel, (ksize, ksize), 0)

def mag_thresh(image, sobel_kernel=3, thresh=(0, 255)):

    # Get the magnitude of the edges that are vertically aligned on the image
    sobelx = np.absolute(sobel(image, orient='x', sobel_kernel=sobel_kernel))

    # Get the magnitude of the edges that are horizontally aligned on the image
    sobely = np.absolute(sobel(image, orient='y', sobel_kernel=sobel_kernel))

    mag = np.sqrt(sobelx ** 2 + sobely ** 2)

    # Return a 2D array that contains 0s and 1s
    return binary_array(mag, thresh)
  
  
def sobel(img_channel, orient='x', sobel_kernel=3):

    # cv2.Sobel(input image, data type, prder of the derivative x, order of the
    # derivative y, small matrix used to calculate the derivative)
    if orient == 'x':
        # Will detect differences in pixel intensities going from
        # left to right on the image (i.e. edges that are vertically aligned)
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 1, 0, sobel_kernel)
    if orient == 'y':
        # Will detect differences in pixel intensities going from
        # top to bottom on the image (i.e. edges that are horizontally aligned)
        sobel = cv2.Sobel(img_channel, cv2.CV_64F, 0, 1, sobel_kernel)

    return sobel
  
  
def threshold(channel, thresh=(128, 255), thresh_type=cv2.THRESH_BINARY):
  
  return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)
  
  
  
  
  
  
  
  

    # If pixel intensity is greater than thresh[0], make that value
    # white (255), else set it to black (0)
    return cv2.threshold(channel, thresh[0], thresh[1], thresh_type)
