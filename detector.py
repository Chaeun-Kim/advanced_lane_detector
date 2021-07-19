"""
1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
2. Apply a distortion correction to raw images.
3. Use color transforms, gradients, etc., to create a thresholded binary image.
4. Apply a perspective transform to rectify binary image (\"birds-eye view\").
5. Detect lane pixels and fit to find the lane boundary.
6. Determine the curvature of the lane and vehicle position with respect to center.
7. Warp the detected lane boundaries back onto the original image.
8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from lib.calibrate import Calibrate
from lib.threshold import abs_sobel_thresh, abs_sobel_thresh, mag_thresh, dir_thresh, combined_thresh
from lib.helper import plot_side_by_side

TEST_SRC = f'{os.getcwd()}/test_images'

cal = Calibrate()
MTX, DIST = cal.get_values()

def process_image(image):
    undistorted = cv2.undistort(img, MTX, DIST, None, MTX)

    gaussian_k = 5
    gray = cv2.cvtColor(undistorted, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (gaussian_k, gaussian_k), 0)

    ksize = 15
    gradx = abs_sobel_thresh(blur_gray, orient='x', sobel_kernel=ksize, thresh=(20, 100))
    grady = abs_sobel_thresh(blur_gray, orient='y', sobel_kernel=ksize, thresh=(20, 100))
    mag_binary = mag_thresh(blur_gray, sobel_kernel=ksize, mag_thresh=(30, 100))
    dir_binary = dir_thresh(blur_gray, sobel_kernel=ksize, thresh=(0.7, 1.3))

    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1) | (mag_binary == 1) & (dir_binary == 1))] = 1

    combined2 = combined_thresh(img)

    return combined, combined2


for image in os.listdir(TEST_SRC):
    img = mpimg.imread(os.path.join(TEST_SRC, image))
    combined, combined2 = process_image(img)
    plot_side_by_side(combined, combined2, l_desc='first', r_desc='second', l_cmap='gray', r_cmap='gray')
