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
from lib.threshold import combined_thresh
from lib.helper import plot_side_by_side, region_of_interest, hough_lines
from lib.transform import corners_unwarp, fit_polynomial

TEST_SRC = f'{os.getcwd()}/test_images'

cal = Calibrate()
MTX, DIST = cal.get_values()

def process_image(image):
    undistorted = cv2.undistort(image, MTX, DIST, None, MTX)

    binary = combined_thresh(undistorted)
    height = binary.shape[0]
    width = binary.shape[1]

    masking_region = np.array([[
        [0,height],
        [width/2.1,height/1.68],
        [width/1.9,height/1.68],
        [width,height]
    ]], dtype=np.int32)
    masked_image = region_of_interest(binary, masking_region)
    src_points = hough_lines(masked_image, 2, np.pi/180, 50, 30, 10)

    warped, M = corners_unwarp(undistorted, binary, src_points)

    # out_img = fit_polynomial(warped)

    plot_side_by_side(warped, binary, l_desc='first', r_desc='second',r_cmap='gray')

    return binary


for image in os.listdir(TEST_SRC):
    img = mpimg.imread(os.path.join(TEST_SRC, image))
    binary = process_image(img)
    # plot_side_by_side(img, binary, l_desc='first', r_desc='second', l_cmap='gray', r_cmap='gray')
    # print(binary)
