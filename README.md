# Advanced lane detector

Self-Driving Car Engineer Nanodegree Project 2

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.


[//]: # (Image References)
[calibration1]: ./camera_cal/calibration1.jpg "calibration1"
[undistorted_calibration1]: ./output_images/undistorted_calibration1.jpg "undistorted_calibration1"
[canny]: ./data/images/canny_edge.jpg "Canny edge detection"
[hough_no_extent]: ./data/images/hough_no_extent.jpg "Hough without extension"
[hough]: ./data/images/hough_transformed.jpg "Hough transformed"
[result]: ./data/test_images_output/solidYellowCurve.jpg "Hough transformed"

---

### Camera Calibration

The code for this step is contained in `lib/calibrate.py`.

Before processing road image we first need to calibrate the camera to better detect distortions in images taken from cameras. We do so by detecting corners in a set of chessboard images and computing the camera calibration and distortion coefficients using the `calibrateCamera()` function from cv2 python library. Using the coefficient we can undistort images like show below.

Distored chessboard        |  Undistored chessboard
:-------------------------:|:-------------------------:
![alt text][calibration1]  |  ![alt text][undistorted_calibration1]

### Pipeline
