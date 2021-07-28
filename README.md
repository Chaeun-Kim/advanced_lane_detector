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
[road]: ./test_images/straight_lines1.jpg "Distored Road image"
[undistorted_road]: ./output_images/undistorted_road.jpg "Undistorted Road image"
[sobel_red]: ./output_images/sobel_red.jpg "Red Channel Threshold"
[saturation_thresh]: ./output_images/saturation_thresh.jpg "Saturation Threshold"
[combined]: ./output_images/combined.jpg "Combined Threshold"
[warped]: ./output_images/warped.jpg "Warped Combined Binary"
[warped_n]: ./output_images/warped_n.jpg "Warped Original"

---

## Image Processing Pipeline

This section will describe what's all involved in the image processing pipeline.

### Camera Calibration

*The code for this step is contained in `lib/calibrate.py`.*

Before processing road images we first need to calibrate the camera to better detect distortions in images taken from cameras. We do so by detecting corners in a set of chessboard images and computing the camera calibration and distortion coefficients using the `calibrateCamera()` function from cv2 python library. Using the coefficient we can undistort images like show below.

Distored chessboard        |  Undistored chessboard
:-------------------------:|:-------------------------:
![alt text][calibration1]  |  ![alt text][undistorted_calibration1]


Distored road image        |  Undistored road image
:-------------------------:|:-------------------------:
![alt text][road]          |  ![alt text][undistorted_road]

### Find lane lines

*The code for this step is contained in `lib/cv.py` (line 25-60)*

Now we can create a binary image, an image that contains either black or white pixels where white represents edges, off of the undistored road image.

To better catch different colors of lane lines under good/bad circumstances (lanes under shady area, or less daylight), the pipeline uses the combination of gradient threshold on red color channel and saturation color channel of images.

Undistored road image            |  Gradient Threshold applied on Red channel of the image
:-------------------------------:|:--------------------------------------------------------:
![alt text][undistorted_road]    |  ![alt text][sobel_red]

Gradient Threshold applied on Saturation channel of the image            |  Combined
:-----------------------------------------------------------------------:|:----------------:
![alt text][saturation_thresh]                                           |  ![alt text][combined]

### Perspective Transform

Now that we've identified the potential lane lines, the pipeline will change the perspective of the view on the images. It will transform the road image to the "birds-eye view" image.

With below source and destination coordinates, the pipeline will get perspective transform matrix with the `getPerspectiveTransform` function and perform transformation with the `warpPerspective` function from cv2 library.

Source points         |  Destination points
:--------------------:|:-------------------------:
| (210,700)           |  (200,720)
| (560, 470)          |  (200,0)
| (720, 470)          |  (980,0)
| (1050, 700)         |  (980,720)

Perspective Transform result:

Combined                   |  Warped combined binary   | Warped undistorted image
:-------------------------:|:-------------------------:|:-------------------------:
![alt text][combined]      |  ![alt text][warped]      | ![alt text][warped_n]

### Lane detection on Transformed image





