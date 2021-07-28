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

from moviepy.editor import VideoFileClip

from lib.cv import CV
from lib.calibrate import Calibrate
from lib.helper import plot_side_by_side, save_video, save_img, draw_lane
from lib.fit_lane_lines import fit_polynomial, measure_curvature

CWD = os.getcwd()
TEST_SRC = f'{CWD}/test_images'
VIDEO_DIR = f'{CWD}/test_videos'
OUT_VIDEO_DIR = f'{CWD}/out_videos'

cal = Calibrate()
MTX, DIST = cal.get_values()

def process_image(image):
    # cal = Calibrate()
    cv = CV(image, cal, use_undistored=True)

    binary = cv.get_image_binary()
    warped, M, inverse_M = cv.warp_image(binary)
    norm_warped = warped / 255

    ploty, left_fitx, right_fitx, curv_left_fit, curv_right_fit, out_img = fit_polynomial(norm_warped)
    left_curverad, right_curverad = measure_curvature(ploty, curv_left_fit, curv_right_fit)

    result = draw_lane(cv.get_image(), inverse_M, ploty, left_fitx, right_fitx, left_curverad, right_curverad)

    return result

def find_lanes_on_images():
    """
    Go through images in the /data/images/ directory and
    find the lanes in each image and save off the output
    """
    for image in os.listdir(TEST_SRC):
        img = mpimg.imread(os.path.join(TEST_SRC, image))
        result = process_image(img)
        save_img(result, 'output_images', f'processed_{image}')

def find_lanes_on_videos():
    """
    Go through videos in the /data/videos/ directory and
    trace the lanes in each video and save off the output
    """
    for video in os.listdir(VIDEO_DIR):
        if video == 'project_video.mp4':
            clip = VideoFileClip(f'{VIDEO_DIR}/{video}')
            processed_clip = clip.fl_image(process_image)
            save_video(processed_clip, OUT_VIDEO_DIR, video)

# First part of Project 2 - lanes on images
find_lanes_on_images()
# Second part of Project 2 - lanes on a videos
find_lanes_on_videos()
