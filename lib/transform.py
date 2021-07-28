import cv2
import numpy as np
import matplotlib.pyplot as plt

from lib.helper import plot_side_by_side

YM_PER_PIX = 30/720 # meters per pixel in y dimension
XM_PER_PIX = 3.7/700 # meters per pixel in x dimension

LEFT_FIT = None
RIGHT_FIT = None

qdef fit_polynomial(warped):
    # Find our lane pixels first
    # if RESET_LANE_SEARCH:
    global LEFT_FIT
    global RIGHT_FIT
    if LEFT_FIT is not None and RIGHT_FIT is not None:
        leftx, lefty, rightx, righty, out_img = _search_around_poly(warped)
    else:
        leftx, lefty, rightx, righty, out_img = _find_lane_pixels(warped)
        if leftx.shape[0] != lefty.shape[0] or rightx.shape[0] != righty.shape[0] :
            leftx, lefty, rightx, righty, out_img = _search_around_poly(warped)

    ploty, left_fitx, right_fitx, left_fit, right_fit, curv_left_fit, curv_right_fit = _fit_poly(warped.shape, leftx, lefty, rightx, righty)

    LEFT_FIT = left_fit
    RIGHT_FIT = right_fit

    return ploty, left_fitx, right_fitx, curv_left_fit, curv_right_fit, out_img

def _find_lane_pixels(warped):
    # Take a histogram of the bottom half of the image
    btm_half = warped[warped.shape[0]//2-50:, :]
    histogram = np.sum(btm_half, axis=0)

    # Create an output image to draw on and visualize the result
    out_img = np.dstack((warped,warped,warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPERPARAMETERS
    # Choose the number of sliding windows
    nwindows = 9
    # Set the width of the windows +/- margin
    margin = 60
    # Set minimum number of pixels found to recenter window
    minpix = 50

    # Set height of windows - based on nwindows above and image shape
    window_height = np.int(warped.shape[0]//nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_idxs = []
    right_lane_idxs = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = warped.shape[0] - (window+1)*window_height
        win_y_high = warped.shape[0] - window*window_height
        # boundaries of the window
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window ###
        good_left_idxs = (
            (nonzeroy >= win_y_low) &
            (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) &
            (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_idxs = (
            (nonzeroy >= win_y_low) &
            (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) &
            (nonzerox < win_xright_high)
        ).nonzero()[0]

        # Append these indices to the lists
        left_lane_idxs.append(good_left_idxs)
        right_lane_idxs.append(good_right_idxs)

        if len(good_left_idxs) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_idxs]))
        if len(good_right_idxs) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_idxs]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    left_lane_idxs = np.concatenate(left_lane_idxs)
    right_lane_idxs = np.concatenate(right_lane_idxs)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_idxs]
    lefty = nonzeroy[left_lane_idxs]
    rightx = nonzerox[right_lane_idxs]
    righty = nonzeroy[right_lane_idxs]

    return leftx, lefty, rightx, righty, out_img

def _search_around_poly(warped):
    # HYPERPARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 40

    # Grab activated pixels
    nonzero = warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_idxs = (
        ( nonzerox > (LEFT_FIT[0] * (nonzeroy ** 2) + LEFT_FIT[1] * nonzeroy + LEFT_FIT[2] - margin) ) &
        ( nonzerox < (LEFT_FIT[0] * (nonzeroy ** 2) + LEFT_FIT[1] * nonzeroy + LEFT_FIT[2] + margin) )
    )
    right_lane_idxs = (
        ( nonzerox > (RIGHT_FIT[0] * (nonzeroy ** 2) + RIGHT_FIT[1] * nonzeroy + RIGHT_FIT[2] - margin) ) &
        ( nonzerox < (RIGHT_FIT[0] * (nonzeroy ** 2) + RIGHT_FIT[1] * nonzeroy + RIGHT_FIT[2] + margin) )
    )

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_idxs]
    lefty = nonzeroy[left_lane_idxs]
    rightx = nonzerox[right_lane_idxs]
    righty = nonzeroy[right_lane_idxs]

    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((warped, warped, warped))*255

    return leftx, lefty, rightx, righty, out_img

def _fit_poly(img_shape, leftx, lefty, rightx, righty):
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    curv_left_fit = np.polyfit(lefty * YM_PER_PIX, leftx * XM_PER_PIX, 2)
    curv_right_fit = np.polyfit(righty * YM_PER_PIX, rightx * XM_PER_PIX, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0]-1, img_shape[0])
    try:
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fitx = 1*ploty**2 + 1*ploty
        right_fitx = 1*ploty**2 + 1*ploty

    return ploty, left_fitx, right_fitx, left_fit, right_fit, curv_left_fit, curv_right_fit

def measure_curvature(ploty, curv_left_fit, curv_right_fit):

    y_eval = np.max(ploty)

    # Calculation of R_curve (radius of curvature)
    left_curverad = (( 1 + ( 2 * curv_left_fit[0] * y_eval * YM_PER_PIX + curv_left_fit[1] ) ** 2 ) ** 1.5 ) / np.absolute(2 * curv_left_fit[0])
    right_curverad = (( 1 + ( 2 * curv_right_fit[0] * y_eval * YM_PER_PIX + curv_right_fit[1] ) ** 2) ** 1.5) / np.absolute(2 * curv_right_fit[0])
    avg_curverad = np.around((left_curverad + right_curverad) / 2., decimals=2)

    return avg_curverad, left_curverad, right_curverad

def lane_overlay(undist, inverse_M, ploty, left_fitx, right_fitx, avg_curverad):
    # Create an image to draw the lines on
    color_warp = np.zeros_like(undist).astype(np.uint8)

    # Recast the x and y points into usable format
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (inverse_M)
    newwarp = cv2.warpPerspective(color_warp, inverse_M, (undist.shape[1], undist.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    stat_cur = 'Radius of Curvature = ' + format(avg_curverad, '.2f') + 'm'
    # stat_dep = 'Vehicle is ' + format(depart, '.2f') + l_r + 'of centre'
    cv2.putText(result, stat_cur, (50, 60), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), lineType=1000)

    return result
