import cv2
import numpy as np
import matplotlib.pyplot as plt

def plot_side_by_side(l_img, r_img, l_desc='Original Image', r_desc='New Image', l_cmap=None, r_cmap=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(l_img, cmap=l_cmap)
    ax1.set_title(l_desc, fontsize=50)
    ax2.imshow(r_img, cmap=r_cmap)
    ax2.set_title(r_desc, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    `vertices` should be a numpy array of integer points.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    left_lines, right_lines = [], []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 != x1:
                slope = ((y2-y1)/(x2-x1))
                abs_slope = abs(slope)
                minX = min(x1, x2)
                maxX = max(x1, x2)
                if abs_slope > 0.5 and abs_slope < 1:
                    if (slope > 0) and minX > img.shape[1]/2:
                        print(f'{((y2-y1)/(x2-x1))}')
                        right_lines.append(np.polyfit([x1,x2], [y1,y2], 1))
                    elif (slope < 0) and maxX < img.shape[1]/2:
                        print(f'{((y2-y1)/(x2-x1))}')
                        left_lines.append(np.polyfit([x1,x2], [y1,y2], 1))

    left_slope, left_intercept = _get_avg_slope_and_intercept(left_lines)
    left_lane_points = _get_points_in_line(left_slope, left_intercept)

    right_slope, right_intercept = _get_avg_slope_and_intercept(right_lines)
    right_lane_points = _get_points_in_line(right_slope, right_intercept)

    return [*left_lane_points, *right_lane_points[::-1]]

def _get_best_fit_lines(lines, shape):
    """
    'lines' is an array containig points for hough lines of the image.
        [ [[x1,y1,x2,y2]], [[x1,y1,x2,y2]], [[x1,y1,x2,y2]] ]

    Decide whether each set of points forms a line on the left/right side
    of the image by getting their slopes. Negative slope means the line
    is on the left side of the image (considering that the origin [0,0]
    is on the top left corner)

    Then, get the average of the left/right lines.
    Average line = (avg slope of lines)*X + (avg y intercepts of lines)
    """
    right_lines = [
        np.polyfit([x1,x2], [y1,y2], 1)
        for line in lines
        for x1,y1,x2,y2 in line
        if ((y2-y1)/(x2-x1)) > 0 and min(x1, x2) > shape[1]/2
    ]
    left_lines = [
        np.polyfit([x1,x2], [y1,y2], 1)
        for line in lines
        for x1,y1,x2,y2 in line
        if ((y2-y1)/(x2-x1)) < 0 and max(x1, x2) < shape[1]/2
    ]

    left_slope, left_intercept = _get_avg_slope_and_intercept(left_lines)
    right_slope, right_intercept = _get_avg_slope_and_intercept(right_lines)

    left_lane_points = _get_points_in_line(shape, left_slope, left_intercept)
    right_lane_points = _get_points_in_line(shape, right_slope, right_intercept)

    return np.array([
        [left_lane_points],
        [right_lane_points]
    ], dtype=np.int32)

def _get_avg_slope_and_intercept(lines):
    slope = 0
    y_intercept = 0
    for line in lines:
        slope += line[0]
        y_intercept += line[1]

    num_lines = max(len(lines), 1)
    return slope/num_lines, y_intercept/num_lines

def _get_points_in_line(slope, y_intercept):
    y1, y2 = 710, 480
    x1 = (y1 - y_intercept) / slope
    x2 = (y2 - y_intercept) / slope

    return [[x1, y1], [x2, y2]]
