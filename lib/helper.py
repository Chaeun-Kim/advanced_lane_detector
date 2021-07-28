import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def save_video(clip, path, filename, audio=False):
    if not os.path.exists(path):
        os.makedirs(path)

    clip.write_videofile(f'{path}/{filename}', audio=audio)

def save_img(img, path, filename):
    if not os.path.exists(path):
        os.makedirs(path)

    return cv2.imwrite(f'{path}/{filename}', img)# cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def plot_side_by_side(l_img, r_img, l_desc='Original Image', r_desc='New Image', l_cmap=None, r_cmap=None):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(l_img, cmap=l_cmap)
    ax1.set_title(l_desc, fontsize=50)
    ax2.imshow(r_img, cmap=r_cmap)
    ax2.set_title(r_desc, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def draw_lane(undistorted, inverse_M, ploty, left_fitx, right_fitx, left_curverad, right_curverad):
    # Create a blank image to draw the lines on
    blank_undistored = np.zeros_like(undistorted).astype(np.uint8)

    # draw lane on the blank image
    left_points = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    right_points = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    points = np.hstack((left_points, right_points))
    cv2.fillPoly(blank_undistored, np.int_([points]), (0, 255, 0))

    newwarp = cv2.warpPerspective(blank_undistored, inverse_M, (undistorted.shape[1], undistorted.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undistorted, 1, newwarp, 0.3, 0)

    avg_curverad = np.around((left_curverad + right_curverad) / 2., decimals=2)
    cv2.putText(result, f'Radius of Curvature = {avg_curverad:.2f}m', (50, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), lineType=1000)

    return result

def get_points_on_lines(lines, shape):
    """
    Loop over the give lines, and return 4 significant cooridinates,
    2 pairs on each of left/right lanes
    """
    left_lines, right_lines = [], []
    left_lane_points, right_lane_points = [], []
    for line in lines:
        for x1,y1,x2,y2 in line:
            if x2 != x1:
                slope = ((y2-y1)/(x2-x1))
                abs_slope = abs(slope)
                minX = min(x1, x2)
                maxX = max(x1, x2)
                if abs_slope > 0.5 and abs_slope < 1:
                    if (slope > 0) and minX > shape[1]/2:
                        right_lines.append(np.polyfit([x1,x2], [y1,y2], 1))
                    elif (slope < 0) and maxX < shape[1]/2:
                        left_lines.append(np.polyfit([x1,x2], [y1,y2], 1))

    if left_lines:
        left_slope, left_intercept = _get_avg_slope_and_intercept(left_lines)
        left_lane_points = _get_points_in_line(left_slope, left_intercept)
    if right_lines:
        right_slope, right_intercept = _get_avg_slope_and_intercept(right_lines)
        right_lane_points = _get_points_in_line(right_slope, right_intercept)

    return left_lane_points, right_lane_points[::-1] #np.float32([*left_lane_points, *right_lane_points[::-1]])

def _get_avg_slope_and_intercept(lines):
    slope = 0
    y_intercept = 0
    for line in lines:
        slope += line[0]
        y_intercept += line[1]

    num_lines = max(len(lines), 1)
    return slope/num_lines, y_intercept/num_lines

def _get_points_in_line(slope, y_intercept):
    y1, y2 = 710, 470
    x1 = (y1 - y_intercept) / slope
    x2 = (y2 - y_intercept) / slope

    return [[x1, y1], [x2, y2]]
