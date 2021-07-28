import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .calibrate import Calibrate
from .helper import get_points_on_lines, save_img

class CV:
    def __init__(self, image, calibrator=None, use_undistored=False):
        if calibrator is None:
            calibrator = Calibrate()
        self.mtx, self.dist = calibrator.get_values()

        self.original_image = self._init_image(image) if type(image) == str else image
        if use_undistored:
            self.original_image = self._undistort_image()

        self.shape = self.original_image.shape
        self.height = self.shape[0]
        self.width = self.shape[1]

    def get_image_binary(self, saturation_thresh=(170, 235), sobel_thresh=(30, 100)):
        s_binary = self._get_saturation_binary(saturation_thresh)
        sobel_binary = self._get_sobel_image_binary(sobel_thresh)

        # Combine the two binaries
        combined_binary = np.zeros_like(sobel_binary)
        combined_binary[(s_binary == 1) | (sobel_binary == 1)] = 1

        return combined_binary

    def _get_saturation_binary(self, saturation_thresh):
        # Get saturation channel image
        hls = self.get_HLS_image()
        s_channel = hls[:,:,2]

        # Create image binary using given thresholds
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= saturation_thresh[0]) & (s_channel <= saturation_thresh[1])] = 1

        return s_binary

    def _get_sobel_image_binary(self, thresh, orient='x', sobel_kernel=3):
        red_channel = self.original_image[:,:,0]
        blurred_red = cv2.GaussianBlur(red_channel, (5, 5), 0)

        sobel = cv2.Sobel(blurred_red, cv2.CV_64F, 1, 0, ksize=sobel_kernel) if orient == 'x' \
            else cv2.Sobel(blurred_red, cv2.CV_64F, 0, 1, ksize=sobel_kernel)

        abs_sobel = np.absolute(sobel)

        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

        return grad_binary

    def warp_image(self, binary):
        # lines = self._get_hough_lines(binary, rho=2, theta=np.pi/180, threshold=50, min_line_len=30, max_line_gap=10)

        # left_lane_points, right_lane_points = get_points_on_lines(lines, self.shape)
        # src_points = np.float32([*left_lane_points, *right_lane_points])
        src_points = np.float32([[210, 700], [560, 470], [720, 470], [1050, 700]])
        dest_points = np.float32([[200, 720], [200, 0], [980, 0], [980, 720]])

        # plt.imshow(self.original_image)
        # for points in src_points:
        #     plt.scatter(*points, color='r')

        # plt.show()

        M = cv2.getPerspectiveTransform(src_points, dest_points)
        # get inverse so that we can unwarp later
        inverse_M = cv2.getPerspectiveTransform(dest_points, src_points)

        warped = cv2.warpPerspective(binary, M, binary.shape[::-1], flags=cv2.INTER_LINEAR)

        return warped, M, inverse_M

    def _get_hough_lines(self, binary, rho, theta, threshold, min_line_len, max_line_gap):
        masked_image = self._mask_region(binary)

        return cv2.HoughLinesP(masked_image, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    def _mask_region(self, binary):
        mask = np.zeros_like(binary)
        vertices = np.array([[
            [0,self.height],
            [self.width/2.1,self.height/1.68],
            [self.width/1.9,self.height/1.68],
            [self.width,self.height]
        ]], dtype=np.int32)

        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(self.shape) > 2:
            channel_count = self.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255

        # filling pixels inside the polygon defined by "vertices" with the fill color
        cv2.fillPoly(mask, vertices, ignore_mask_color)

        # returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(binary, mask)
        return masked_image

    def get_image(self):
        return self.original_image

    def get_gray_image(self):
        return cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)

    def get_HLS_image(self):
        return cv2.cvtColor(self.original_image, cv2.COLOR_RGB2HLS)

    def get_image_shape(self):
        return self.shape

    def _undistort_image(self):
        return cv2.undistort(self.original_image, self.mtx, self.dist, None, self.mtx)

    def _init_image(self, image_path):
        if not os.path.exists(image_path):
            print(f"image does not exist: '{image_path}'")
            sys.exit()

        return mpimg.imread(image_path)
