import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

class CV:
	def __init__(self, image_path):
		self.cv2_image = self._init_image(image_path)

	def undistort_image(self, mtx, dist):
		return cv2.undistort(self.cv2_image, mtx, dist, None, mtx)

	def calibrate_camera(self, objp, imgp):
		return cv2.calibrateCamera(objp, imgp, self.get_image_shape(), None, None)

	def find_chess_board_corners(self, gray=True, corners=(9,6)):
		image = self.get_gray_image() if gray else self.cv2_image

		return cv2.findChessboardCorners(image, corners, None)

	def draw_chess_board_corners(self, corners, coordinates, ret, wait=500):
		img = cv2.drawChessboardCorners(self.cv2_image, corners, coordinates, ret)
		cv2.imshow('img',img)
		cv2.waitKey(wait)

	def get_gray_image(self):
		return cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2GRAY)

	def get_HLS_image(self):
		return cv2.cvtColor(self.cv2_image, cv2.COLOR_BGR2HLS)

	def get_image_shape(self):
		return self.cv2_image.shape[1::-1]

	def _init_image(self, image_path):
		if not os.path.exists(image_path):
			print(f"image file does not exist: '{image_path}'")
			sys.exit()

		return cv2.imread(image_path)

# 	f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
# ax1.imshow(img)
# ax1.set_title('Original Image', fontsize=30)
# ax2.imshow(dst)
# ax2.set_title('Undistorted Image', fontsize=30)
