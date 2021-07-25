import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from .helper import plot_side_by_side

CWD = os.getcwd()

class Calibrate:
    def __init__(self, board_corners=(9,6), src_dir='camera_cal', verbose=False):
        self.verbose = verbose

        self.nx = board_corners[0]
        self.ny = board_corners[1]

        self.src_dir = os.path.join(CWD, src_dir)
        self.dump_path =os.path.join(CWD, '/lib/cal.dat')

        if os.path.exists(self.dump_path):
            with open(self.dump_path, "rb") as data:
                cal_data = pickle.load(data)
                self.mtx = cal_data['mtx']
                self.dist = cal_data['dist']
        else:
            self.mtx, self.dist = self._calibrate()

    def get_values(self):
        return self.mtx, self.dist

    def _calibrate(self):
        # Arrays to store object points and image points from all the images.
        obj_points, img_points, gray = self._get_image_points()
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

        if self.verbose:
            for file in os.listdir(SRC_DIR)[:2]:
                img = mpimg.imread(os.path.join(SRC_DIR, file))
                undistorted = cv2.undistort(img, mtx, dist)

                plot_side_by_side(img, undistorted)

        with open(self.dump_path, 'wb') as data:
            pickle.dump({'mtx': mtx, 'dist': dist}, data)

        return mtx, dist

    def _get_image_points(self):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        init_points = np.zeros((self.nx * self.ny, 3), np.float32)
        init_points[:,:2] = np.mgrid[0:self.nx, 0:self.ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        obj_points = [] # 3d points in real world space
        img_points = [] # 2d points in image plane.
        # Step through the list and search for chessboard corners
        for file in os.listdir(self.src_dir):
            path = os.path.join(self.src_dir, file)

            image = mpimg.imread(path)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (self.nx,self.ny), None)
            if ret:
                obj_points.append(init_points)
                img_points.append(corners)

            if self.verbose:
                img = cv2.drawChessboardCorners(image, (self.nx,self.ny), corners, ret)
                cv2.imshow('img',img)
                cv2.waitKey(500)

        return obj_points, img_points, gray
