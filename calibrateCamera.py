# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 11:27:50 2021

@author: prabal
"""

import numpy as np
import cv2
import glob
import re 
import math
from pathlib import Path 

file_pattern = re.compile(r'.*?(\d+).*?')
def get_order(file):
    match = file_pattern.match(Path(file).name)
    if not match:
        return math.inf
    return int(match.groups()[0])

def getChessboardFrames(vidPath,outPath):
    cap = cv2.VideoCapture(vidPath)
    count = 0
    while(cap.isOpened()):
    # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video
        ret, frame = cap.read()    
#        print(frame.shape)
        if ret:
            frame = cv2.resize(frame,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
            count +=1
            dstn = outPath + 'img_' + str(count) + '.jpg'
            cv2.imwrite(dstn,frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
        
def displayImg(img):
    cv2.imshow('Image',img)        
    cv2.waitKey(0)

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def calibrate(imgPath,square_size, width=9, height=6,f_init = 10000):
    """ Apply camera calibration operation for images in the given directory path. """
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    objp = np.zeros((height*width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

    objp = objp * square_size

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.


    # Import Images
    imgPath = imgPath + '*.jpg'
    images = glob.glob(imgPath,recursive = True)

    sorted_images = sorted(images,key=get_order)
#    sample1 = sorted_images[::15]
    sample1 = sorted_images[::100]
    
    for fname in sample1:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        print(gray.shape[::-1])
        return gray.shape[::-1]
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
#            img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
#            cv2.imshow('img', img)
#            cv2.waitKey(50)
#    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    (width1,height1) = (3840,2160)
#    (width1,height1) = (1600,900)
    K_init = np.identity(3)
    K_init[0,0] = f_init
    K_init[1,1] = f_init
    K_init[0,2] = width1/2
    K_init[1,2] = height1/2
    dist_co_zeros = np.zeros((1,5))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                    gray.shape[::-1],K_init,
                                                    dist_co_zeros,
                                                    flags= cv2.CALIB_FIX_PRINCIPAL_POINT
                                                    |cv2.CALIB_USE_INTRINSIC_GUESS
                                                    |cv2.CALIB_FIX_ASPECT_RATIO
                                                    |cv2.CALIB_ZERO_TANGENT_DIST
                                                    |cv2.CALIB_FIX_K1
                                                    |cv2.CALIB_FIX_K2
                                                    |cv2.CALIB_FIX_K3
                                                    )
    return [ret, mtx, dist, rvecs, tvecs]
def save_coefficients(mtx, dist, path):
    """ Save the camera matrix and the distortion coefficients to given path/file. """
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_WRITE)
    cv_file.write("K", mtx)
    cv_file.write("D", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()
def load_coefficients(path):
    """ Loads camera matrix and distortion coefficients. """
    # FILE_STORAGE_READ
    cv_file = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode("K").mat()
    dist_matrix = cv_file.getNode("D").mat()

    cv_file.release()
    return [camera_matrix, dist_matrix]

if __name__ == "__main__":
    vidPath = 'C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/osmo_calibration.mp4'
    outPath = 'C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/Chessboard Images/'
#    outPath = 'C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/Chessboard Images_3840x2160/'
#    getChessboardFrames(vidPath,outPath)
    imgPath = outPath    
    ret, mtx, dist, rvecs, tvecs = calibrate(imgPath,square_size = 0.0254)
    print(mtx)
    print(dist)
    path = 'camera.yml'
#    path = 'camera_3840x2160_noflags.yml'
#    path = 'camera_3840x2160.yml'
    save_coefficients(mtx, dist, path)
#    mtx,dist = load_coefficients(path)
