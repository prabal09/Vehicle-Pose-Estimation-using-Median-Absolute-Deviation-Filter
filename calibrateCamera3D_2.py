# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 12:28:51 2021

@author: prabal
"""

import cv2
import numpy as np
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

def calibCamera(points_xyz,points_c0,points_c1,points_c2,points_c3,points_c4,points_cTV,f_init = 10000):

    # termination criteria 
#    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # Prepare object points, like (0,0,0), (1,0,0), ....,(6,5,0)
#    objp = np.zeros((9*6,3), np.float32)
#    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)*0.0254
    
    # Arrays to store object points and image points from all images
    objpoints = []
    imgpoints = []
    for i in range(6):
        pts_3D = points_xyz.copy()
        if i==0:
            nan_pts = list(set(np.where(np.isnan(points_c0.astype(float)))[0]))
            pts = points_c0.copy()
            pts = np.delete(pts,nan_pts,0)
            pts_3D = np.delete(pts_3D,nan_pts,0)
        elif i==1:
            nan_pts = list(set(np.where(np.isnan(points_c1.astype(float)))[0]))
            pts = points_c1.copy()
            pts = np.delete(pts,nan_pts,0)
            pts_3D = np.delete(pts_3D,nan_pts,0)
        elif i==2:
            nan_pts = list(set(np.where(np.isnan(points_c2.astype(float)))[0]))
            pts = points_c2.copy()
            pts = np.delete(pts,nan_pts,0)
            pts_3D = np.delete(pts_3D,nan_pts,0)
        elif i==3:
            nan_pts = list(set(np.where(np.isnan(points_c3.astype(float)))[0]))
            pts = points_c3.copy()
            pts = np.delete(pts,nan_pts,0)
            pts_3D = np.delete(pts_3D,nan_pts,0)
        elif i==4:
            nan_pts = list(set(np.where(np.isnan(points_c4.astype(float)))[0]))
            pts = points_c4.copy()
            pts = np.delete(pts,nan_pts,0)
            pts_3D = np.delete(pts_3D,nan_pts,0)   
        elif i==5:
            nan_pts = list(set(np.where(np.isnan(points_cTV.astype(float)))[0]))
            pts = points_cTV.copy()
            pts = np.delete(pts,nan_pts,0)
            pts_3D = np.delete(pts_3D,nan_pts,0)
        objpoints.append(pts_3D)
        imgpoints.append(pts) 
        
    
#    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (1600, 900),None,None)
    
    
    (width,height) = (1600,900)
    K_init = np.identity(3)
    K_init[0,0] = f_init
    K_init[1,1] = f_init
    K_init[0,2] = width/2
    K_init[1,2] = height/2
    dist_co_zeros = np.zeros((1,5))
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                    (width,height),K_init,
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
    points_xyz = np.array([
                    [0.0, 0.0, 0.0],
                    [0.0,3.70, 0.0],
                    [21.83, 0.0, 0.0],
                    [9.53,-3.52,0.0],
                    [0.0,-3.52, 0.0],
                    [9.63,0.0,0.0],
                    [19.4,3.70, 0.0],
                    [12.91,0.0, 0.0],
                    [0.97,0,0.0],
                    [7.65,3.70,0.0],
                    [3.30,3.70,0.0],
                    [2.86,-3.52,0.0],
                    [19.28,-3.52, 0.0],
                    [-2.66,0,0.0],
                    [5.08,0,0.0]]
                     ,dtype = np.float32)
    points_c0 = np.array([[[1486,509]],
                          [[None, None]],
                          [[None, None]],
                        [[649,469]],
                        [[1283,449]],
                        [[510,533]],
                        [[None, None]],
                        [[262,536]],
                        [[1393,512]],
                        [[467,703]],
                        [[1345,712]],
                        [[1090,454]],
                        [[124,486]],
                        [[1600, 500]],
                        [[928,525]]], dtype=np.float32)

    points_c1 = np.array([[[1043,196]],
                          [[1301,285]],
                          [[None, None]],
                            [[266,291]],
                            [[880,136]],
                            [[303,472]],
                            [[None, None]],
                            [[None, None]],
                            [[1010,209]],
                            [[762,654]],
                            [[1150,389]],
                            [[759,168]],
                            [[None, None]],
                            [[1150,157]],
                            [[769,302]]], dtype=np.float32)
    
    points_c2 = np.array([[[533,253]],
                          [[702,178]],
                          [[None, None]],
                            [[1103,882]],
                            [[282,359]],
                            [[1283,502]],
                            [[None, None]],
                            [[None, None]],
                            [[571,265]],
                            [[1164,277]],
                            [[862,212]],
                            [[397,434]],
                            [[None, None]],
                            [[430,216]],
                            [[820,348]]], dtype=np.float32)
    
    points_c3 = np.array([[[957,399]],
                          [[662,630]],
                          [[171,143]],
                            [[604,194]],
                            [[1082,301]],
                            [[420,226]],
                            [[None, None]],
                            [[335,197]],
                            [[883,376]],
                            [[187,310]],
                            [[382,436]],
                            [[913,261]],
                            [[None, None]],
                            [[1300,510]],
                            [[597,284]]], dtype=np.float32)
    
    points_c4 = np.array([[[307,473]],
                          [[218,304]],
                          [[1341,120]],
                            [[1359,298]],
                            [[None, None]],
                            [[1071,213]],
                            [[1151,108]],
                            [[1167,180]],
                            [[431,432]],
                            [[805,180]],
                            [[537,237]],
                            [[988,608]],
                            [[1517,169]],
                            [[3,574]],
                            [[843,290]]], dtype=np.float32)
    
    points_cTV = np.array([[[824,276]],
                           [[871,276]],
                           [[824,559]],
                            [[780,394]],
                            [[780,276]],
                            [[824,401]],
                            [[868,528]],
                            [[824,441]],
                            [[824,286]],
                            [[870,378]],
                            [[870,319]],
                            [[780,308]],
                            [[780,523]],
                            [[824,243]],
                            [[824,339]]], dtype=np.float32)
    
    ret, mtx, dist, rvecs, tvecs = calibCamera(points_xyz,points_c0,points_c1,points_c2,points_c3,points_c4,points_cTV,f_init = 10000)
    print(mtx)
    print(dist)
    print(rvecs)
    print(tvecs)
#    path = 'camera3D.yml'
#    save_coefficients(mtx, dist, path)
#    load_coefficients(path)