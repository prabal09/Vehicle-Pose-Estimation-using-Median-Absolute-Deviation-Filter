# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 04:12:08 2021

@author: prabal
"""

## compute the distance of the wheel p-o-c to the front and rear end

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from vp_tangent import *
from vanishing_point_sp import *
from point_of_contact import *

#vpc0 = (-3992, 655)

img = cv2.imread("c0_4_2_30_mask_new.jpg")
img2 = cv2.imread("c0_4_2_30.jpg")


def ext_FRc0(img):
    frnt,rear = FrontRearSeg(img)
    frnt_max = 1600;
    for pt in frnt:
        #print(pt[0])
        if pt[0][0]<frnt_max:
            frnt_end = pt[0]
            frnt_max = pt[0][0]
    rear_min = 0
    for pt in rear:
        if pt[0][0]>rear_min:
            rear_end = pt[0]
            rear_min = pt[0][0]
    imgC = img.copy()            
    imgC =cv2.circle(imgC,tuple(frnt_end),4,(255,0,0),-1)
    imgC =cv2.circle(imgC,tuple(rear_end),4,(0,0,255),-1)
    disp_img(imgC)
    #print(frnt_end,rear_end)
    return frnt_end,rear_end        

def solveeqn(pt,m,c):
    a,b = pt
    A = np.array([
        [m.item(),-1],
        [1,m.item()]])
    B = np.array([[-c.item()], [a+b*m.item()]])
    x0, y0 = np.linalg.solve(A, B)
    x0, y0 = int(np.round(x0)), int(np.round(y0))
    return x0,y0

def wheel2frnt_rear(img,vp,max_c):
    #max_c = drawalltangents(img,contours,vp)
    p,q = PointofContact('c0',img,vp,max_c)  #p->frnt_wheel_pt;q->rear_wheel_pt
    frnt_end,rear_end = ext_FRc0(img)   #a->frnt_end[0]/rear_end[0];b->frnt_end[1]/rear_end[1]   
    m = (p[1]-q[1])/[p[0]-q[0]]
    ##frnt
    c = (p[1]-m*p[0])
    frnt_end_gpt = solveeqn(frnt_end,m,c)
    rear_end_gpt = solveeqn(rear_end,m,c)
    imgC = img.copy()            
    imgC =cv2.circle(imgC,tuple(frnt_end),4,(255,0,0),-1)
    imgC =cv2.circle(imgC,tuple(rear_end),4,(0,0,255),-1)
    imgC =cv2.circle(imgC,tuple(p),4,(255,0,0),-1)
    imgC =cv2.circle(imgC,tuple(q),4,(0,0,255),-1) 
    imgC =cv2.circle(imgC,tuple(frnt_end_gpt),4,(0,255,0),-1)
    imgC =cv2.circle(imgC,tuple(rear_end_gpt),4,(0,255,0),-1)    
    disp_img(imgC) 
    #print(frnt_end_gpt,rear_end_gpt)
    return frnt_end_gpt,rear_end_gpt
    
    

if __name__ == "__main__":
    vpc0 = (-4955, 648)     #(-3992, 655)
    img = cv2.imread("c0_4_2_30_mask_new.jpg")
    img2 = cv2.imread("c0_4_2_30.jpg")
    img0_mask = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c0_4_4_21_mask.jpg")
    img0 = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c0_4_4_21.jpg")
    img = img0_mask
    contours = getContour(img)
    max_c = drawalltangents(img,contours,vpc0)
    frnt_end_gpt,rear_end_gpt = wheel2frnt_rear(img,vpc0,max_c)
    












