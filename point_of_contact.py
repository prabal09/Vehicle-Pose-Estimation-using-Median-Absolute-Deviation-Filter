# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 14:56:07 2021

@author: prabal
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from vp_tangent import *
from vanishing_point_sp import *
vpc4 = (1835,-15)
vpc1 = (1987, -119)

img = cv2.imread("c4_2_1_40_mask_new.jpg")
img2 = cv2.imread("c4_2_1_40.jpg")
#img = cv2.imread("c1_1_0_58_mask.jpg")
img = cv2.imread("c1_1_0_58_mask_new.jpg")
img2 = cv2.imread("c1_1_0_58.jpg")
def slope(x1,y1,x2,y2):
    ###finding slope
    if x2!=x1:
        return((y2-y1)/(x2-x1))
    else:
        return 'NA'
def drawLine(image,pt1,pt2):
    image1 = image.copy()
    x1,y1 = pt1
    x2,y2 = pt2
    m=slope(x1,y1,x2,y2)
    h,w=image.shape[:2]
    imgL = np.zeros_like(image)
    if m!='NA':
        ### here we are essentially extending the line to x=0 and x=width
        ### and calculating the y associated with itc
        ##starting point
        px=0
        py=-(x1-0)*m+y1
        ##ending point
        qx=w
        qy=-(x2-w)*m+y2
    else:
    ### if slope is zero, draw a line with x=x1 and y=0 and y=height
        px,py=x1,0
        qx,qy=x1,h
    image1 = cv2.line(image1, (int(px), int(py)), (int(qx), int(qy)), (0, 255, 0), 1)
    #disp_img(image1)  
    imgL = np.zeros_like(image)
    imgL = cv2.line(imgL, (int(px), int(py)), (int(qx), int(qy)), (255, 255, 255), 2)
    #cv2.imshow("FullLine",imgL)  
    #cv2.waitKey(0)
    return image1

def tangentLineinRho_Theta(img,vp,max_c):
    img__ = np.zeros_like(img,np.uint8)
    img__ = cv2.line(img__,tuple(max_c),tuple(vp),(255,255,255),1)
    #disp_img(img__,False)
    img__ = cv2.Canny(img__, 0, 255)
    Htangent_line = cv2.HoughLines(img__,1.3,np.pi/180,200)
    return Htangent_line[0]
    
def tangentLineinSlope(img,vp,max_c):
    m = slope(max_c[0],max_c[1],vp[0],vp[1])
    c = max_c[1]-m*max_c[0]
    drawLine(img,vp,max_c)
    return m,c


def distPt_Tgnt(pt,line):
    m,c = line
    sq = math.sqrt(m**2+1)
    val = abs(pt[1]-m*pt[0]-c)/sq
    return val

def ContourCenter(img):
    imgCC = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    mass_y, mass_x = np.where(imgCC >= 255)
    cent_x = np.average(mass_x)
    cent_y = np.average(mass_y)
    img = cv2.circle(img,(int(cent_x),int(cent_y)),1,(0,255,0),thickness = 5)
    #disp_img(img)
    return int(cent_x*1.10),int(cent_y)



def FrontRearSeg(img):
    imgC = img.copy()
    contours = getContour(img)
    ctr_pts = np.vstack(contours)
    cx,cy = ContourCenter(imgC)
    front = [];rear = []
    for pt in ctr_pts:
        if pt[0][0] < cx:
            front.append(pt)
        else:
            rear.append(pt)
    fr_img = np.zeros_like(img,np.uint8)
    for pt in front:
        #print(pt)
        y,x = pt[0]
        fr_img[x,y] = [255,0,0]
        fr_img[x+1,y] = [255,0,0];fr_img[x-1,y] = [255,0,0]
        fr_img[x,y+1] = [255,0,0];fr_img[x,y-1] = [255,0,0]
        fr_img[x+1,y+1] = [255,0,0];fr_img[x-1,y-1] = [255,0,0]
        fr_img[x+2,y] = [255,0,0];fr_img[x-2,y] = [255,0,0]
        fr_img[x,y+2] = [255,0,0];fr_img[x,y-2] = [255,0,0]
        fr_img[x+2,y+2] = [255,0,0];fr_img[x-2,y-2] = [255,0,0] 
    for pt in rear:
        #print(pt)
        y,x = pt[0]
        fr_img[x,y] = [0,0,255]   
        fr_img[x+1,y] = [0,0,255];fr_img[x-1,y] = [0,0,255]
        fr_img[x,y+1] = [0,0,255];fr_img[x,y-1] = [0,0,255]
        fr_img[x+1,y+1] = [0,0,255];fr_img[x-1,y-1] = [0,0,255]
        fr_img[x+2,y] = [0,0,255];fr_img[x-2,y] = [0,0,255]
        fr_img[x,y+2] = [0,0,255];fr_img[x,y-2] = [0,0,255]
        fr_img[x+2,y+2] = [0,0,255];fr_img[x-2,y-2] = [0,0,255]     
    fr_img = cv2.circle(fr_img,(cx,cy),1,(0,255,0),thickness = 5)
#    disp_img(fr_img)
    return front,rear

def PointofContact(cam_name,img,vp,max_c):
    imgP = img.copy()
    contours = np.vstack(getContour(imgP))
    if cam_name == 'c4' or cam_name == 'c3':
        rear,front = FrontRearSeg(img)
    else:
        front,rear = FrontRearSeg(img)
    line = tangentLineinSlope(img,vp,max_c)
    #ctr_pts = np.vstack(contours)
    val_min_f = np.inf;frnt_pt = None
    for c_pt in front:
        pt = c_pt[0]
        #print(pt)
        val = distPt_Tgnt(pt,line)
        #print('pt is',pt,'val', val)
        if val < val_min_f:
            val_min_f = val
            frnt_pt = pt
    val_min_r = np.inf;rear_pt = None
    for c_pt in rear:
        pt = c_pt[0]
        #print(pt)
        val = distPt_Tgnt(pt,line)
        #print('pt is',pt,'val', val)
        if val < val_min_r:
            val_min_r = val
            rear_pt = pt 
#    for pts in contours:
#        if pts[0][1] > rear_pt[1]:
#            print(pts[0][1])
#            rear_pt = [pts[0][0],pts[0][1]]
    imgP = drawLine(imgP,tuple(frnt_pt),tuple(rear_pt))        
    imgP = cv2.circle(imgP,tuple(frnt_pt),1,(255,0,0),thickness = 5)
    imgP = cv2.circle(imgP,tuple(rear_pt),1,(0,0,255),thickness = 5)
#    disp_img(imgP)
    return frnt_pt,rear_pt

def contourMax(contours):
    max_c = -np.inf;rp = None
    for pt in contours:
        if pt[0][1]>max_c:
            print(pt[0][1])
            max_c = pt[0][1]
            rp = [pt[0][0],pt[0][1]]
    
if __name__ == "__main__":
    cam = input("CAM?")
    if cam == 'c4':
        img = cv2.imread("c4_2_1_40_mask_new.jpg")
        img2 = cv2.imread("c4_2_1_40.jpg")
        img4_mask  = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c4_3_1_52_mask.jpg")
        img4 = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c4_3_1_52.jpg")    
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgC = img4_mask.copy()
        vp = (1740, -16)    #(1835,-15)
    elif cam == 'c1':
        img = cv2.imread("c1_1_0_58_mask_new.jpg")
        img2 = cv2.imread("c1_1_0_58.jpg")
        img1_mask  = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c1_2_1_10_mask.jpg")
        img1 = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c1_2_1_10.jpg")
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        imgC = img1_mask.copy()
        vp = (1892, -120)   #(1987, -119) 
    else:
        img = cv2.imread("c0_4_2_30_mask_new.jpg")
        img2 = cv2.imread("c0_4_2_30.jpg")
        img0_mask = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c0_5_2_41_mask.jpg")
        img0 = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c0_5_2_41.jpg")
        imgC = img0_mask.copy()
        vp = (-4955, 648)   #(-3992, 655)
    contours = getContour(imgC)
    max_c = drawalltangents(imgC,contours,vp)
    frnt_pt,rear_pt = PointofContact(cam,imgC,vp,max_c)
    print(frnt_pt,rear_pt)
    
#    img0_mask = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c0_5_2_41_mask.jpg")
#    img0 = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c0_5_2_41.jpg")
#    img1_mask  = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c1_2_1_10_mask.jpg")
#    img1 = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c1_2_1_10.jpg")
#    img4_mask  = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c4_3_1_52_mask.jpg")
#    img4 = cv2.imread("C:/Users/praba/PycharmProjects/AvaCar/OpenCV Practice/Snapshots/Calibration Frame/img/c4_3_1_52.jpg")     
    
    
    
    