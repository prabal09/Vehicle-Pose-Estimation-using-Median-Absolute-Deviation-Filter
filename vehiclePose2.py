# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 22:35:53 2021

@author: prabal
"""

import cv2
import numpy as np
import math
from vanishing_point_sp import *
from homography_projection import *
import glob
from utils_2 import get_imgL


def position_framework(c0,c1,c4,cTV,dict_m,dict_s):
    color = (255,255,255);thickness = 1
    img_G = cTV.img.copy()
    line_xb = [(0,0),(1600,0)];line_yb = [(1600,0),(1600,900)]
    frame_array = []
    for ii in range(len(dict_s['c0'])):
        m0 = cv2.imread(dict_m['c0'][ii]);im0 = cv2.imread(dict_s['c0'][ii])
        m0 = cv2.resize(m0,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC);im0 = cv2.resize(im0,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        c0.img = im0;c0.mask = m0
        m1 = cv2.imread(dict_m['c1'][ii]);im1 = cv2.imread(dict_s['c1'][ii])
        m1 = cv2.resize(m1,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC);im1 = cv2.resize(im1,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)        
        c1.img = im1;c1.mask = m1
#        m2 = cv2.imread(dict_m['c2'][ii]);im2 = cv2.imread(dict_s['c2'][ii])
#        m2 = cv2.resize(m2,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC);im2 = cv2.resize(im2,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
#        c2.img = im2;c2.mask = m2
#        m3 = cv2.imread(dict_m['c3'][ii]);im3 = cv2.imread(dict_s['c3'][ii])
#        m3 = cv2.resize(m3,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC);im3 = cv2.resize(im3,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)        
#        c3.img = im3;c3.mask = m3
        m4 = cv2.imread(dict_m['c4'][ii]);im4 = cv2.imread(dict_s['c4'][ii])
        m4 = cv2.resize(m4,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC);im1 = cv2.resize(im4,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)        
        c4.img = im4;c4.mask = m4        
        cTV.img = img_G.copy()
        imgDisp,cntr = locVehicle(c0,c1,c4,cTV,ret_type='img')
        masks = dict_m['c4'][0:ii+1];imgs = dict_s['c4'][0:ii+1]
        if len(masks) >1:
            poc = optFlowVector_PoC(masks,imgs,horizon = True)
            #poc = optFlowVector_PoC(masks,imgs)
            #poc = optFlowVector_PoC(masks,imgs,concensus = 'rad')
            #poc = optFlowVector_PoC(masks,imgs,concensus = 'rad',horizon = True)
            cntr_h = frameChange(cntr,cTV,c4)
            line_c4 = [tuple(cntr_h),tuple(poc)]
            if intersection_mc([line_xb,line_c4])[0]<= 1600:
                poc_ = intersection_mc([line_xb,line_c4])
            else:
                poc_ = intersection_mc([line_yb,line_c4])
            poc_w = frameChange(poc_,c4,cTV)
            imgDisp = cv2.line(imgDisp,tuple(cntr),tuple(poc_w),color,thickness)            
            imgDisp,vpcx_ = drawVPLinePose(imgDisp,c4,cntr,line_xb,line_yb)
            m1,_ = mc([tuple(cntr),tuple(poc_w)]);m2,_=mc([tuple(cntr),tuple(vpcx_)])
            m1,m2 = abs(m1),abs(m2)
            dev_angle = math.atan(abs((m1-m2)/(1+m1*m2)))*180/np.pi
            text = 'deviation from VP: ' + str(dev_angle)
            imgDisp = cv2.putText(imgDisp,text, (1100, 100),cv2.FONT_HERSHEY_SIMPLEX,1, 
                 (0, 0, 255),2, cv2.LINE_AA, False)
            print(dev_angle)
        cv2.imshow("Pose", imgDisp)
        cv2.waitKey(300)
        frame_array.append(imgDisp)

        if cv2.waitKey(10) & 0xFF == ord('q') :
            break
    cv2.destroyAllWindows()
    return frame_array

def saveframe2vid(frame_array,pathOut,fps):
    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, (1600, 900))
    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release() 

    
def drawVPLinePose(img_,cx,cntr,line_xb,line_yb):
    color = (0,255,0);thickness=2
    vpcx = ShowVanishingPoint(cx.img.copy(),getRho_Theta(cx.img.copy(),cx.x_lines))
    line_cx = [tuple(cntr),tuple(vpcx)]
    if intersection_mc([line_xb,line_cx])[0]<= 1600:
        vpcx_ = intersection_mc([line_xb,line_cx])
    else:
        vpcx_ = intersection_mc([line_yb,line_cx])
    vpcx_ = frameChange(vpcx_,cx,cTV)
    img_ = drawDottedline(img_,tuple(cntr),tuple(vpcx_),color,thickness)
    return img_,vpcx_

def frameChange(pt,cx,cy):  #locating pt in cy with originally being in cx
    hcx = cx.homography(cy)
    pt_temp = getH_InvCoordinate(pt)
    pt_ = np.matmul(hcx,np.transpose(pt_temp))
    pt__ = getH_InvCoordinate(pt_,inv=True)
    return pt__

def optFlowVector_PoC(masks,imgs,concensus = 'grad',horizon = False):   #concensus = 'grad/rad'
    # Parameters for ShiTomasi corner detection
#    feature_params = dict(maxCorners=100, qualityLevel=0.4, minDistance=2, blockSize=7)
#
#    # Parameters for Lucas Kanade optical flow
#    lk_params = dict(
#        winSize=(15, 15),
#        maxLevel=2,
#        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
#    )
    feature_params = dict( maxCorners = 200,
                       qualityLevel = 0.2,
                       minDistance = 2,
                       blockSize = 7 )
    lk_params = dict(winSize = (15,15), maxLevel = 3, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    # Create random colors
    color = np.random.randint(0, 255, (1000, 3),dtype = np.int)
    #color = (0, 255, 0)
    first_frame = cv2.imread(masks[0])
    first_frame = cv2.resize(first_frame,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    prev = cv2.goodFeaturesToTrack(prev_gray, mask = None, **feature_params)
    mask = np.zeros_like(first_frame)
    points = np.array([]);q = np.array([False])
    for j in range(len(masks)):
        m = cv2.imread(masks[j]);im = cv2.imread(imgs[j])
        m = cv2.resize(m,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        im = cv2.resize(im,(1600,900),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(m, cv2.COLOR_BGR2GRAY)
        new, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev, None, **lk_params)
        good_old = prev[status == 1]
        good_new = new[status == 1]
        of_vect = []        
        if horizon:
            of_vect = [[(1740, -16), (141, -36)]]
#        horizon = [(1740, -16), (141, -36)]

        for i, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            col_ = (int(color[i][0]),int(color[i][1]), int (color[i][2])) 
            if (c-a) !=0:
                #print(str(i),(b-d)/(c-a))
                if concensus == 'grad':
                    grad = (b-d)/(c-a)
                    points = np.append(points,grad)
                elif concensus == 'rad':
                    rad = math.sqrt((a-b)**2+(c-d)**2)
                    points = np.append(points,rad)
                if len(points)>2:
                    q = is_outlier(points)
            if q[-1].item() is False:
                #print('(a,b)',(a,b),'(c,d)',(c,d))
                # Draws line between new and old position with green color and 2 thickness
                mask = cv2.line(mask, (a, b), (c, d), tuple(col_), 2)
                # Draws filled circle (thickness of -1) at new position with green color and radius of 3
                m = cv2.circle(m, (a, b), 3, tuple(col_), -1)
                of_vect.append([(c,d),(a,b)])
#                of_vect = [(c,d),(a,b)]
#                line__ = [of_vect,horizon]
#                if j>0:
#                    poc = intersection_mc(line__)
#                    print(poc)
        if j>0:
            poc = intersection_mc(of_vect)
        else:
            poc = None
            #print(poc)
        # Overlays the optical flow tracks on the original frame
        output = cv2.add(im, mask)
        # Updates previous frame
        prev_gray = gray.copy()
        # Updates previous good feature points
        prev = good_new.reshape(-1, 1, 2)
        # Opens a new window and displays the output frame
#        cv2.imshow("sparse optical flow", output)
#        cv2.waitKey(300)
#        if cv2.waitKey(10) & 0xFF == ord('q') :
#            break
#    cv2.destroyAllWindows()   
    print(len(points))
    print(len(q))
    points = points[np.where(q==False)]
    print('Outliers',sum(q))
#    print(of_vect)
    return poc

def is_outlier(points, thresh=0.25):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points.shape) == 1:
        points = points[:,None]
    median = np.median(points, axis=0)
    diff = np.sum((points - median)**2, axis=-1)
    diff = np.sqrt(diff)
    med_abs_deviation = np.median(diff)

    modified_z_score = 0.6745 * diff / med_abs_deviation

    return modified_z_score > thresh


if __name__ =="__main__":
    ptsc4=np.float32([[307,473],[218,304],[1341,120],[1359,298],[None,None],[1071,213],[1151,108],[1167,180],[431,432],[805,180],[537,237],[988,608],[1517,169],[None,None],[843,290]]) #c4
    ptsTV=np.float32([[824, 276],[871 ,276],[824 ,559],[780 ,394],[780,276],[824,401],[868,528],[824,441],[824 ,286],[870,376],[870,317],[None,None],[780,523],[824,243],[824,339],[700,464],[691,323],[730,322],[570,335]]) #TV
    ptsc1=np.float32([[1043,196],[1301,285],[None,None],[266,291],[880,136],[303,472],[None,None],[None,None],[1010,209],[762,654],[1150,389],[759,168],[None,None],[1150,157],[769,302]]) #c1
    ptsc0=np.float32([[1486,509],[None,None],[None,None],[649,469],[1283,449],[510,533],[None,None],[262,536],[1393,512],[467,703],[1345,712],[1090,454],[124,486],[None,None],[928,525],[529,421],[1012,397],[1033,414],[907,363]]) #c0
    ptsc3=np.float32([[957,399],[662,630],[171,143],[604,194],[1082,301],[420,226],[None,None],[335,197],[883,376],[187,310],[382,436],[913,261],[None, None],[1300,510],[597,284]]) #c3
    x_linesC0 = [[(649,469),(1283,449)],[(1486,509),(262,536)],[(467,703),(1345,712)]]
    x_linesC1 = [[(266,291),(880,136)],[(303,472),(1043,196)],[(762,654),(1301,285)]]
    x_linesC4 = [[(1517,169),(988,608)],[(307,473),(1341,120)],[(218,304),(1151,108)]]
    x_linesC3 = [[(604,194),(1082,301)],[(957,399),(171,143)],[(662,630),(187,310)]]
    
    y_linesC0 = [[(1486,509), (1283,449)]]
    y_linesC1 = [[(1301,285), (880,136)]]
    y_linesC4 = [[(301, 479), (244, 295)]]
    y_linesC3 = [[(662,630), (1082,301)]]
    y_linesC2 = [[(702,178), (282,359)]]
    y_lines_cTV4 = [[(824,339),(871,339)],[(780,523),(871,523)]]
    
    imgTV = cv2.imread('google_top_view_1600x900.jpg')
    imgc0 = cv2.imread('frame_0.jpg')
    imgc1 = cv2.imread('frame_1.jpg')
    imgc2 = cv2.imread('frame_2.jpg')
    imgc3 = cv2.imread('frame_3.jpg')    
    imgc4 = cv2.imread('frame_4.jpg')
    add_lm_c0 = get_imgL(cam='c0',img_type = 'Mask')
    add_ls_c0 = get_imgL(cam='c0',img_type='Screenshots')
    add_lm_c1 = get_imgL(cam='c1',img_type = 'Mask')
    add_ls_c1 = get_imgL(cam='c1',img_type='Screenshots')
    add_lm_c2 = get_imgL(cam='c2',img_type = 'Mask')
    add_ls_c2 = get_imgL(cam='c2',img_type='Screenshots')
    add_lm_c3 = get_imgL(cam='c3',img_type = 'Mask')
    add_ls_c3 = get_imgL(cam='c3',img_type='Screenshots')
    add_lm_c4 = get_imgL(cam='c4',img_type = 'Mask')
    add_ls_c4 = get_imgL(cam='c4',img_type='Screenshots')
    
    dict_m = {'c0':add_lm_c0,'c1':add_lm_c1,'c2':add_lm_c2,'c3':add_lm_c3,'c4':add_lm_c4}
    dict_s = {'c0':add_ls_c0,'c1':add_ls_c1,'c2':add_ls_c2,'c3':add_ls_c3,'c4':add_ls_c4}
        
    y_lines_cTV0_3 = [[(780,394),(871,394)],[(824,339),(871,339)]]
    c0 = camera_projection('c0',imgc0,None,ptsc0,x_linesC0,y_linesC0 )
    c1 = camera_projection('c1',imgc1,None,ptsc1,x_linesC1,y_linesC1)
    c4 = camera_projection('c4',imgc4,None,ptsc4,x_linesC4,y_linesC4)
    #c3 = camera_projection('c3',img3,img3_mask,ptsc3,x_linesC3,y_linesC3)
    cTV = camera_projection('cTV',imgTV,None,ptsTV,None,None)
    frame_array_loc = position_framework(c0,c1,c4,cTV,dict_m,dict_s)