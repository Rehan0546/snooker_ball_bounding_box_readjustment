# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 16:55:13 2023

@author: rehan
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt


def ball_detection(img,class_):
    thresh1 = cv2.GaussianBlur(img, (7,7), 0)
    # ret,thresh1 = cv2.threshold(img,105,255,cv2.THRESH_BINARY)
    plt.imshow(thresh1)
    hsv = cv2.cvtColor(thresh1, cv2.COLOR_BGR2HSV)
    plt.imshow(thresh1)
    plt.title('hsv')
    if class_ == 'green':
        lower_red = np.array([36,0,0])
        upper_red = np.array([86,255,255])
    elif class_ == 'red':
        
        lower_red = np.array([160,50,50])
        upper_red = np.array([180,255,255])
    elif class_ == 'yellow':
        lower_red = np.array([22, 93, 0])
        upper_red = np.array([45, 255, 255])
    elif class_ == 'white':
        lower_red = np.array([37, 0, 131])
        upper_red = np.array([170, 25, 152])
    elif class_ == 'brown':
        lower_red = np.array([6, 63, 0])
        upper_red = np.array([23, 255, 81])
    elif class_ =='blue':
        lower_red = np.array([94, 80, 2])
        upper_red = np.array([120, 255, 255])
    elif class_ == 'pink':
        lower_red = np.array([233, 88, 233])
        upper_red = np.array([241, 82, 240])
    
    green_mask = cv2.inRange(hsv, lower_red, upper_red)
    output = cv2.bitwise_and(img, img, mask = green_mask)
    plt.imshow(output)
    plt.title('sss')
    thresh1_inverted = cv2.cvtColor(thresh1, cv2.COLOR_BGR2GRAY)
    green_mask_inverted = ~green_mask
    
    shadow = cv2.bitwise_and(green_mask_inverted, thresh1_inverted, mask = None)

    
    # kernel = np.ones((7,7),np.uint8)
    # dilation = cv2.dilate(shadow,kernel,iterations = 1)
    thres = 1
    dilation = shadow.copy()
    dilation[dilation<thres] = 0
    dilation[dilation>=thres] = 1

    y_min = None
    y_max = None
    x_min = None
    x_max = None
    
    for i in range(dilation.shape[0]):
        data = dilation[i]
        # print(data)
        if 1 in data and y_min==None:
            y_min = i
        if 1 in data:
            y_max = i
    
    dilation_t = dilation.transpose()
    
    for i in range(dilation_t.shape[0]):
        data = dilation_t[i]
        # print(data)
        if 1 in data and x_min==None:
            x_min = i
        if 1 in data:
            x_max = i
    if x_min<0:
        x_min=0
    if y_min<0:
        y_min=0
    return x_min,y_min,x_max,y_max


img = cv2.imread('red2.jpg')
plt.imshow(img,cmap='gray')
x_min,y_min,x_max,y_max =ball_detection(img,'red')
cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255,255,255))

plt.figure()
plt.imshow(img[:,:,::-1])
