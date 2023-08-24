#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:44:21 2022

@author: Taru
"""

################################# Imports #####################################
import cv2
import os, time
import numpy as np
# Camera Imports
from pypylon import pylon
from pypylon_opencv_viewer import BaslerOpenCVViewer
# SLM Imports
import detect_heds_module_path
from holoeye import slmdisplaysdk
from showSLMPreview import showSLMPreview

############################## Method Definitions #############################           
def imageInit(imgToCheck):
    
    # CONVERTING IMAGE TO BINARY
    imgToCheck = cv2.cvtColor(imgToCheck, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("1: Binary Image", imgToCheck)
    
    # ADAPTIVE THRESHOLDING
    #imgToCheck = cv2.adaptiveThreshold(imgToCheck, 255,
           #cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
    imgToCheck = cv2.adaptiveThreshold(imgToCheck, 255,
                                       cv2.ADAPTIVE_THRESH_MEAN_C, 
                                       cv2.THRESH_BINARY_INV, 21, 10)
    #(T, imgToCheck) = cv2.threshold(imgToCheck, 0, 255, 
           #cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    #cv2.imshow("2: Threshold", imgToCheck)
    
    # FINDING CONTOURS
    cnts = cv2.findContours(imgToCheck, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnts) == 2):
        cnts = cnts[0] 
    else: 
        cnts = cnts[1]
  
    # FILLING IN RECTANGULAR CONTOURS - this needs to be better
    for c in cnts:
        cv2.drawContours(imgToCheck, [c], -1, (255,255,255), -1)
    #cv2.imshow("3: Filled Rectangular Contours", imgToCheck)
   
    # MORPH OPEN
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    imgToCheck = cv2.morphologyEx(imgToCheck, cv2.MORPH_OPEN, kernel, 
                                  iterations=4)
    #cv2.imshow("4: Morph Open", imgToCheck)
    
    return imgToCheck
    
def PVIdentification(imgToCheck, p, q, r, s):
    imgToCheckInit = imageInit(imgToCheck)
    
    # INITIALISING BLACK IMAGE FOR SLM 
    blackBG = cv2.imread("black.tif")         
    dim = (1920, 1080) # making same width and height as ImageTo Check  
    blackBG = cv2.resize(blackBG, dim, interpolation = cv2.INTER_AREA)
    
    # FINDING CONTOURS
    cnts = cv2.findContours(imgToCheckInit, cv2.RETR_EXTERNAL, 
                            cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnts) == 2):
        cnts = cnts[0] 
    else: 
        cnts = cnts[1]
        
    for c in cnts:
        # COMPUTING CENTER OF CONTOURS
        M = cv2.moments(c)
        if (M["m00"] == 0):
            M["m00"]=1
        if (M["m10"] == 0):
            M["m10"]=1
        if (M["m01"] == 0):
            M["m01"]=1
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        
        c = c.astype("float")
        c = c.astype("int")
        
        # DRAWING CONTOURS
        
        if (len(c) <= 100): 
        # (to avoid bubbles - they tend to have contour numbers > 100)
            cv2.drawContours(imgToCheck, [c], -1, (0, 255, 0), 3)
            # PLACING CIRCLE IN CENTER OF CONTOUR
            cv2.circle(imgToCheck, (cX, cY), 2, (0, 0, 255), -1)
            
            # NUMBER OF CONTOURS
            cv2.putText(imgToCheck, str(len(c)), (cX, cY), 
                        cv2.FONT_HERSHEY_SIMPLEX,
        		0.5, (255, 255, 255), 2)
            
            # Doing lookup with a tolerance test
            x = int(p*cX + q*cY + b)
            y = int(r*cY + s*cY + d)
            print(x, y)
            # CREATING IMAGE FOR SLM
            cv2.circle(blackBG, (x, y), 10, (255, 255, 255), -1)
            print(x, y)
            
    cv2.imshow('5: Annotated Robots Image', imgToCheck)
    cv2.imshow('6: SLM Image', blackBG)
    cv2.imwrite("SLMImage.png", blackBG) # Image to be loaded onto SLM

def findingContours(imgToCheck):
    # Assume that the loaded image is already greyscaled and thresholded

    # FINDING CONTOURS
    cnts = cv2.findContours(imgToCheck, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if (len(cnts) == 2):
        cnts = cnts[0] 
    else: 
        cnts = cnts[1]
    
    return cnts

start = time.time()

############################## Initialising SLM ###############################           
# INITIALIZING SLM LIBRARY
slm = slmdisplaysdk.SLMInstance()
if not slm.requiresVersion(3): # checking library version
     exit(1)
  
# DETECT SLM AND OPEN WINDOW
error = slm.open()
assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)

# OPEN SLM IN PREVIEW WINDOW IN NON-SCALED MODE
showSLMPreview(slm, scale=0.0)

########################### Initialising Camera ###############################           
# CREATING AND OPENING CAMERA OBJECT
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
print("Using device ", camera.GetDeviceInfo().GetModelName())
viewer = BaslerOpenCVViewer(camera)

# SETTING PARAMETERS
countOfImagesToGrab = 100
camera.MaxNumBuffer = 5

########################## Alignment Initialisation ###########################
# IMAGE BEFORE SLM DISPLAY

threshVal = 20

# Grabbing image before SLM display, from camera
cameraImgBeforeSlm = viewer.get_image()
cv2.imshow("cameraImgBeforeSlm", cameraImgBeforeSlm)

# Applying Polygon Mask
mask = np.zeros(cameraImgBeforeSlm.shape[:2], dtype="uint8")
pts = np.array([[1690,18],[1837,856],[276,1156],[70,360]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.fillPoly(mask,[pts], (255, 255, 255))
cameraImgBeforeSlm = cv2.bitwise_and(cameraImgBeforeSlm, cameraImgBeforeSlm, 
                                     mask=mask)
# cv2.imshow("Mask Applied to cameraImgBeforeSlm", cameraImgBeforeSlm)

# Convert to grayscale
cameraImgBeforeSlm = cv2.cvtColor(cameraImgBeforeSlm, cv2.COLOR_BGR2GRAY) 
# Thresholding
ret, cameraImgBeforeSlm = cv2.threshold(cameraImgBeforeSlm,threshVal,255,
                                        cv2.THRESH_BINARY) 
# Saving, reading
cv2.imwrite("cameraImgBeforeSlm.tif", cameraImgBeforeSlm)
cameraImgBeforeSlm = cv2.imread("cameraImgBeforeSlm.tif")      
cv2.imshow("cameraImgBeforeSlm processed", cameraImgBeforeSlm)

# FINDING PARAMETERS FOR MATRICES METHOD

# Test points - input array

testInputList = [[800, 500], [500, 800], [1600, 650]]
testOutputList = [[0, 0, 1], [0, 0, 1], [0, 0, 1]]
testOutputList = np.array(testOutputList)   
testInputList = np.array(testInputList)   

# Test points - output array
for loopNumber in range(0, 3):
    # Creating image for SLM
    # Loading and resizing black image
    imgToLoad = cv2.imread("black.tif")         
    dim = (1920, 1080)
    imgToLoad = cv2.resize(imgToLoad, dim, interpolation = cv2.INTER_AREA)
    print(testInputList[loopNumber][0], testInputList[loopNumber][1])
    cv2.circle(imgToLoad,(testInputList[loopNumber][0], testInputList[loopNumber][1]), 25, (255,255,255), -1)
    cv2.imshow("imgToLoad", imgToLoad)
    cv2.imwrite("imgToLoad.tif", imgToLoad)
    
    # Loading Image onto SLM
    thisScriptPath = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(thisScriptPath, "imgToLoad.tif")
    error = slm.showDataFromFile(filename, slmdisplaysdk.ShowFlags.PresentAutomatic)
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
    # Accounting for SLM lag
    time.sleep(0.5)
    
    # IMAGE AFTER SLM DISPLAY
        
    # Grabbing image from camera
    imgToCheck = viewer.get_image()
    OGimgToCheck =imgToCheck 
    cv2.imshow("Grabbed Image", imgToCheck)
        
    # Applying Polygon Mask
    mask = np.zeros(imgToCheck.shape[:2], dtype="uint8")
    pts = np.array([[1876,318],[1490,1090],[7,866],[5,360]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.fillPoly(mask,[pts], (255, 255, 255))
    cv2.imshow("Mask", mask)
    imgToCheck = cv2.bitwise_and(imgToCheck, imgToCheck, mask=mask)
    # cv2.imshow("Mask Applied to imgToCheck", imgToCheck)
        
    # Convert to grayscale
    imgToCheck = cv2.cvtColor(imgToCheck, cv2.COLOR_BGR2GRAY) 
    # Thresholding
    ret, imgToCheck = cv2.threshold(imgToCheck,threshVal,255,cv2.THRESH_BINARY) 
    # Saving, reading
    cv2.imwrite("cameraImgAfterSlm.tif", imgToCheck)
    imgToCheck = cv2.imread("cameraImgAfterSlm.tif")
    cv2.imshow("Grabbed Image processed", imgToCheck)
        
    # Applying Exclusive OR Mask
    imgToCheck = cv2.bitwise_xor(imgToCheck, cameraImgBeforeSlm)
    cv2.imshow("Mask Applied to Image", imgToCheck)
    # Convert to grayscale
    imgToCheck = cv2.cvtColor(imgToCheck, cv2.COLOR_BGR2GRAY) 
    # Thresholding
    ret, imgToCheck = cv2.threshold(imgToCheck,threshVal,255,cv2.THRESH_BINARY)
    cv2.imshow("Mask Applied to Image2", imgToCheck)        
        
    # Finding coordinates of dot
    refImage = cv2.imread("spot1.2.png")
    refImage = cv2.cvtColor(refImage, cv2.COLOR_BGR2GRAY) # convert to grayscale
    ret, refImage = cv2.threshold(refImage,threshVal,255,cv2.THRESH_BINARY)
    refImageCnts = findingContours(refImage)
    imgToCheckCnts = findingContours(imgToCheck)
        
    totalcX = 0
    totalcY = 0
    count = 0
    for c in imgToCheckCnts:
        # COMPUTING CENTER OF CONTOURS
        M = cv2.moments(c)
        if (M["m00"] == 0):
            M["m00"]=1
        if (M["m10"] == 0):
            M["m10"]=1
        if (M["m01"] == 0):
            M["m01"]=1
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
            
        c = c.astype("float")
        c = c.astype("int")    
            
        # CONTOUR MATCHING
        for i in refImageCnts:
            match = cv2.matchShapes(i, c, 1, 0.0)
            if (match < 10):
                count += 1
                totalcX += cX
                totalcY += cY
                print(cX, cY)
        
    # FINDING AVERAGE CAMERA COOORDINATES 
    if (count == 0):
        avecX = 0
        avecY = 0
    else:       
        avecX = int(totalcX / count)
        avecY = int(totalcY / count)
    print("ave", avecX, avecY)
    cv2.circle(OGimgToCheck, (avecX, avecY), 50, (255, 0, 0), 5)
    # Output array inputting
    print(loopNumber)
    testOutputList[loopNumber][0] = avecX
    testOutputList[loopNumber][1] = avecY
    cv2.imshow("Contourmarked", OGimgToCheck)

# Solving for parameters
print(testOutputList)
# x values
testInputListX = [testInputList[0][0], testInputList[1][0], testInputList[2][0]]
A = testOutputList
B = testInputListX   
X = np.linalg.inv(A).dot(B)
p = X[0]
q = X[1]
b = X[2]

# y values
testInputListY = [testInputList[0][1], testInputList[1][1], testInputList[2][1]]
C = testOutputList
D = testInputListY 
Y = np.linalg.inv(C).dot(D)
r = Y[0]
s = Y[1]
d = Y[2]

############################## Closed Loop ####################################           
print("Switch on frontlight!")
time.sleep(10)
num = 0 # number of times we want the loop to run

while num < 100:
    # STARTING CONTINOUS IMAGE ACQUISITION
    # Save image
    viewer.save_image('~C:/Users/MiskinLab/.spyder-py3/grabbedImg.tif')
            
    # Get grabbed image
    imgToCheck = viewer.get_image()
    # imgToCheck = cv2.imread("lookupTest.png")
    PVIdentification(imgToCheck, p, q, r, s)
    
    # To send the pic to the SLM within the while loop
    # Show image file data on SLM:
    thisScriptPath = os.path.dirname(os.path.abspath(__file__))
    print(thisScriptPath)
    filename = os.path.join(thisScriptPath, "SLMImage.png")
    error = slm.showDataFromFile(filename, 
                                 slmdisplaysdk.ShowFlags.PresentAutomatic)
    assert error == slmdisplaysdk.ErrorCode.NoError, slm.errorString(error)
    num += 1
    
    time.sleep(0.5)

###############################################################################

end = time.time()
print(end - start)

###############################################################################
