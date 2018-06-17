# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 11:45:54 2018

@author: YongchaoDing
"""

import numpy as np
import os
import glob as gb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

READ_IMAGE_SHOW = 0;
LIST_SAMPLE = 1;
IMAGE_SAMPLE = 20;


#
#   P1------------P2
#   |             |
#   |             |
#   |             |
#   P4------------P3
#
class BoundingBox:
    def __init__(self):
        self.Point1 = (0, 0);# x,y
        self.Point2 = (0, 0);# x,y
        self.Point3 = (0, 0);# x,y
        self.Point4 = (0, 0);# x,y
 
class BoundingBoxRect:
    def __init__(self):
        self.PointMin = (0,0);
        self.PointMax = (0,0);
        
class CenterPoint:
    def __init__(self):
        self.x = 0;
        self.y = 0;

class Velocity:
    def __init__(self):
        self.x = 0;
        self.y = 0;

def datasetListLoad(listFile):
    listName = [];
    try:
        fd = open(listFile);
    except IOError:
        print("File Not existed!");
        return listName;
    
    line = fd.readline();
    while(line):
        line=line.strip('\n')
        #print(line);
        listName.append(line);
        line = fd.readline();
    return listName;

def imageListLoad(imagePath):
    os.chdir(imagePath);
    #print(os.getcwd());
    imageList = [];
    imageNames = gb.glob("*.jpg");
    for image in imageNames:#[:IMAGE_SAMPLE]:
        img = mpimg.imread(image);
        imageList.append(img);
    os.chdir("../");        ## BACK TO vot2017
    if(READ_IMAGE_SHOW):
        plt.imshow(img);
        plt.axis('off');
        plt.show();
    return imageList;

def boundingBoxLoad(imagePath):
    boundingBoxList = [];
    groundTruthFile = imagePath + "/groundtruth.txt";
    fd = open(groundTruthFile);
    line = fd.readline();
    while(line):    
        line=line.strip('\n')
        #print(line);
        splitLine = line.split(','); 
        boundingBox = BoundingBox();
        #print(splitLine);
        boundingBox.Point1 = (round(float(splitLine[0])), round(float(splitLine[1])));
        boundingBox.Point2 = (round(float(splitLine[2])), round(float(splitLine[3])));
        boundingBox.Point3 = (round(float(splitLine[4])), round(float(splitLine[5])));
        boundingBox.Point4 = (round(float(splitLine[6])), round(float(splitLine[7])));
        #print(boundingBox);
        boundingBoxList.append(boundingBox);
        line = fd.readline();
    if(0):
        print(boundingBoxList[0].Point1);
        print(boundingBoxList[0].Point2);
        print(boundingBoxList[0].Point3);
        print(boundingBoxList[0].Point4);
    return boundingBoxList;

def drawBoundingBox(image, boundingBox):
    BULE = (0, 0, 255);
    print(boundingBox.Point1);
    print(boundingBox.Point2);
    print(boundingBox.Point3);
    print(boundingBox.Point4);
    cv2.line(image, boundingBox.Point1, boundingBox.Point2, BULE, 5)
    cv2.line(image, boundingBox.Point2, boundingBox.Point3, BULE, 5)
    cv2.line(image, boundingBox.Point3, boundingBox.Point4, BULE, 5)
    cv2.line(image, boundingBox.Point4, boundingBox.Point1, BULE, 5)
    #cv2.rectangle(image, (50, 200), (200, 225), red, 5) #15
    plt.imshow(image)
    plt.axis('off');
    plt.show();

def drawBoundingBoxByOPENCV(imageList, boundingBoxList, centerPointList=None):
    RED = (0, 0, 255);
    cv2.namedWindow("Video Show");  
    assert(len(imageList) == len(boundingBoxList));
    for i, image, boundingBox in zip(range(0, len(centerPointList) - 1),imageList, boundingBoxList):
        cv2.line(image, boundingBox.Point1, boundingBox.Point2, RED, 2)
        cv2.line(image, boundingBox.Point2, boundingBox.Point3, RED, 2)
        cv2.line(image, boundingBox.Point3, boundingBox.Point4, RED, 2)
        cv2.line(image, boundingBox.Point4, boundingBox.Point1, RED, 2)
        if i != 0:
            cv2.line(image, (centerPointList[i].x,centerPointList[i].y), (centerPointList[i-1].x,centerPointList[i-1].y), RED, 2);
        #print("Frame:", i);
        cv2.imshow("Video Show", image);
        cv2.waitKey(50);
    cv2.destroyAllWindows();


def motionDetection(boundingBoxList):
    centerPointList = [];
    motionVelocity = [];
    for boundingBox in boundingBoxList:
        centerPoint = CenterPoint();
        centerPoint.x = round((boundingBox.Point1[0]+boundingBox.Point2[0]+boundingBox.Point3[0]+boundingBox.Point4[0])/4);
        centerPoint.y = round((boundingBox.Point1[1]+boundingBox.Point2[1]+boundingBox.Point3[1]+boundingBox.Point4[1])/4);
        #print(centerPoint.x, " ", centerPoint.y);
        centerPointList.append(centerPoint)
    for i in range(0, len(centerPointList) - 1):
        velocity = Velocity();
        if i != 0:
            velocity.x = centerPointList[i].x - centerPointList[i-1].x;
            velocity.y = centerPointList[i].y - centerPointList[i-1].y;
        # print(velocity.x, " ", velocity.y);
        motionVelocity.append(velocity);
    return centerPointList, motionVelocity;
 

def drawMotionVelocity(motionVelocity):
    velocity_x = [];
    velocity_y = [];
    for velocity in motionVelocity:
        velocity_x.append(velocity.x);
        velocity_y.append(velocity.y);
    plt.plot(velocity_x);
    plt.plot(velocity_y);
    plt.show();
    


## Using Feature Point to calculate motion
def featurePointDetectionByORB(ROIList):
    keyPointList = [];
    descriptorList = [];
    # ORB = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE);
    SURF = cv2.xfeatures2d.SURF_create()
    for imageROI in ROIList:
        keyPoint, descriptor = SURF.detectAndCompute(imageROI,None);
        keyPointList.append(keyPoint);
        # print(len(keyPoint));
        descriptorList.append(descriptor);
        #print(len(descriptor));
    return keyPointList, descriptorList;

def drawFeaturePointByOPENCV(imageROIList, keyPointList):
    RED = (0, 0, 255);
    cv2.namedWindow("Feature Point");
    for imageROI, keyPoint in zip(imageROIList, keyPointList):
        # image = cv2.drawKeypoints(imageROI,keyPoint,None,RED,5)
        for Point in keyPoint:
            print(Point.pt[0], " ", Point.pt[1]);
            cv2.circle(imageROI, (int(Point.pt[0]),int(Point.pt[1])), 1, RED, 3)
        print("~~~~~~~~~~~~~~~~~~~");
        cv2.imshow("Feature Point", imageROI);
        cv2.waitKey(100);
    cv2.destroyAllWindows();

def featureMatchByBF(descriptorList):
    matchesList = [];
    BF = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True);
    for i in range(0, len(descriptorList) - 1):
        if i != 0:
            Matches = BF.match(descriptorList[i], descriptorList[i - 1]);
            Matches = sorted(Matches, key = lambda x:x.distance)
        #else:
        #    Matches = BF.match(descriptorList[i], descriptorList[i]);
        #    Matches = sorted(Matches, key = lambda x:x.distance)
        #matchesList.append(Matches);
    return matchesList;
    
def ROIRegionCal(imageList, boundingBoxRectList):
    imageROIList = [];
    for image, boundBoxRect in zip(imageList, boundingBoxRectList):
        imageROI = image[boundBoxRect.PointMin[1]:boundBoxRect.PointMax[1], boundBoxRect.PointMin[0]:boundBoxRect.PointMax[0]];
        imageROI = cv2.cvtColor(imageROI, cv2.COLOR_BGR2GRAY);
        imageROIList.append(imageROI);
    #plt.imshow(imageROI);
    #plt.show();
    return imageROIList;
    
def boundingBoxRectCal(boundingBoxList):
    boundingBoxRectList = [];
    for boundingBox in boundingBoxList:
        boundingBoxRect = BoundingBoxRect();
        boundingBoxRect.PointMin = (min(boundingBox.Point1[0], boundingBox.Point2[0],boundingBox.Point3[0],boundingBox.Point4[0]), min(boundingBox.Point1[1], boundingBox.Point2[1],boundingBox.Point3[1],boundingBox.Point4[1]));
        boundingBoxRect.PointMax = (max(boundingBox.Point1[0], boundingBox.Point2[0],boundingBox.Point3[0],boundingBox.Point4[0]), max(boundingBox.Point1[1], boundingBox.Point2[1],boundingBox.Point3[1],boundingBox.Point4[1]));
        boundingBoxRectList.append(boundingBoxRect);
    # print(boundingBoxRectList[0].PointMin, " ", boundingBoxRectList[0].PointMax);
    return boundingBoxRectList;

def drawRectBoundingBoxByOPENCV(imageList, boundingBoxRectList):
    RED = (0, 0, 255);
    cv2.namedWindow("Video Show (Rect)");
    assert(len(imageList) == len(boundingBoxRectList));
    for image, boundingBoxRect in zip(imageList, boundingBoxRectList):
       cv2.rectangle(image,boundingBoxRect.PointMin,boundingBoxRect.PointMax,RED,3);
       cv2.imshow("Video Show (Rect)", image);
       cv2.waitKey(50);
    cv2.destroyAllWindows();
    
    
if __name__ == "__main__":
    Path = "../../DataSet/vot2017/";
    listFile = Path + "list.txt";
    datasetList = datasetListLoad(listFile);
    for imageName in  datasetList[3:LIST_SAMPLE+3]:  #3
        #imageName = 
        imagePath = os.path.join(Path, imageName);
        print(imagePath);
        imageList = imageListLoad(imagePath);
        boundingBoxList = boundingBoxLoad(imagePath);
        centerPointList, motionVelocity = motionDetection(boundingBoxList);
        
        boundingBoxRectList = boundingBoxRectCal(boundingBoxList);
        imageROIList = ROIRegionCal(imageList, boundingBoxRectList);
        keyPointList, descriptorList = featurePointDetectionByORB(imageROIList);
        # matchesList = featureMatchByBF(descriptorList);
        drawFeaturePointByOPENCV(imageROIList, keyPointList);
        #drawRectBoundingBoxByOPENCV(imageList, boundingBoxRectList);
        #drawMotionVelocity(motionVelocity);
        #drawBoundingBox(imageList[0], boundingBoxList[0]);
        #drawBoundingBoxByOPENCV(imageList, boundingBoxList, centerPointList);
        

        
        