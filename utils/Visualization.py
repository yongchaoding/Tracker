import matplotlib.pyplot as plt
import logging
import tensorflow as tf
import matplotlib.patches as patches
import cv2
import math

def visualization_boundingboxs(imageOrigin, boundboxs, color=(255,0,0)):
    image = imageOrigin.copy();
    
    for Num, boundbox in enumerate(boundboxs):
        cv2.rectangle(image, (int(math.floor(boundbox[0])),int(math.floor( boundbox[1]))), (int(math.floor(boundbox[0]+boundbox[2])), int(math.floor(boundbox[1]+boundbox[3]))), color, 5)
        cv2.putText(image, str(Num),(int(math.floor(boundbox[0])), int(math.floor(boundbox[1]))),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
    
    #height, width, channel = image.shape;
    #logging.info('Height:%d, Width:%d', height, width)
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5);
    cv2.imshow('image',image)
    cv2.waitKey()
    #plt.figure(figsize=(20, 20));
    #plt.imshow(image)


def visualization_tracklet(imageInput, tracklets):
    image = imageInput.copy();
    Red = (0, 255, 0);
    for tracklet in tracklets:
        cv2.rectangle(image, (int(tracklet.Boundbox[-1][0]),int(tracklet.Boundbox[-1][1])), (int(tracklet.Boundbox[-1][0]+tracklet.Boundbox[-1][2]), int(tracklet.Boundbox[-1][1]+tracklet.Boundbox[-1][3])), Red, 5)
        cv2.putText(image, str(tracklet.ID),(int(tracklet.Boundbox[-1][0]), int(tracklet.Boundbox[-1][1])),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, Red, 2) 
    #image = cv2.resize(image, (0,0), fx=0.5, fy=0.5);
    #cv2.imshow('image',image)
    #cv2.waitKey()
    return image;
