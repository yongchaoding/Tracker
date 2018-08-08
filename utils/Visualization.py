import matplotlib.pyplot as plt
import logging
import tensorflow as tf
import matplotlib.patches as patches
import cv2
import math

def visualization_boundingboxs(imageOrigin, boundboxs):
    image = imageOrigin.copy();
    Red = (255, 0, 0)   # RGB
    for boundbox in boundboxs:
        cv2.rectangle(image, (int(math.floor(boundbox[0])),int(math.floor( boundbox[1]))), (int(math.floor(boundbox[0]+boundbox[2])), int(math.floor(boundbox[1]+boundbox[3]))), Red, 5)
        cv2.putText(image, str(boundbox[4]),(int(math.floor(boundbox[0])), int(math.floor(boundbox[1]))),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
    height, width, channel = image.shape;
    logging.info('Height:%d, Width:%d', height, width)
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5);
    cv2.imshow('image',image)
    cv2.waitKey()
    #plt.figure(figsize=(20, 20));
    #plt.imshow(image)

