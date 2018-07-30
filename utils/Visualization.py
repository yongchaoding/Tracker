import matplotlib.pyplot as plt
import logging
import tensorflow as tf
import matplotlib.patches as patches
import cv2
import math

def visualization_boundingboxs(image, boundboxs):
    Red = (255, 0, 0)   # RGB
    for boundbox in boundboxs:
        cv2.rectangle(image, (int(math.floor(boundbox[0])),int(math.floor( boundbox[1]))), (int(math.floor(boundbox[0]+boundbox[2])), int(math.floor(boundbox[1]+boundbox[3]))), Red, 5)
    height, width, channel = image.shape;
    logging.info('Height:%d, Width:%d', height, width)
    plt.figure(figsize=(20, 20));
    plt.imshow(image)

