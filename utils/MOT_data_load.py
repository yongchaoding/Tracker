from __future__ import division
import tensorflow as tf
import logging
import cv2
import glob as gb
import math

logging.basicConfig(level=logging.INFO)

def ImageLoad(ImagePath):
    imgList = []
    gbPath = ImagePath + '/*.jpg'
    imgPaths = gb.glob(gbPath)
    imgPaths.sort()
    #logging.info('Image: %s', imgList)
    logging.debug('Num of Image: %d', len(imgList))
    for imgName in imgPaths:
        #logging.info('Image: %s', imgName)
        img = cv2.imread(imgName);
        imgList.append(img);
    Num = len(imgList);
    logging.info('Read %d images from %s', Num, ImagePath);
    return imgList;

def DetLoad(DetPath):
    DetList = []
    f = open(DetPath);
    frameStart = 1;
    frameDet = [];
    while True:
        line = f.readline();
        if not line:
            DetList.append(frameDet);
            break;
        infoList = line.split(',');
        if frameStart == int(infoList[0]):
            frameDet.append([math.floor(float(infoList[2])), math.floor(float(infoList[3])), math.floor(float(infoList[4])), math.floor(float(infoList[5])), float(infoList[6])])
        else:
            frameStart += 1;
            DetList.append(frameDet);
            frameDet = [];
            frameDet.append([math.floor(float(infoList[2])), math.floor(float(infoList[3])), math.floor(float(infoList[4])), math.floor(float(infoList[5])), float(infoList[6])])
    logging.debug('Frame detection: %s', DetList[0]);
    logging.info('Read %d frame detection', len(DetList));
    return DetList;

def GtLoad(GtPath):
    GtList = []
    f = open(GtPath)
    trackletNum = 1
    tracklet = []
    while True:
        line = f.readline();
        if not line:
            GtList.append(tracklet)
            break;
        info = line.split(',');
        if trackletNum == int(info[1]):
            tracklet.append([int(info[0]), int(info[2]), int(info[3]), int(info[4]), int(info[5])])
        else:
            trackletNum += 1;
            GtList.append(tracklet)
            tracklet = []
            tracklet.append([int(info[0]), int(info[2]), int(info[3]), int(info[4]), int(info[5])])
    logging.debug('Tracklet: %s', GtList[0]);
    logging.info('Read %d tracklet', len(GtList));
    return GtList;

def MOT17DataLoad(data_dir):
    DataPath = data_dir
    # For MOT17
    DataImagePath = DataPath + '/img1';
    DataDetPath = DataPath + '/det/det.txt';
    DataGtPath = DataPath + '/gt/gt.txt';
    logging.info('ImagePath: %s, DetPath: %s, GtPath: %s',DataImagePath, DataDetPath, DataGtPath);
    DataImage = ImageLoad(DataImagePath);
    DataDet = DetLoad(DataDetPath);
    DataGtPath = GtLoad(DataGtPath);

    return DataImage, DataDet,DataGtPath;


#def main(_):
#    flags.mark_flag_as_required('data_dir')
#    DataPath = FLAGS.data_dir
#    # For MOT17
#    DataImagePath = DataPath + '/img1';
#    DataDetPath = DataPath + '/det/det.txt';
#    DataGtPath = DataPath + '/gt/gt.txt';
#    logging.info('ImagePath: %s, DetPath: %s, GtPath: %s',DataImagePath, DataDetPath, DataGtPath);
#    DataImage = ImageLoad(DataImagePath);
#    DataDet = DetLoad(DataDetPath);
#    DataGtPath = GtLoad(DataGtPath);
#
#    return DataImage, DataDet,DataGtPath;

