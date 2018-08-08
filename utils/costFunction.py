import cv2
import math

FirstFrame = 1;
# def struct of pos:[x_center, y_center, x_width, y_height]
# pos need control in [0, 1], which 1 means similarity
def posCostFunction(pos1, pos2):
    # pos1 is latest pos, and pos2 is old position offset by motion
    x_offset = math.fabs(pos1[0] - pos2[0]) / pos2[2];
    y_offset = math.fabs(pos1[1] - pos2[1]) / pos2[3];
    offset = (x_offset + y_offset) / 2;
    posCost = math.exp(-offset);
    return posCost;

def detectionCostFunction(accuracy):
    return accuracy;

def BoundboxSizeCostFunction(Boundbox1, Boundbox2):
    # Boundbox1 is latest bb, and Boundbox2 is old position offset by distance
    width1 = Boundbox1[2];
    height1 = Boundbox1[3];

    width2 = Boundbox2[2];
    height2 = Boundbox2[3];

    widthOffset = math.fabs(width1 - width2) / min(width1, width2);
    heightOffset = math.fabs(height1 - height2) / min(height1, height2);

    offset = (widthOffset + heightOffset) / 2;
    sizeCost = math.exp(-offset);
    return sizeCost;

channels =[0, 1]
h_bins = 256;
s_bins = 256;
bins = [h_bins, s_bins];
ranged = [0,h_bins, 0, s_bins];

def RGBHistCostFunction(image1, image2):
    imageHSV1 = [];
    imageHSV2 = [];
    imageHSV1 = cv2.cvtColor(image1,  cv2.COLOR_BGR2HSV);
    imageHSV2 = cv2.cvtColor(image2,  cv2.COLOR_BGR2HSV);
    Hist1 = cv2.calcHist([imageHSV1], channels, None, bins, ranged);    
    cv2.normalize(Hist1, Hist1, 0, 1, cv2.NORM_MINMAX);
    Hist2 = cv2.calcHist([imageHSV2], channels, None, bins, ranged);    
    cv2.normalize(Hist2, Hist2, 0, 1, cv2.NORM_MINMAX);
    RGBHistCost = cv2.compareHist(Hist1, Hist2, cv2.HISTCMP_CORREL);
    return RGBHistCost;

def posCalc(Boundingbox):
    x_center = Boundingbox[0] + Boundingbox[2] / 2;
    y_center = Boundingbox[1] + Boundingbox[3] / 2;
    x_width = Boundingbox[2];
    y_height = Boundingbox[3];
    return [x_center, y_center, x_width, y_height];

def imageROI(image, ROIboundbox):
    #print(int(ROIboundbox[2]))
    #print(int(ROIboundbox[0]))
    #print(int(ROIboundbox[3]))
    #print(int(ROIboundbox[1]))
    if ROIboundbox[0] < 0:
        ROIboundbox[0] = 0;
    if ROIboundbox[1] < 0:
        ROIboundbox[1] = 0;
    imageROI = image[int(ROIboundbox[1]):int(ROIboundbox[1]+int(ROIboundbox[3])), int(ROIboundbox[0]):int(ROIboundbox[0])+int(ROIboundbox[2])];
    #cv2.imshow('image ROI', imageROI);
    return imageROI;

def TrackletAddDetection(tracklet, boundbox):
    tempboundbox = tracklet.Boundbox;
    print(tempboundbox)
    tempboundbox.append(boundbox);
    print(tempboundbox)
    tracklet.Boundbox = tempboundbox;
    return tracklet;

class tracklet:
    def __init__(self):
        tracklet.ID = 0;
        tracklet.Boundbox = [];
        tracklet.PosTransform = [0, 0];
        tracklet.SizeTransform = [0, 0];
        tracklet.Status = 0;    # 0 for active, 1 for tracked, 2 for losted, 3 for inactive

ID_Generate = 1;
Tracklets = [];
imageOrigin = [];

def CostFunctionCalc(image, boundboxs):
    global Tracklets
    global FirstFrame
    global ID_Generate
    global imageOrigin
    if FirstFrame == 1:
        imageOrigin = image;
        boundboxOrigin = boundboxs;
        FirstFrame = 0;
        for boundbox in boundboxs:
            Tracklet = tracklet();
            Tracklet.ID = ID_Generate;
            TrackletAddDetection(Tracklet, boundbox[0:4])
            #(Tracklet.Boundbox).append(boundbox[0:4]);
            #(Tracklet.Boundbox) = [boundbox[0:4]];
            Tracklet.Status = 0;
            Tracklets.append(Tracklet);
            ID_Generate += 1;
    else:

        for Tracklet in Tracklets:
            if Tracklet.Status == 3:
                continue;
            for detectionNum, detection in enumerate(boundboxs):
                
                accuracy = detection[4]
                accuracyCost = detectionCostFunction(accuracy);
                
                posOrigin = posCalc(Tracklet.Boundbox[-1]);
                posLatest = posCalc(detection[0:4])
                posCost = posCostFunction(posLatest, posOrigin);

                sizeCost = BoundboxSizeCostFunction(detection[0:4], Tracklet.Boundbox[-1])

                TrackletImageROI = imageROI(imageOrigin, Tracklet.Boundbox[-1]);
                DetectionImageROI = imageROI(image, detection[0:4]);

                rgbHistCost = RGBHistCostFunction(DetectionImageROI, TrackletImageROI);

                print('ID: ', Tracklet.ID, 'Num: ', detectionNum, ' posCost', posCost, 'Size Cost: ', sizeCost, 'RGBHist Cost:', rgbHistCost);

    return Tracklets;
