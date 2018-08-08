import cv2
from utils import MOT_data_load
from utils import Visualization
from utils import costFunction

import matplotlib.pyplot as plt

def main():
  Images, DataDet, GT = MOT_data_load.MOT17DataLoad("Data/MOT17-09")
  for image,BBs in zip(Images,DataDet):
      print(BBs)
      tracklets = costFunction.CostFunctionCalc(image, BBs)
      print(tracklets[0].ID)
      print(tracklets[0].Boundbox)
      print(tracklets[1].ID)
      print(tracklets[1].Boundbox)
      ImageTracklet = Visualization.visualization_tracklet(image, tracklets)
      Visualization.visualization_boundingboxs(ImageTracklet, BBs)
if __name__ == "__main__":
    main();
