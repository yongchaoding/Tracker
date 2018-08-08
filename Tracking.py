import cv2
from utils import MOT_data_load
from utils import Visualization
from utils import costFunction

import matplotlib.pyplot as plt

def main():
  Images, DataDet, GT = MOT_data_load.MOT17DataLoad("Data/MOT17-09")
  for image,BBs in zip(Images,DataDet):
      #print(BBs)
      tracklet = costFunction.CostFunctionCalc(image, BBs)
      print(tracklet[0].Boundbox[0])
      Visualization.visualization_boundingboxs(image, BBs)
      plt.show()
if __name__ == "__main__":
    main();
