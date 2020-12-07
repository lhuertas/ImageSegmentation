import wget
import os
import matplotlib.pyplot as plt
import cv2
from optparse import OptionParser
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2 import model_zoo
import time
from PIL import Image

def inference(cfg, image):
  predictor = DefaultPredictor(cfg)
  outputs = predictor(image)
  v = Visualizer(image[:, :, ::-1], MetadataCatalog.get("train"), scale=1.2)
  out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
  #cv2.imwrite( outputfile, out.get_image()[:, :, ::-1] )
  img = out.get_image()[:, :, ::-1]
  return img

def get_image():

  parser = OptionParser()
  parser.add_option("-u", "--url", dest="imageURL",
                    help="image url", metavar="FILE")

  (options, args) = parser.parse_args()


  if (options.imageURL == None):
     print (parser.usage)
     exit(0)
  else:
     imageURL = options.imageURL

  image_file = wget.download(imageURL)

  print(" ")
  print("Image downloaded")
  print(" ")

  im = cv2.imread(image_file)

  #configuration from trained model
  cfg = get_cfg()
  cfg.MODEL.DEVICE='cpu'
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
  cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3
  os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

  #get the prediction
  pred = inference(cfg, im)

  #saving prediction
  outputfile = "f{}_{}__{}.jpg".format("Huertas","Lina",time.strftime("%H:%M:%S", time.localtime()))
  cv2.imwrite(cfg.OUTPUT_DIR + "/" + outputfile, pred)
  print("Image {} created".format(cfg.OUTPUT_DIR + "/" + outputfile))

if __name__ == '__main__':
  
  get_image() 
