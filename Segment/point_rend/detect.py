# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import torch
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend


def generate_masks(prediction, name_image, dir_out):
    mask_pred = prediction["instances"].pred_masks.cpu().numpy()
    label_pred = prediction["instances"].pred_classes.cpu().numpy()
    score_pred = prediction["instances"].scores.cpu().numpy()
    
    for label, mask, score in zip(label_pred, mask_pred, score_pred):
        if label == 0 and score > 0.8:
            mask_pred = mask.astype(np.int8)*255
            cv2.imwrite(dir_out + name_image + ".png", mask_pred)
        else:
            continue
        


if __name__ == "__main__":

    dir_image = "../public_test_segment_data/image"
    dir_out = "../mask_generate_by_pointrend/"
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    cfg = get_cfg()
    # Add PointRend-specific config
    point_rend.add_pointrend_config(cfg)
    # Load a config from file
    cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
    print(cfg.DEVICE)
    exit()
    predictor = DefaultPredictor(cfg)
        
    for image in os.listdir(dir_image):
        print(image)
        name = image.split(".")[0]
        dir_ = os.path.join(dir_image, image)
        img = cv2.imread(dir_)
        
        outputs = predictor(img)
        generate_masks(outputs, name, dir_out)