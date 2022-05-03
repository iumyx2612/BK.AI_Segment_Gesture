import cv2
import os
import numpy as np
import tqdm

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog
coco_metadata = MetadataCatalog.get("coco_2017_val")

# import PointRend project
from detectron2.projects import point_rend


def generate_masks(prediction):
    mask_pred = prediction["instances"].pred_masks.cpu().numpy()
    label_pred = prediction["instances"].pred_classes.cpu().numpy()
    score_pred = prediction["instances"].scores.cpu().numpy()
    
    for label, mask, score in zip(label_pred, mask_pred, score_pred):
        if label == 0 and score > 0.8:
            mask_pred = mask.astype(np.int8)*255
            return mask_pred
        else:
            continue


def main():
    dir_video = '../peekaboo'
    dir_save = '../peekaboo_mask'
    if not os.path.exists(dir_save):
    	os.makedirs(dir_save)
    
    # load model
    cfg = get_cfg()
    # Add PointRend-specific config
    point_rend.add_pointrend_config(cfg)
    # Load a config from file
    cfg.merge_from_file("detectron2_repo/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
    # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
    cfg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_edd263.pkl"
    cfg.MODEL.DEVICE = 'cuda'
    predictor = DefaultPredictor(cfg)
    
    for vid in tqdm.tqdm(os.listdir(dir_video)):
        name_video = vid.split(".")[0] + "_mask.avi"
        
        cap = cv2.VideoCapture(os.path.join(dir_video, vid))
        if (cap.isOpened()== False):
            print("Error opening video stream or file")
        
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))
        
        out = cv2.VideoWriter(os.path.join(dir_save, name_video), cv2.VideoWriter_fourcc('M','J','P','G'), int(cap.get(cv2.CAP_PROP_FPS)), (frame_width,frame_height))
        
		i = 0
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
				i += 1
                out_image = np.zeros(frame.shape)
				try:
                	outputs = predictor(frame)
                	mask = generate_masks(outputs) / 255
                	out_image[:, :, 0] = frame[:, :, 0] ** mask
                	out_image[:, :, 1] = frame[:, :, 1] ** mask
                	out_image[:, :, 2] = frame[:, :, 2] ** mask
                	out_image = np.uint8(out_image)
                	out.write(out_image)
                except:
                	print(f"{vid} o frame thu {i}")
                	continue
            else:
                break
                
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
    