

import detectron2
from detectron2.utils.logger import setup_logger
import argparse

setup_logger()
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, VisImage
from detectron2.data import MetadataCatalog, DatasetCatalog
from matplotlib import pyplot as plt
import glob
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.solver import build_lr_scheduler
from vis import *
from dataset_bjtu import *

import torch
from detectron2.engine import DefaultTrainer
from eval_func import *

print(torch.cuda.is_available())
print(torch.cuda.get_device_capability())
os.environ["CUDA_VISIBLE_DEVICES"] = "1,"




parser = argparse.ArgumentParser(description="Detectron2 Training Script")
parser.add_argument('--cfg', default="config/hit.yaml", help='Path to the config file')
parser.add_argument('--weight_dir', required=True, help='Path to the weight directory')
parser.add_argument('--image_path', required=True, help='Path to the iamge')


args = parser.parse_args()

CFG = args.cfg
CHECK_DIR = args.weight_dir
IMG_PATH = args.image_path

cfg = get_cfg()
cfg.merge_from_file(CFG)



def predict_img(cfg, img, weight_dir=None, threshold=0.5):
    
    cfg = cfg.clone()
    out_imgs = []
    v = Visualizer(
            img[:, :, ::-1],
            metadata=MetadataCatalog.get("bjtu_test_washed"),
            scale=1,
        )
    for weight in weight_dir:
        cfg.MODEL.WEIGHTS = os.path.join(weight, "model_final.pth")
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
        predictor = DefaultPredictor(cfg)
        outputs = predictor(img)
        
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_imgs.append([out.get_image()[:, :, ::-1], weight]) 
        v.output = VisImage(img[:, :, ::-1], scale=1)
        
    return out_imgs


img_path = IMG_PATH
outputs = predict_img(cfg, cv2.imread(img_path)[:, :, ::-1], [CHECK_DIR], 0.5)

plt.figure(figsize=(16, 16))  # 创建一个宽幅图形窗口

for i, (img, weight) in enumerate(outputs):
    plt.imshow(img)
    plt.title(weight)
    plt.axis('off')  # 关闭坐标轴
plt.show()  # 在所有子图绘制完后显示图形
# plt.rcParams['figure.figsize'] = (10, 8)
# plt.rcParams['figure.dpi'] = 200
# plt.imshow(pred_img)