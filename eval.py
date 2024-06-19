
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
args = parser.parse_args()

CFG = args.cfg
CHECK_DIR = args.weight_dir

cfg = get_cfg()
cfg.merge_from_file(CFG)


eval(cfg, "bjtu_test_washed", CHECK_DIR)
