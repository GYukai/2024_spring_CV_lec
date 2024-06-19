import detectron2
from detectron2.utils.logger import setup_logger
import argparse
setup_logger()
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.solver import build_lr_scheduler
# import some common libraries
import numpy as np
import os, json, cv2, random
from detectron2.engine import default_setup, hooks, launch
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from matplotlib import pyplot as plt
import glob
from detectron2.data.datasets.coco import convert_to_coco_json

from vis import *
from dataset_bjtu import *

import torch
from detectron2.engine import DefaultTrainer
from transform import *


parser = argparse.ArgumentParser(description="Detectron2 Training Script")
parser.add_argument('--cfg', required=True, help='Path to the config file')
args = parser.parse_args()


# print(torch.cuda.is_available())
# print(torch.cuda.get_device_capability())


CFG = args.cfg
# CFG = "config/hit.yaml"

train_dataset = None
test_dataset = None

register_coco_instances("bjtu_train_washed", {}, "BJTU_washed/train.json", ".")
register_coco_instances("bjtu_test_washed", {}, "BJTU_washed/test.json", ".")
register_coco_instances("RCTW_train", {}, "RCTW_17/rctw_train.json", "RCTW_17/raw")

# 定义Trainer和Aug
from detectron2.data import transforms as T
class MyColorAugmentation(T.Augmentation):
    def get_transform(self, image):
        r = np.random.rand(2)
        return T.ColorTransform(lambda x: x * r[0] + r[1] * 10)
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader


def build_text_detect_train_aug(cfg):
    
    augs = [
        T.RandomRotation(angle=[-20, 20], expand=False),
        # T.ResizeScale(0.5,2,1024,1024),
        NoiseAugmentation(10),
        MaskAugmentation(30, 0.1),
        T.RandomCrop("relative_range", [0.5, 0.5]),
        # T.ResizeShortestEdge(
        #     cfg.INPUT.MIN_SIZE_TRAIN,
        #     cfg.INPUT.MAX_SIZE_TRAIN,
        #     cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
        # ),
        # T.Resize((800,600)),
        T.RandomBrightness(0.8, 1.8),
        T.RandomContrast(0.6, 1.3),
        T.RandomSaturation(0.8, 1.4),
        T.RandomLighting(0.7),
        T.RandomResize([[512, 512], [768, 512], [512, 768], [256, 512], [512, 256]])  

    ]
    return augs


def build_text_detect_val_aug(cfg):
    return [
        T.Resize((800,600)),
    ]


class CocoTrainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        os.makedirs(cfg.EVAL_DIR, exist_ok=True)
        return COCOEvaluator(dataset_name, cfg, False, cfg.EVAL_DIR)

    @classmethod
    def build_train_loader(cls, cfg):

        mapper = DatasetMapper(
                cfg, is_train=True, augmentations=build_text_detect_train_aug(cfg)
            )

        return build_detection_train_loader(cfg, mapper=mapper)
    


cfg = get_cfg()
cfg.merge_from_file(CFG)
cfg.EVAL_DIR = os.path.join(cfg.OUTPUT_DIR, "coco_eval")
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = CocoTrainer(cfg)

trainer.resume_or_load(resume=False)

# 查看trainer示例
# 训练

trainer.train()
# 测试
""" 
Eval
"""

# dataset_dicts = get_bjtu_dicts(PATH_test)
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # set a custom testing threshold


