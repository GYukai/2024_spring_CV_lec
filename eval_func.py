
import detectron2
from detectron2.utils.logger import setup_logger

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
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from matplotlib import pyplot as plt
import glob
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.solver import build_lr_scheduler
from vis import *
from dataset_bjtu import *

import torch
from detectron2.engine import DefaultTrainer

def eval(cfg, dataset, weight_dir=None, score=0.5):
    cfg = cfg.clone()
    if weight_dir is None:
        weight_dir = cfg.OUTPUT_DIR
    cfg.MODEL.WEIGHTS = os.path.join(weight_dir, "model_final.pth")
    cfg.EVAL_DIR = os.path.join(weight_dir, "coco_eval")
    predictor = DefaultPredictor(cfg)
    output_dir = os.path.join(weight_dir, "coco_eval")
    os.makedirs(output_dir, exist_ok=True)
    evaluator = COCOEvaluator(dataset, output_dir=output_dir)
    val_loader = build_detection_test_loader(cfg, dataset)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

def check_single(cfg, dataset, weight_dir=None, threshold=0.5):
    dataset_dicts = DatasetCatalog.get(dataset)
    cfg = cfg.clone()
    if weight_dir is None:
        weight_dir = cfg.OUTPUT_DIR
    cfg.MODEL.WEIGHTS = os.path.join(weight_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    predictor = DefaultPredictor(cfg)
    outputs_list = []
    for d in random.sample(dataset_dicts, 20):
        out_list = []
        print(d)
        im = cv2.imread(d["file_name"])
        outputs = predictor(
            im
        )  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v1 = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get(dataset),
            scale=1,
        )
        out = v1.draw_dataset_dict(d)
        out_list.append(out.get_image()[:, :, ::-1])
        v2 = Visualizer(
            im[:, :, ::-1],
            metadata=MetadataCatalog.get(dataset),
            scale=1,
        )
        out = v2.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_list.append(out.get_image()[:, :, ::-1])
        out_list.append(im)
        outputs_list.append(out_list)
    return outputs_list

def predict_img(cfg, img, weight_dir=None, threshold=0.5):
    cfg = cfg.clone()
    if weight_dir is None:
        weight_dir = cfg.OUTPUT_DIR
    cfg.MODEL.WEIGHTS = os.path.join(weight_dir, "model_final.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)
    v = Visualizer(
        img[:, :, ::-1],
        metadata=MetadataCatalog.get(cfg.DATASET.TRAIN[0]),
        scale=1,
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]