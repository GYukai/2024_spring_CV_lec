
from detectron2.structures import BoxMode
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog, DatasetCatalog
import glob
import json
import cv2

def get_bjtu_dicts(img_dir):
    img_ext = ['jpg', 'png', 'jpeg', 'webp']
    img_files=sorted([filename for ext in img_ext for filename in glob.glob(img_dir + '/**/*.' + ext,recursive=True) ])
    json_files = sorted([filename for filename in glob.glob(img_dir + '/**/*.json',recursive=True) ])
    dataset_dicts = []
    for idx, (img_path, json_path) in enumerate(zip(img_files, json_files)):
        assert img_path.split('/')[-1].split('.')[0] == json_path.split('/')[-1].split('.')[0] # json-img 1-1 correspondence
        with open(json_path) as f:
            imgs_anns = json.load(f)
        record = {}
        height, width = cv2.imread(img_path).shape[:2]
        record["file_name"] = img_path
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
        annotations = []
        
        shapes = imgs_anns['shapes']
        if len(shapes) != 1:
            print(f"{len(shapes)} shapes in {img_path}")
            continue
        if shapes[0]["shape_type"] != "rectangle":
            print(f"{shapes[0]['shape_type']} shape_type in {img_path}")
            continue
        label = shapes[0]["label"]
        points = shapes[0]["points"]
        x_min, x_max = int(min(points[0][0], points[1][0])), int(max(points[0][0], points[1][0]))
        y_min, y_max = int(min(points[0][1], points[1][1])), int(max(points[0][1], points[1][1]))
        obj = {
            "bbox": [x_min, y_min, x_max, y_max],
            "bbox_mode": BoxMode.XYXY_ABS,
            "category_id": label2id[label],
        }
        record["annotations"] = [obj]
        dataset_dicts.append(record)

    return dataset_dicts