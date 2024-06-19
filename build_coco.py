from dataset_bjtu import *
from detectron2.data.datasets.coco import convert_to_coco_json

DatasetCatalog.register("bjtu_train_washed", lambda: get_bjtu_dicts(PATH_train_washed))
DatasetCatalog.register("bjtu_test_washed", lambda: get_bjtu_dicts(PATH_test_washed))

MetadataCatalog.get("bjtu_train_washed").set(thing_classes=labels)
MetadataCatalog.get("bjtu_test_washed").set(thing_classes=labels)

convert_to_coco_json("bjtu_train_washed", "BJTU_washed/train.json",allow_cached=True)
convert_to_coco_json("bjtu_test_washed", "BJTU_washed/test.json",allow_cached=True)
