import cv2
from matplotlib import pyplot as plt
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def cv2_imshow(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.rcParams['figure.figsize'] = (10, 8)
    plt.rcParams['figure.dpi'] = 200
    plt.axis('off')
    plt.imshow(im)
    plt.show()

def show_batch(output, index):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    for i, ax in enumerate(axs):
        show_img = output[index][i]
        show_img = cv2.cvtColor(show_img, cv2.COLOR_BGR2RGB)
        # ax.axis('off')
        ax.imshow(show_img)
    plt.show()

from detectron2.data import detection_utils as utils
def show_sample_from_trainer(cfg, trainer):
    train_data_loader = trainer.build_train_loader(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    data_iter = iter(train_data_loader)
    batch = next(data_iter)
    rows, cols = 2, 2
    plt.figure(figsize=(20,20))

    for i, per_image in enumerate(batch[:4]):
        
        plt.subplot(rows, cols, i+1)
        
        # Pytorch tensor is in (C, H, W) format
        img = per_image["image"].permute(1, 2, 0).cpu().detach().numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = utils.convert_image_to_rgb(img, "BGR")

        visualizer = Visualizer(img, metadata=metadata, scale=1)

        target_fields = per_image["instances"].get_fields()
        labels = None
        vis = visualizer.overlay_instances(
            labels=labels,
            boxes=target_fields.get("gt_boxes", None),
            masks=target_fields.get("gt_masks", None),
            keypoints=target_fields.get("gt_keypoints", None),
        )
        show_img = cv2.cvtColor(vis.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB) 
        
        plt.imshow(show_img)
# show_sample_from_trainer(cfg, trainer)