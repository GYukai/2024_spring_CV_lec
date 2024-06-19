import detectron2.data.transforms as T
from detectron2.data import DatasetMapper  # the default mapper
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader
from fvcore.transforms.transform import (
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    Transform,
    TransformList,
)
import PIL.Image as Image


from detectron2.data import transforms as T
import torch
import numpy as np
import random

class NoiseAugmentation(T.Augmentation):
    def __init__(self, intensity):
        super().__init__()
        self._init(locals())
    def get_transform(self, image):
        return NoiseTransform(self.intensity)

class NoiseTransform(Transform):

    def __init__(self, intensity):
        # TODO decide on PIL vs opencv
        super().__init__()
        self._set_attributes(locals())

    def apply_image(self, img, interp=None):
        assert len(img.shape) <= 4
        intensity = np.random.uniform(0, self.intensity)
        if img.dtype == np.uint8:
            img = img.astype(np.float32)
            if len(img.shape) > 2 and img.shape[2] == 1:
                noise_mask = np.random.normal(0, intensity, img.shape[:2])
                noise_mask = np.expand_dims(noise_mask, axis=-1)
            else:
                noise_mask = np.random.normal(0, intensity, img.shape)
            ret = img + noise_mask
            ret = np.clip(ret, 0, 255).astype(np.uint8)
        else:
            if any(x < 0 for x in img.strides):
                img = np.ascontiguousarray(img)
            img = torch.from_numpy(img)
            noise = torch.randn_like(img) * intensity
            img = img + noise
            ret = img.numpy()

        return ret

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

class MaskAugmentation(T.Augmentation):
    def __init__(self, k, p, mode="Black"):
        ''' 
        k: grid number
        p: ratio of mask
        mode: "Gaussian" or "Black"
        '''
        super().__init__()
        self._init(locals())
    def get_transform(self, image):
        return MaskTransform(self.k, self.p, self.mode)

class MaskTransform(Transform):
    def __init__(self, k, p ,mode):
        '''
        mode: "Gaussian" or "Black"
        '''
        # TODO decide on PIL vs opencv
        super().__init__()
        self._set_attributes(locals())


    def apply_image(self, img):
        if isinstance(img, Image.Image):
            img = np.array(img)
        
        img = img.copy()  # Make a copy to ensure no negative strides
        img = torch.tensor(img, dtype=torch.float32)
        h, w = img.shape[:2]
        k, p = self.k, self.p
        mode = self.mode
        num_grid_x = w // k
        num_grid_y = h // k

        num_masked_grids = int(p * num_grid_x * num_grid_y)
        masked_grids = random.sample(range(num_grid_x * num_grid_y), num_masked_grids)

        grid_indices = torch.arange(num_grid_x * num_grid_y).reshape(num_grid_y, num_grid_x)

        mask_indices = torch.isin(grid_indices, torch.tensor(masked_grids, dtype=torch.int64))
        mask_indices = mask_indices.repeat_interleave(k, dim=0).repeat_interleave(k, dim=1)

        mask = torch.zeros((h, w), dtype=torch.float32)
        mask[:mask_indices.shape[0], :mask_indices.shape[1]] = mask_indices.float()

        # Broadcast 
        mask = mask.unsqueeze(-1).expand(-1, -1, img.shape[-1])

        if mode == "Gaussian":
            noise = torch.randn_like(img) * mask
            masked_image = img * (1 - mask) + noise
        elif mode == "Black":
            masked_image = img * (1 - mask)
        
        return masked_image.numpy().astype(np.uint8)

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation
    

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation