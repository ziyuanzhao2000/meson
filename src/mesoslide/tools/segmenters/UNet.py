import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import torch
import cv2 
import os
import numpy as np
import zarr
from skimage.morphology import remove_small_holes, remove_small_objects
from tqdm.auto import tqdm

def preprocess_datasets(images, labels=None, positive_class_idx=[0], target_size=448, smooth_size=31, cleanup=0):
    n_samples = len(images)
    out_images = np.zeros((n_samples, 3, target_size, target_size), dtype=np.uint8)
    out_labels = np.zeros((n_samples, 1, target_size, target_size))

    if labels is not None:
        labels = np.isin(labels, positive_class_idx)
    for i in tqdm(range(n_samples)):
        image = images[i]
        for j in range(3):
            out_images[i, j] = cv2.resize(image[j], (target_size, target_size))
        if labels is not None:
            label = labels[i]
            if cleanup > 0:
                label = remove_small_objects(label, min_size=cleanup)
                label = remove_small_holes(label, area_threshold=cleanup)
            resized = cv2.resize(label.astype(np.float32), 
                                (target_size, target_size),
                                interpolation=cv2.INTER_CUBIC)
            out_labels[i, 0] = cv2.GaussianBlur(resized, (smooth_size, smooth_size), 0)
    return out_images, out_labels


def save_segmentation_data_to_zarr(images_array, masks_array, output_dir):
    """
    Save image and mask arrays to Zarr format.
    
    Parameters:
    -----------
    images_array : numpy.ndarray
        Array of images with shape (N, C, H, W)
    masks_array : numpy.ndarray
        Array of masks with shape (N, 1, H, W)
    output_dir : str
        Path to save the Zarr dataset
    """
    # Validate input shapes
    assert len(images_array.shape) == 4, f"Images array should be 4D (N,C,H,W), got {images_array.shape}"
    assert len(masks_array.shape) == 4, f"Masks array should be 4D (N,1,H,W), got {masks_array.shape}"
    assert images_array.shape[0] == masks_array.shape[0], "Number of images and masks must match"
    assert masks_array.shape[1] == 1, f"Masks should have 1 channel, got {masks_array.shape[1]}"
    assert images_array.shape[2:] == masks_array.shape[2:], "Image and mask spatial dimensions must match"
    
    # Create the zarr store
    os.makedirs(os.path.dirname(output_dir) or ".", exist_ok=True)
    root = zarr.open_group(output_dir, mode="w")

    N, C, H, W = images_array.shape
    images = root.create_array("images", shape=images_array.shape, chunks=(1, C, H, W),
                               dtype=images_array.dtype)
    masks = root.create_array("masks", shape=masks_array.shape, chunks=(1, 1, H, W),
                              dtype=masks_array.dtype)
    root.attrs["shape"] = {
        "images": list(images_array.shape),
        "masks": list(masks_array.shape),
        "channels": C,
        "height": H,
        "width": W,
        "samples": N,
    }
    images[:] = images_array
    masks[:] = masks_array

    print(f"Successfully saved {N} image-mask pairs to {output_dir}")
    return root

import torch
from torch.utils.data import Dataset
import zarr
from torchvision import tv_tensors  # <-- Crucial import for v2 transforms

class ZarrSegmentationDataset(Dataset):
    """PyTorch Dataset for loading segmentation data from Zarr storage"""
    
    def __init__(self, zarr_path, transform=None, normalize=True, mask_transform=None, 
                mode='binary'):
        self.root = zarr.open(zarr_path, mode='r')
        self.images = self.root['images']
        self.masks = self.root['masks']
        self.transform = transform
        self.mask_transform = mask_transform
        self.normalize = normalize
        self.mode = mode
        
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        
        if image.ndim == 2:
            image = image.unsqueeze(0)
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if self.normalize and image.max() > 1.0:
            image = image.float() / 255.0
            
        if self.transform is not None:
            image = tv_tensors.Image(image)
            mask = tv_tensors.Mask(mask)
            
            image, mask = self.transform(image, mask)
        
        mask = torch.clip(mask, 0, 1)
        
        return torch.as_tensor(image), torch.as_tensor(mask)
class GenericSegmenter(pl.LightningModule):
    """smp segmentation model (e.g. UNet) for binary masks, trained with Adam and a per-step cosine schedule.

    Parameters
    ----------
    arch, encoder_name, in_channels, out_classes
        Passed to `smp.create_model`, with any other keyword arguments (e.g. encoder_weights).
    loss : {"dice", "symmetric_dice"}, default "dice"
        "dice": Dice loss on logits. "symmetric_dice": 0.5 * (Dice(p, y) + Dice(1 - p, 1 - y)) on sigmoid
        probabilities p, which also scores the background, so batches without positive pixels still contribute.
    t_max : int, default 100
        Steps of the cosine learning-rate schedule (epochs x steps per epoch). `T_max` is accepted as an alias.
    mean, std : sequence, optional
        Input standardization; defaults to the encoder's ImageNet statistics.
    """

    def __init__(self, arch, encoder_name, in_channels, out_classes, *, loss="dice", t_max=None,
                 mean=None, std=None, **kwargs):
        super().__init__()
        if loss not in ("dice", "symmetric_dice"):
            raise ValueError(f"loss must be 'dice' or 'symmetric_dice', got {loss!r}")
        t_max = kwargs.pop("T_max", 100) if t_max is None else t_max
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        params = smp.encoders.get_preprocessing_params(encoder_name)
        if mean is not None and std is not None:
            pass
        elif in_channels == 3:
            mean = params["mean"]
            std = params["std"]
        else:
            # Average the 3 ImageNet channels and repeat for 'in_channels'
            mean = [sum(params["mean"]) / len(params["mean"])] * in_channels
            std = [sum(params["std"]) / len(params["std"])] * in_channels

        # Dynamically set the view to match in_channels instead of hardcoding 3
        self.register_buffer("std", torch.tensor(std).view(1, in_channels, 1, 1).float())
        self.register_buffer("mean", torch.tensor(mean).view(1, in_channels, 1, 1).float())
        
        self.loss = loss
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=(loss == "dice"))

        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.T_max = t_max

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        if isinstance(batch, dict):
            image, mask = batch['image'], batch['mask']
        else:
            image, mask = batch

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32,
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image.float())
        prob_mask = logits_mask.sigmoid()
        if self.loss == "symmetric_dice":
            loss = 0.5 * (self.loss_fn(prob_mask, mask) + self.loss_fn(1 - prob_mask, 1 - mask))
        else:
            loss = self.loss_fn(logits_mask, mask)

        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(
            pred_mask.long(), mask.long(), mode="binary"
        )
        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(
            tp, fp, fn, tn, reduction="micro-imagewise"
        )

        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset
        # with "empty" images (images without target class) a large gap could be observed.
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        train_loss_info = self.shared_step(batch, "train")
        # append the metics of each step to the
        self.training_step_outputs.append(train_loss_info)
        return train_loss_info

    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # empty set output list
        self.training_step_outputs.clear()
        return

    def validation_step(self, batch, batch_idx):
        valid_loss_info = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(valid_loss_info)
        return valid_loss_info

    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs.clear()
        return

    def test_step(self, batch, batch_idx):
        test_loss_info = self.shared_step(batch, "test")
        self.test_step_outputs.append(test_loss_info)
        return test_loss_info

    def on_test_epoch_end(self):
        self.shared_epoch_end(self.test_step_outputs, "test")
        # empty set output list
        self.test_step_outputs.clear()
        return

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=2e-4)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.T_max, eta_min=1e-5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
        return