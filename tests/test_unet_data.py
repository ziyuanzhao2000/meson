"""Tests for the U-Net training-data helpers in mesoslide.tools.segmenters.UNet."""

import numpy as np

from mesoslide.tools.segmenters.UNet import (ZarrSegmentationDataset, preprocess_datasets,
                                             save_segmentation_data_to_zarr)


def test_preprocess_and_zarr_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    images = rng.integers(0, 255, size=(3, 3, 64, 64), dtype=np.uint8)
    labels = np.zeros((3, 14, 14), dtype=np.int64)
    labels[:, 4:10, 4:10] = 2
    out_images, out_masks = preprocess_datasets(images, labels, positive_class_idx=[2], target_size=32,
                                                smooth_size=5, cleanup=2)
    assert out_images.shape == (3, 3, 32, 32) and out_masks.shape == (3, 1, 32, 32)
    assert out_masks[:, 0, 16, 16].min() > 0.5 and out_masks[:, 0, 0, 0].max() < 0.5

    path = tmp_path / "data" / "positive"
    save_segmentation_data_to_zarr(out_images, out_masks, str(path))
    dataset = ZarrSegmentationDataset(str(path), normalize=False)
    image, mask = dataset[1]
    assert len(dataset) == 3
    np.testing.assert_array_equal(image.numpy(), out_images[1])
    np.testing.assert_allclose(mask.numpy(), np.clip(out_masks[1], 0, 1))


def test_generic_segmenter_losses_and_schedule():
    import pytest
    import torch

    from mesoslide.tools.segmenters.UNet import GenericSegmenter

    torch.manual_seed(0)
    image = torch.randint(0, 255, (2, 3, 64, 64)).float()
    empty = torch.zeros(2, 1, 64, 64)
    for loss in ("dice", "symmetric_dice"):
        model = GenericSegmenter("UNet", "resnet18", in_channels=3, out_classes=1, encoder_weights=None,
                                 loss=loss, t_max=7)
        assert model.T_max == 7
        out = model.shared_step((image, empty), "train")
        # Plain Dice ignores an all-empty batch; symmetric Dice still scores the background.
        assert (out["loss"].item() > 0) == (loss == "symmetric_dice")
    assert GenericSegmenter("UNet", "resnet18", 3, 1, encoder_weights=None, T_max=11).T_max == 11
    with pytest.raises(ValueError):
        GenericSegmenter("UNet", "resnet18", 3, 1, encoder_weights=None, loss="bce")
