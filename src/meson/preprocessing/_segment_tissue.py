import cv2
import numpy as np
from skimage.morphology import square
from skimage.filters.rank import entropy
from scipy.ndimage import binary_fill_holes
from spatialdata.models import Labels2DModel
from spatialdata.transformations import set_transformation, Identity


def segment_tissue(sdata, image_name, entropy_ksize=7,
                  median_ksize=51,
                  dilation_ksize=0, 
                  cs: str | None = None):
    if cs is None:
        cs = image_name.split('_')[0]
    img = sdata[image_name]
    img_hsv = cv2.cvtColor(img.transpose('y', 'x', 'c').compute().to_numpy(), cv2.COLOR_RGB2HSV) 
    img_sat = img_hsv[:, :, 1] # measures saturation of some sort...
    img_ent = entropy(img_sat, square(entropy_ksize))
    img_ent_rescaled = ((img_ent - img_ent.min()) / \
                        (img_ent.max() - img_ent.min()) * 255).astype('uint8')
    img_med = cv2.medianBlur(img_ent_rescaled, median_ksize) 
    _, img_thres = cv2.threshold(img_med, 0, 255, 
                        cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    img_filled = binary_fill_holes(img_thres).astype(np.uint8)
    img_dilated = cv2.dilate(img_filled, square(dilation_ksize))
    label = Labels2DModel.parse(img_dilated)
    set_transformation(label, Identity(), to_coordinate_system=cs)
    sdata[f'{image_name}_tissue'] = label
    return sdata