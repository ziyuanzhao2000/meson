import cv2
import numpy as np
from skimage.morphology import footprint_rectangle
from skimage.filters.rank import entropy
from scipy.ndimage import binary_fill_holes
from spatialdata.models import Labels2DModel
from spatialdata.transformations import set_transformation, Affine
from meson._readwrite import get_top_level, get_scaling_factor

def segment_tissue(sdata, image_name, 
                   entropy_ksize=7,
                  median_ksize=51,
                  dilation_ksize=0, 
                  cs: str | None = None):
    if cs is None:
        cs = image_name.split('_')[0]
    img_obj = sdata[image_name]
    img = get_top_level(img_obj)
    img_hsv = cv2.cvtColor(img.transpose('y', 'x', 'c').compute().to_numpy(), cv2.COLOR_RGB2HSV) 
    img_sat = img_hsv[:, :, 1] # measures saturation of some sort...
    img_ent = entropy(img_sat, footprint_rectangle((entropy_ksize, entropy_ksize)))
    img_ent_rescaled = ((img_ent - img_ent.min()) / \
                        (img_ent.max() - img_ent.min()) * 255).astype('uint8')
    img_med = cv2.medianBlur(img_ent_rescaled, median_ksize) 
    _, img_thres = cv2.threshold(img_med, 0, 255, 
                        cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    img_filled = binary_fill_holes(img_thres).astype(np.uint8)
    img_dilated = cv2.dilate(img_filled, footprint_rectangle((dilation_ksize, dilation_ksize)))
    label = Labels2DModel.parse(img_dilated, dims=['y', 'x'])

    # ensure tissue mask transforms to the base layer of image pyramid
    scaling_factors = get_scaling_factor(img_obj)
    affine = Affine(np.eye(3) * [*scaling_factors, 1], 
                    input_axes=['x', 'y'], output_axes=['x', 'y'])
    set_transformation(label, affine, to_coordinate_system=cs)

    sdata[f'{image_name}_tissue'] = label
    return sdata