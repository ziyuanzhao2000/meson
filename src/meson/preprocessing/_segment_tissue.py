import cv2
import numpy as np
from skimage.morphology import disk
from skimage.filters.rank import entropy
from skimage.filters import rank
from scipy.ndimage import binary_fill_holes
from spatialdata.models import Labels2DModel
from spatialdata.transformations import set_transformation, Affine
from meson._readwrite import get_top_level, get_scaling_factor

def segment_tissue(sdata, image_name, 
                    image_level=-1,
                   entropy_ksize=7,
                   percentile=0,
                  median_ksize=51,
                  closing_ksize=0,
                  opening_ksize=0,
                  dilation_ksize=0, 
                  downsample=1,
                  remove_holes=True,
                  remove_light_regions=0,
                  label_name=None,
                  kernel_type='disk',
                  cs: str | None = None):
    if cs is None:
        cs = image_name.split('_')[0]
    img_obj = sdata[image_name]
    if image_level == -1:
        img = get_top_level(img_obj).compute().transpose('y', 'x', 'c').to_numpy()
    else:
        import spatialdata as sd
        img = sd.get_pyramid_levels(img_obj, n=image_level).compute().transpose('y', 'x', 'c').to_numpy()
    img = cv2.resize(img, (img.shape[1] // downsample, img.shape[0] // downsample))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 
    img_sat = img_hsv[:, :, 1] # measures saturation of some sort...
    if kernel_type == 'disk':
        ent_kernel = disk(entropy_ksize)
    elif kernel_type == 'square':
        ent_kernel = np.ones((entropy_ksize, entropy_ksize))
    if percentile == 0:
        img_ent = entropy(img_sat, ent_kernel)
        img_ent_rescaled = ((img_ent - img_ent.min()) / \
                            (img_ent.max() - img_ent.min()) * 255).astype('uint8')
        img_med = cv2.medianBlur(img_ent_rescaled, median_ksize) 
        img_thres = cv2.threshold(img_med, 0, 255, 
                            cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
    else:
        img_pct = rank.percentile(img_sat, 
            footprint=ent_kernel, 
            p0=percentile)
        img_thres = cv2.threshold(img_pct, 0, 255, 
                            cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
    if remove_holes:
        img_filled = binary_fill_holes(img_thres).astype(np.uint8)
    else:
        img_filled = img_thres.astype(np.uint8)
    if remove_light_regions:
        img_pct = rank.percentile(img_hsv[:, :, 1], 
            footprint=ent_kernel, 
            p0=remove_light_regions)
        img_keep = cv2.threshold(img_pct, 0, 255, 
                        cv2.THRESH_OTSU+cv2.THRESH_BINARY)[1]
        img_filled = np.bitwise_and(img_filled, img_keep)
    
    img_clopened = img_filled
    if closing_ksize:
        img_clopened = cv2.morphologyEx(img_filled, cv2.MORPH_CLOSE, 
                                        disk(closing_ksize))
    if opening_ksize:
        img_clopened = cv2.morphologyEx(img_filled, cv2.MORPH_CLOSE, 
                                        disk(opening_ksize))
    img_dilated = cv2.dilate(img_clopened, disk(dilation_ksize))

    label = Labels2DModel.parse(img_dilated, dims=['y', 'x'])

    # ensure tissue mask transforms to the base layer of image pyramid
    scaling_factors = [f*downsample for f in get_scaling_factor(img_obj, image_level)]
    affine = Affine(np.eye(3) * [*scaling_factors, 1], 
                    input_axes=['x', 'y'], output_axes=['x', 'y'])
    set_transformation(label, affine, to_coordinate_system=cs)
    if label_name is None:
        label_name = f'{image_name}_tissue'
    sdata[label_name] = label

    return sdata