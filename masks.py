import scipy.ndimage as ndi
from skimage import filters, morphology


def get_implant_mask(image, filter_radius=3):
    image = ndi.median_filter(image, footprint=morphology.ball(filter_radius)) # maybe unnecessary?
    mask = image < filters.threshold_otsu(image)  # global Otsu
    if filter_radius is not None:
        mask = morphology.binary_opening(mask, morphology.ball(filter_radius))  # opening = erosion (min) then dilation (max)
    # may want to dilate in some cases - leave it to those specific use cases?
    return mask

# to replace get_mask_register with this, you'll need to not the result; filter_radius was previously 5 for get_mask_register
def get_union(masks, close=True, open=True, filter_radius=2):
    mask = (sum(masks) > 0)
    if close:
        mask = ndi.binary_closing(mask, structure=morphology.ball(filter_radius), border_value=1) # closing = dilation (max) then erosion (min)
    if open:
        mask = ndi.binary_opening(mask, structure=morphology.ball(filter_radius), border_value=1) # opening = erosion (min) then dilation (max)
    return mask
