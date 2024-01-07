import itk
import numpy as np
import scipy.ndimage as ndi

from os import path
from skimage import morphology
from time import time

import masks
from util import masked_copy, safe_divide

kHz_mm_over_G_cm = 0.42577

def net_pixel_bandwidth(pixel_bandwidth_2, pixel_bandwidth_1):
    if pixel_bandwidth_1 == 0:
        return pixel_bandwidth_2
    else:
        return 1 / (1 / pixel_bandwidth_2 - 1 / pixel_bandwidth_1)

def get_true_field(field_dir):
    metal_field = np.load(path.join(field_dir, 'field-metal.npy')) # kHz
    plastic_field = np.load(path.join(field_dir, 'field-plastic.npy'))  # kHz
    true_field = metal_field - plastic_field
    true_field = ndi.median_filter(true_field, footprint=morphology.ball(4))
    # true_field = ndi.generic_filter(true_field, np.mean, footprint=morphology.ball(3))
    return true_field

def simulated_deformation_fse(field_kHz, gx_Gcm, gz_Gcm, voxel_size_x_mm, voxel_size_z_mm, pbw_kHz=None):
    field_x = field_kHz / (gx_Gcm * kHz_mm_over_G_cm * voxel_size_x_mm)  # voxels
    if pbw_kHz is not None:
        field_x_alt = field_kHz / pbw_kHz
        print('field_x', np.max(field_x), np.max(field_x_alt))
    field_y = np.zeros_like(field_kHz)
    field_z = field_kHz / (gz_Gcm * kHz_mm_over_G_cm * voxel_size_z_mm) / 2 # TODO is this right? originally coded without the 2, but dividing by 2 empirically helps
    return np.stack((field_x, field_y, field_z), axis=-1)   

def elastix_registration(fixed_image, moving_image, fixed_mask, moving_mask, parameter_object, verbose=False):
    t0 = time()
    # good change these to image_view_from_array to pass by reference? I don't *think* they are modified
    moving_image = itk.image_from_array(moving_image)
    fixed_image = itk.image_from_array(fixed_image)
    fixed_mask = itk.image_from_array(fixed_mask.astype(np.uint8))
    moving_mask = itk.image_from_array(moving_mask.astype(np.uint8))
    result_image, result_transform_parameters = itk.elastix_registration_method(
        fixed_image,
        moving_image,
        parameter_object=parameter_object,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        log_to_console=verbose)
    print('registration time elapsed (s): {:.1f}'.format(time() - t0))
    return np.asarray(result_image), result_transform_parameters

def transform(image, elastix_parameters):
    # from https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_UnitTestExample3_BsplineRegistration.ipynb
    # and https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_Example09_PointSetAndMaskTransformation.ipynb
    is_mask = (image.dtype == np.bool)
    if is_mask:
        image = image.astype(np.uint8)
        elastix_parameters.SetParameter('FinalBSplineInterpolationOrder','0') # per page 23 of elastix manual; this will make sure that the deformed segmentation is still a binary label image
    image = itk.image_from_array(image)
    image_transformed = itk.transformix_filter(image, elastix_parameters)
    # below 4 lines accomplish the same thing as the above one-liner
    # transformix_object = itk.TransformixFilter.New(image)
    # transformix_object.SetTransformParameterObject(elastix_parameters)
    # transformix_object.UpdateLargestPossibleRegion()
    # image_transformed = transformix_object.GetOutput()
    image_transformed = np.asarray(image_transformed)
    if is_mask:
        image_transformed = image_transformed.astype(np.bool)
    return image_transformed

def setup_nonrigid(verbose=True):
    if verbose:
        print('Beginning ITK setup...')
    t0 = time()
    parameter_object = itk.ParameterObject.New()  # slow
    default_bspline_parameter_map = parameter_object.GetDefaultParameterMap('bspline', 1)
    # 20-30 seems to work well empirically; but below 30, start to see some displacement showing up in y & z maps when they should be zero
    default_bspline_parameter_map['FinalGridSpacingInPhysicalUnits'] = ['30']  
    default_bspline_parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    default_bspline_parameter_map['BSplineTransformSplineOrder'] = ['3']
    default_bspline_parameter_map['ImageSampler'] = ['Full']
    parameter_object.AddParameterMap(default_bspline_parameter_map)
    if verbose:
        print(parameter_object)
        print('Done ITK setup. {:.2f} seconds elapsed'.format(time() - t0))
    return parameter_object

def setup_rigid(verbose=True):
    t0 = time()
    parameter_object = itk.ParameterObject.New()
    default_affine_parameter_map = parameter_object.GetDefaultParameterMap('rigid', 1)
    parameter_object.AddParameterMap(default_affine_parameter_map)
    if verbose:
        print(parameter_object)
        print('Done ITK setup. {:.2f} seconds elapsed'.format(time() - t0))
    return parameter_object

def get_deformation_field(moving_image, transform):
    # from https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_Example11_Transformix_DeformationField.ipynb
    moving_image = itk.image_from_array(moving_image)
    field = itk.transformix_deformation_field(moving_image, transform)
    return np.asarray(field).astype(np.float)[..., ::-1]  # in 2D, these dimensions seem to be swapped. not sure about 3D

def get_jacobian(moving_image, transform):
    # from https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_Example10_Transformix_Jacobian.ipynb
    moving_image = itk.image_from_array(moving_image)
    jacobians = itk.transformix_jacobian(moving_image, transform)
    spatial_jacobian = np.asarray(jacobians[0]).astype(np.float)
    det_spatial_jacobian = np.asarray(jacobians[1]).astype(np.float)
    return spatial_jacobian, det_spatial_jacobian

def map_distortion(fixed_image, moving_image, fixed_mask=None, moving_mask=None, itk_parameters=None, rigid_prep=True):
    if itk_parameters is None:
        itk_parameters = setup_nonrigid()
    if fixed_mask is None or moving_mask is None:
        fixed_mask, moving_mask = get_registration_masks([fixed_image, moving_image])
    if rigid_prep:
        rigid_itk_parameters = setup_rigid()
        moving_image, rigid_transform = elastix_registration(fixed_image, moving_image, fixed_mask, moving_mask, rigid_itk_parameters)
        moving_mask = transform(moving_mask, rigid_transform)
    moving_image_masked = moving_image.copy()
    moving_image_masked[~moving_mask] = 0
    result, nonrigid_transform = elastix_registration(fixed_image, moving_image, fixed_mask, moving_mask, itk_parameters)
    deformation_field = get_deformation_field(moving_image, nonrigid_transform)
    result_mask = transform(moving_mask, nonrigid_transform)
    result_masked = masked_copy(result, result_mask)
    return result, result_masked, deformation_field

def get_registration_masks(images, thresh):
    mask_empty = masks.get_mask_empty(images[0])
    mask_implant = masks.get_mask_implant(mask_empty)
    mask_signal = masks.get_mask_signal(images[0])
    signal_ref = masks.get_typical_level(images[0], mask_signal, mask_implant)
    masks_register = []
    for image in images[1:]:
        mask_artifact = masks.get_mask_artifact(images[0], image, mask_implant=mask_implant, signal_ref=signal_ref, thresh=thresh)
        mask_register = masks.get_mask_register(mask_empty, mask_implant, mask_artifact)
        masks_register.append(mask_register)
    return masks_register
