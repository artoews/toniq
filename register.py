import itk
import numpy as np
from time import time

import masks
from util import masked_copy, safe_divide

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
    is_mask = (image.dtype == np.bool)
    if is_mask:
        # TODO see page 23 of elastix manual: need to manually change the FinalBSplineInterpolationOrder to 0. This will make sure that the deformed segmentation is still a binary label image
        image = image.astype(np.uint8)
    image = itk.image_from_array(image)
    transformix_object = itk.TransformixFilter.New(image)
    transformix_object.SetTransformParameterObject(elastix_parameters)
    transformix_object.UpdateLargestPossibleRegion()
    image_transformed = transformix_object.GetOutput()
    image_transformed = np.asarray(image_transformed)
    if is_mask:
        # image_transformed = image_transformed.astype(np.bool)
        image_transformed = image_transformed.astype(np.float)
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
    parameter_object = itk.ParameterObject.New()  # slow
    default_affine_parameter_map = parameter_object.GetDefaultParameterMap('rigid', 1)
    parameter_object.AddParameterMap(default_affine_parameter_map)
    if verbose:
        print(parameter_object)
    return parameter_object

def get_deformation_field(moving_image, transform):
    # from https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_Example11_Transformix_DeformationField.ipynb
    moving_image = itk.image_from_array(moving_image)
    field = itk.transformix_deformation_field(moving_image, transform)
    return np.asarray(field).astype(np.float)[..., ::-1]  # in 2D, these dimesnions seem to be swapped. not sure about 3D

def get_jacobian(moving_image, transform):
    # from https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_Example10_Transformix_Jacobian.ipynb
    moving_image = itk.image_from_array(moving_image)
    jacobians = itk.transformix_jacobian(moving_image, transform)
    spatial_jacobian = np.asarray(jacobians[0]).astype(np.float)
    det_spatial_jacobian = np.asarray(jacobians[1]).astype(np.float)
    return spatial_jacobian, det_spatial_jacobian

def map_distortion(fixed_image, moving_image, fixed_mask=None, moving_mask=None, thresh=0.1, itk_parameters=None):
    if itk_parameters is None:
        itk_parameters = setup_nonrigid()
    if fixed_mask is None or moving_mask is None:
        fixed_mask, moving_mask = get_registration_masks([fixed_image, moving_image])
    moving_image_masked = moving_image.copy()
    moving_image_masked[~moving_mask] = 0
    result, transfrm = elastix_registration(fixed_image, moving_image, fixed_mask, moving_mask, itk_parameters)
    deformation_field = get_deformation_field(moving_image, transfrm)
    result_masked = transform(moving_image_masked, transfrm)
    result_mask = np.logical_and(np.abs(result_masked) > thresh, fixed_mask)
    result_masked = masked_copy(result_masked, result_mask)
    return result, result_masked, deformation_field

def get_registration_masks(images):
    mask_empty = masks.get_mask_empty(images[0])
    mask_implant = masks.get_mask_implant(mask_empty)
    mask_signal = masks.get_mask_signal(images[0])
    signal_ref = masks.get_typical_level(images[0], mask_signal, mask_implant)
    masks_register = []
    for image in images[1:]:
        error = image - images[0]
        normalized_error = safe_divide(error, signal_ref)
        mask_artifact = masks.get_mask_artifact(normalized_error)
        mask_register = masks.get_mask_register(mask_empty, mask_implant, mask_artifact)
        masks_register.append(mask_register)
    return masks_register
