import itk
import matplotlib.pyplot as plt
import numpy as np
from os import path
import scipy.ndimage as ndi
from skimage import morphology
from time import time

from toniq.masks import get_artifact_mask, get_signal_mask
from toniq.plot import overlay_mask, imshow2, colorbar_axis
from toniq.plot_params import *
from toniq.util import masked_copy

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
        # print('field_x', np.max(field_x), np.max(field_x_alt))
    field_y = np.zeros_like(field_kHz)
    field_z = field_kHz / (gz_Gcm * kHz_mm_over_G_cm * voxel_size_z_mm)
    return np.stack((field_x, field_y, field_z), axis=-1)   

def elastix_registration(fixed_image, moving_image, fixed_mask, moving_mask, parameter_object, initial_transform=None, verbose=False):
    t0 = time()
    # could change these to image_view_from_array to pass by reference? I don't *think* they are modified
    moving_image = itk.image_from_array(moving_image)
    fixed_image = itk.image_from_array(fixed_image)
    if fixed_mask is not None:
        fixed_mask = itk.image_from_array(fixed_mask.astype(np.uint8))
    if moving_mask is not None:
        moving_mask = itk.image_from_array(moving_mask.astype(np.uint8))
    if initial_transform is None:
        result_image, result_transform_parameters = itk.elastix_registration_method(
            fixed_image,
            moving_image,
            parameter_object=parameter_object,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            log_to_console=verbose)
    else:
        result_image, result_transform_parameters = itk.elastix_registration_method(
            fixed_image,
            moving_image,
            parameter_object=parameter_object,
            fixed_mask=fixed_mask,
            moving_mask=moving_mask,
            log_to_console=verbose,
            initial_transform_parameter_object=initial_transform)
    print('registration time elapsed (s): {:.1f}'.format(time() - t0))
    return np.asarray(result_image), result_transform_parameters

def elastix_registration_2d(fixed_image, moving_image, fixed_mask, moving_mask, parameter_object, verbose=True):
    results = []
    transforms = []
    for i in range(fixed_image.shape[1]):
        print('registering y slice', i)
        if i < fixed_image.shape[1] - 1:
            if i == 0:
                result, transfrm = elastix_registration(fixed_image[:, i, :], moving_image[:, i, :], fixed_mask[:, i, :], moving_mask[:, i, :], parameter_object, verbose=verbose)
            else:
                result, transfrm = elastix_registration(fixed_image[:, i, :], moving_image[:, i, :], fixed_mask[:, i, :], moving_mask[:, i, :], parameter_object, verbose=verbose, initial_transform=transfrm)
        else:
            print('skipping registration')
            # temporary hack to avoid dark slice resulting from rigid registration prep
            pass
        results.append(result)
        transforms.append(transfrm)
    results = np.stack(results, axis=1)
    return results, transforms

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

def transform_2d(image, elastix_parameters_list):
    transformed_images = []
    for i in range(image.shape[1]):
        transformed_image = transform(image[:, i, :], elastix_parameters_list[i])
        transformed_images.append(transformed_image)
    return np.stack(transformed_images, axis=1)

def setup_nonrigid(verbose=False):
    if verbose:
        print('Beginning ITK setup...')
    t0 = time()
    parameter_object = itk.ParameterObject.New()  # slow
    default_bspline_parameter_map = parameter_object.GetDefaultParameterMap('bspline', 1)
    # 20-30 seems to work well empirically; but below 30, start to see some displacement showing up in y & z maps when they should be zero
    default_bspline_parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    default_bspline_parameter_map['BSplineTransformSplineOrder'] = ['3']
    default_bspline_parameter_map['ImageSampler'] = ['Full']
    default_bspline_parameter_map['ErodeMask'] = ['true']
    del default_bspline_parameter_map['FinalGridSpacingInPhysicalUnits']
    default_bspline_parameter_map['FinalGridSpacingInVoxels'] = ['20']
    # default_bspline_parameter_map['NumberOfResolutions'] = ['1']
    # default_bspline_parameter_map['GridSpacingSchedule'] = ['1.0', '1.0', '1.0'] # this number times final grid spacing is the b-spline grid size for each dim at each pyramid level
    # default_bspline_parameter_map['ImagePyramidSchedule'] = ['2', '2', '2'] # this number over 2 is the sigma of gaussian blurring applied to each dim at each pyramid level
    # default_bspline_parameter_map['GridSpacingSchedule'] = ['2.0', '2.0', '2.0', '1.5', '1.5', '1.5'] # this number times final grid spacing is the b-spline grid size for each dim at each pyramid level
    # default_bspline_parameter_map['ImagePyramidSchedule'] = ['1', '1', '1', '1', '1', '1'] # this number over 2 is the sigma of gaussian blurring applied to each dim at each pyramid level
    parameter_object.AddParameterMap(default_bspline_parameter_map)
    if verbose:
        print(parameter_object)
        print('Done ITK setup for non-rigid reg. {:.2f} seconds elapsed'.format(time() - t0))
    return parameter_object

def setup_rigid(verbose=False):
    t0 = time()
    parameter_object = itk.ParameterObject.New()
    default_affine_parameter_map = parameter_object.GetDefaultParameterMap('rigid', 1)
    parameter_object.AddParameterMap(default_affine_parameter_map)
    if verbose:
        # print(parameter_object)
        print('Done ITK setup for rigid reg. {:.2f} seconds elapsed'.format(time() - t0))
    return parameter_object

def get_deformation_field(moving_image, transform):
    # from https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_Example11_Transformix_DeformationField.ipynb
    moving_image = itk.image_from_array(moving_image)
    field = itk.transformix_deformation_field(moving_image, transform)
    return np.asarray(field).astype(np.float)[..., ::-1]

def get_jacobian(moving_image, transform):
    # from https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_Example10_Transformix_Jacobian.ipynb
    moving_image = itk.image_from_array(moving_image)
    jacobians = itk.transformix_jacobian(moving_image, transform)
    spatial_jacobian = np.asarray(jacobians[0]).astype(np.float)
    det_spatial_jacobian = np.asarray(jacobians[1]).astype(np.float)
    return spatial_jacobian, det_spatial_jacobian

def get_map(plastic_image, metal_image, plastic_mask, metal_mask, itk_parameters=None, inverse=True, rigid_prep=False):
    if itk_parameters is None:
        itk_parameters = setup_nonrigid()
    if rigid_prep:
        rigid_itk_parameters = setup_rigid()
        rigid_result, rigid_transform = elastix_registration(metal_image, plastic_image, metal_mask, plastic_mask, rigid_itk_parameters, verbose=False)
        rigid_result_mask = transform(metal_mask, rigid_transform)
        rigid_result_masked = masked_copy(rigid_result, rigid_result_mask)
        plastic_image = rigid_result
        plastic_mask = rigid_result_mask
    else:
        rigid_result = None
        rigid_result_masked = None
        rigid_transform = None
    if inverse:
        fixed_image, fixed_mask = metal_image, metal_mask
        moving_image, moving_mask = plastic_image, plastic_mask
    else:
        fixed_image, fixed_mask = plastic_image, plastic_mask
        moving_image, moving_mask = metal_image, metal_mask
    result, nonrigid_transform = elastix_registration(fixed_image, moving_image, fixed_mask, moving_mask, itk_parameters, verbose=False)
    deformation_field = get_deformation_field(moving_image, nonrigid_transform)
    result_mask = transform(moving_mask, nonrigid_transform)
    result_masked = masked_copy(result, result_mask)
    return result, result_masked, rigid_result, rigid_result_masked, deformation_field, rigid_transform, nonrigid_transform

def get_masks(implant_mask, artifact_map, threshold):
    artifact_mask = get_artifact_mask(artifact_map, threshold)
    fixed_mask = get_signal_mask(implant_mask)
    moving_mask = get_signal_mask(implant_mask, artifact_masks=[artifact_mask])
    return fixed_mask, moving_mask

def colorbar(ax, im, lim=2, offset=0):
    cbar = plt.colorbar(im, cax=colorbar_axis(ax, offset=offset), ticks=[-lim, -lim/2, 0, lim/2, lim], extend='both')
    cbar.set_label('Displacement\n(pixels)', size=SMALL_SIZE)
    cbar.ax.tick_params(labelsize=SMALLER_SIZE)
    return cbar

def plot_map(ax, gd_map, mask, lim=2, show_cbar=True):
    im = ax.imshow(gd_map, cmap=CMAP['field'], vmin=-lim, vmax=lim)
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar = colorbar(ax, im, lim=lim)
        return cbar
