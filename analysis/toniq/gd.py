"""Functions for geometric distortion (GD) mapping & plotting.

"""
import itk
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from time import time

from toniq.masks import get_artifact_mask, get_signal_mask
from toniq.plot import overlay_mask, colorbar_axis
from toniq.plot_params import *
from toniq.util import masked_copy

kHz_mm_over_G_cm = 0.42577


def elastix_registration(
        fixed_image: npt.NDArray[np.float64],
        moving_image: npt.NDArray[np.float64],
        fixed_mask: npt.NDArray[np.bool],
        moving_mask: npt.NDArray[np.bool],
        parameter_object: itk.ParameterObject,
        initial_transform: itk.ParameterObject = None,
        verbose: bool = False
        ) -> tuple[npt.NDArray[np.float64], itk.ParameterObject]:
    """Perform image registration using ITK Elastix.

    Args:
        fixed_image (npt.NDArray[np.float64]): fixed image
        moving_image (npt.NDArray[np.float64]): moving image
        fixed_mask (npt.NDArray[np.bool]): registration mask for fixed image
        moving_mask (npt.NDArray[np.bool]): registration mask for moving image
        parameter_object (itk.ParameterObject): parameters specifying how the registration is performed
        initial_transform (itk.ParameterObject, optional): initial transform applied to moving image. Defaults to None.
        verbose (bool, optional): whether to print progress updates from ITK registration process. Defaults to False.

    Returns:
        tuple[npt.NDArray[np.float64], itk.ParameterObject]: [registered moving image, transform parameters]
    """

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
    if verbose:
        print('registration time elapsed (s): {:.1f}'.format(time() - t0))
    return np.asarray(result_image), result_transform_parameters

def transform(
        image: npt.NDArray[np.float64 | np.bool],
        elastix_parameters: itk.ParameterObject
        ) -> npt.NDArray[np.float64]:
    """Apply coordinate transformation to image.

    Image data type is preserved, i.e. float stays float, bool stays bool.

    Implementation adapted from the following two ITK Elastix examples.
    [1] https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_UnitTestExample3_BsplineRegistration.ipynb
    [2] https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_Example09_PointSetAndMaskTransformation.ipynb

    Args:
        image (npt.NDArray[np.float64 | np.bool]): image to be transformed
        elastix_parameters (itk.ParameterObject): coordinate transformation

    Returns:
        npt.NDArray[np.float64 | np.bool]: transformed image
    """
    is_mask = (image.dtype == np.bool)
    if is_mask:
        image = image.astype(np.uint8)
        elastix_parameters.SetParameter('FinalBSplineInterpolationOrder','0') # ensures binary mask is still binary after deformation (per page 23 of elastix manual)
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

def setup_nonrigid(
        verbose: bool = False
        ) -> itk.ParameterObject:
    """Initialize parameters for non-rigid registration with ITK Elastix.

    Args:
        verbose (bool, optional): whether to print progress updates. Defaults to False.

    Returns:
        itk.ParameterObject: registration parameters
    """
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
        # print(parameter_object)
        print('Done ITK setup for non-rigid reg. {:.2f} seconds elapsed'.format(time() - t0))
    return parameter_object

def setup_rigid(
        verbose: bool = False
        ) -> itk.ParameterObject:
    """Initialize parameters for rigid registration with ITK Elastix.

    Args:
        verbose (bool, optional): whether to print progress updates. Defaults to False.

    Returns:
        itk.ParameterObject: registration parameters
    """
    t0 = time()
    parameter_object = itk.ParameterObject.New()
    default_affine_parameter_map = parameter_object.GetDefaultParameterMap('rigid', 1)
    parameter_object.AddParameterMap(default_affine_parameter_map)
    if verbose:
        # print(parameter_object)
        print('Done ITK setup for rigid reg. {:.2f} seconds elapsed'.format(time() - t0))
    return parameter_object

def get_deformation_field(
        moving_image: npt.NDArray[np.float64],
        transform: itk.ParameterObject,
        ) -> npt.NDArray[np.float64]:
    """Compute the deformation field corresponding to a given coordinate transformation.

    Adapted from the following ITK Elastix example.
    https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_Example11_Transformix_DeformationField.ipynb

    Args:
        moving_image (npt.NDArray[np.float64]): moving image (unused?)
        transform (itk.ParameterObject): coordinate transformation

    Returns:
        npt.NDArray[np.float64]: deformation field
    """
    moving_image = itk.image_from_array(moving_image)
    field = itk.transformix_deformation_field(moving_image, transform)
    return np.asarray(field).astype(np.float)[..., ::-1]

def get_jacobian(
        moving_image: npt.NDArray[np.float64],
        transform: itk.ParameterObject
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute the jacobian matrix corresponding to a given coordinate transformation.

    Adapted from the following ITK Elastix example.
    https://github.com/InsightSoftwareConsortium/ITKElastix/blob/main/examples/ITK_Example10_Transformix_Jacobian.ipynb

    Args:
        moving_image (npt.NDArray[np.float64]): moving image (unused?)
        transform (itk.ParameterObject): coordinate transformation

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: [jacobian matrix, determinant of jacobian matrix]
    """
    moving_image = itk.image_from_array(moving_image)
    jacobians = itk.transformix_jacobian(moving_image, transform)
    spatial_jacobian = np.asarray(jacobians[0]).astype(np.float)
    det_spatial_jacobian = np.asarray(jacobians[1]).astype(np.float)
    return spatial_jacobian, det_spatial_jacobian

def get_map(
        plastic_image: npt.NDArray[np.float64],
        metal_image: npt.NDArray[np.float64],
        plastic_mask: npt.NDArray[np.bool],
        metal_mask: npt.NDArray[np.bool],
        itk_parameters: itk.ParameterObject = None,
        inverse: bool = True,
        rigid_prep: bool = False
        ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Compute GD map.

    The GD map is computed as the deformation field resulting from an image-based non-rigid 3D registration.
    Each image is masked to exclude regions for which no spatial correspondence can or should be made due to the severity of intensity artifact or lack of structure.

    Args:
        plastic_image (npt.NDArray[np.float64]): image with no GD (no metal)
        metal_image (npt.NDArray[np.float64]): image with GD (from metal)
        plastic_mask (npt.NDArray[np.bool]): registration mask applied to plastic_image
        metal_mask (npt.NDArray[np.bool]): registration mask applied to metal_image
        itk_parameters (itk.ParameterObject, optional): registration parameters. Defaults to None.
        inverse (bool, optional): whether to compute the inverse transformation (fixed<->moving). Defaults to True.
        rigid_prep (bool, optional): whether to initialize the non-rigid registration with a rigid registration. Defaults to False.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDarray[np.float64]]: [registered image, registered image (masked), deformation field]
    """

    if itk_parameters is None:
        itk_parameters = setup_nonrigid()
    if rigid_prep:
        rigid_itk_parameters = setup_rigid()
        rigid_result, rigid_transform = elastix_registration(metal_image, plastic_image, metal_mask, plastic_mask, rigid_itk_parameters, verbose=False)
        rigid_result_mask = transform(metal_mask, rigid_transform)
        # rigid_result_masked = masked_copy(rigid_result, rigid_result_mask)
        plastic_image = rigid_result
        plastic_mask = rigid_result_mask
    else:
        rigid_result = None
        # rigid_result_masked = None
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
    return result, result_masked, deformation_field

def get_masks(
        implant_mask: npt.NDArray[np.bool],
        ia_map: npt.NDArray[np.float64],
        threshold: float
        ) -> tuple[npt.NDArray[np.bool], npt.NDArray[np.bool]]:
    """Get masks for registration.

    Both masks exclude the implant region for its lack of signal/structure.
    The metal mask additionally excludes areas with intensity artifact above a given threshold.

    Args:
        implant_mask (npt.NDArray[np.bool]): mask excluding the implant region
        ia_map (npt.NDArray[np.float64]): map of intensity artifact
        threshold (float): intensity artifact threshold

    Returns:
        tuple[npt.NDArray[np.bool], npt.NDArray[np.bool]]: [mask for plastic image, mask for metal image]
    """
    ia_mask = get_artifact_mask(ia_map, threshold)
    plastic_mask = get_signal_mask(implant_mask)
    metal_mask = get_signal_mask(implant_mask, artifact_masks=[ia_mask])
    return plastic_mask, metal_mask

def colorbar(
        ax: plt.Axes,
        im: mpl.image.AxesImage,
        lim: float = 2,
        offset: float = 0
        ) -> mpl.colorbar.Colorbar:
    """Plot colorbar for IA map.

    Args:
        ax (plt.Axes): where map is plotted
        im (mpl.image.AxesImage): mappable image from IA plot
        lim (float, optional): +/- limit for colorbar range. Defaults to 2.
        offset (float, optional): positional offset of colorbar from IA map. Defaults to 0.

    Returns:
        mpl.colorbar.Colorbar: colorbar
    """
    cbar = plt.colorbar(im, cax=colorbar_axis(ax, offset=offset), ticks=[-lim, -lim/2, 0, lim/2, lim], extend='both')
    cbar.set_label('Displacement\n(pixels)', size=SMALL_SIZE)
    cbar.ax.tick_params(labelsize=SMALLER_SIZE)
    return cbar

def plot_map(
        ax: plt.Axes,
        gd_map: npt.NDArray[np.float64],
        mask: npt.NDArray[np.bool],
        lim: float = 2,
        show_cbar: bool = True
        ) -> mpl.colorbar.Colorbar | None:
    """Plot GD map.

    Args:
        ax (plt.Axes): target plot 
        ia_map (npt.NDArray[np.float64]): GD map
        mask (npt.NDArray[np.bool]): mask identifying areas where map is valid
        lim (float, optional): +/- limit for color range. Defaults to 2.
        show_cbar (np.bool, optional): whether to include a colorbar. Defaults to True.

    Returns:
        mpl.colorbar.Colorbar: colorbar
    """
    im = ax.imshow(gd_map, cmap=CMAP['field'], vmin=-lim, vmax=lim)
    if mask is not None:
        overlay_mask(ax, ~mask)
    if show_cbar:
        cbar = colorbar(ax, im, lim=lim)
        return cbar
    else:
        return None
