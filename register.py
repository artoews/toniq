import itk
import numpy as np

def nonrigid(fixed_image, moving_image, moving_mask, verbose=False):
    moving_image_bspline = itk.image_view_from_array(moving_image)
    fixed_image_bspline = itk.image_view_from_array(fixed_image)
    moving_mask = itk.image_from_array(moving_mask.astype(np.uint8))

    # Import Default Parameter Map
    parameter_object = itk.ParameterObject.New()  # slow

    default_affine_parameter_map = parameter_object.GetDefaultParameterMap('rigid', 1)
    # default_affine_parameter_map['FinalBSplineInterpolationOrder'] = ['1']
    parameter_object.AddParameterMap(default_affine_parameter_map)

    default_bspline_parameter_map = parameter_object.GetDefaultParameterMap('bspline', 1)
    default_bspline_parameter_map['ErodeMask'] = ['true']
    default_bspline_parameter_map['FinalGridSpacingInPhysicalUnits'] = ['20.0']
    default_bspline_parameter_map['ImageSampler'] = ['Full']
    # default_bspline_parameter_map['FinalBSplineInterpolationOrder'] = ['1']
    # default_bspline_parameter_map['Metric'] = ['AdvancedMattesMutualInformation']
    # default_bspline_parameter_map['Registration'] = ['MultiResolutionRegistration']
    # default_bspline_parameter_map['ImagePyramidSchedule'] = ['2', '1']
    # default_bspline_parameter_map['GridSpacingSchedule'] = ['2', '1']
    parameter_object.AddParameterMap(default_bspline_parameter_map)
 
    if verbose:
        print(parameter_object)

    result_image_bspline, result_transform_parameters = itk.elastix_registration_method(
        fixed_image_bspline,
        moving_image_bspline,
        parameter_object=parameter_object,
        fixed_mask=moving_mask,
        log_to_console=False)
    
    return result_image_bspline, result_transform_parameters