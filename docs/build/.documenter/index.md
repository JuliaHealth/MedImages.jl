<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.formulate_sto_xyz' href='#MedImage3D.formulate_sto_xyz'>#</a>&nbsp;<b><u>MedImage3D.formulate_sto_xyz</u></b> &mdash; <i>Function</i>.




helper function for nifti returns a 4x4 matrix for srow_x, srow_y and srow_z


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L175-L178)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.get_base_indicies_arr' href='#MedImage3D.get_base_indicies_arr'>#</a>&nbsp;<b><u>MedImage3D.get_base_indicies_arr</u></b> &mdash; <i>Function</i>.




return array of cartesian indices for given dimensions in a form of array


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Utils.jl#L3-L5)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.CoordinateTerms' href='#MedImage3D.CoordinateTerms'>#</a>&nbsp;<b><u>MedImage3D.CoordinateTerms</u></b> &mdash; <i>Type</i>.




enums based on https://github.com/InsightSoftwareConsortium/ITK/blob/311b7060ef39e371f3cd209ec135284ff5fde735/Modules/Core/Common/include/itkSpatialOrientation.h#L88


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/MedImage_data_struct.jl#L98-L100)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.interpolate_my' href='#MedImage3D.interpolate_my'>#</a>&nbsp;<b><u>MedImage3D.interpolate_my</u></b> &mdash; <i>Function</i>.




perform the interpolation of the set of points in a given space input_array - array we will use to find interpolated val input_array_spacing - spacing associated with array from which we will perform interpolation Interpolator_enum - enum value defining the type of interpolation keep_begining_same - will keep unmodified first layer of each axis - usefull when changing spacing extrapolate_value - value to use for extrapolation

IMPORTANT!!! - by convention if index to interpolate is less than 0 we will use extrapolate_value (we work only on positive indicies here)


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Utils.jl#L66-L75)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.unique_series_id_within_dicom_files' href='#MedImage3D.unique_series_id_within_dicom_files'>#</a>&nbsp;<b><u>MedImage3D.unique_series_id_within_dicom_files</u></b> &mdash; <i>Function</i>.




helper function for dicom #1 returns an array of unique SERIES INSTANCE UID within dicom files within a dicom directory


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L18-L21)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.formulate_xform_string' href='#MedImage3D.formulate_xform_string'>#</a>&nbsp;<b><u>MedImage3D.formulate_xform_string</u></b> &mdash; <i>Function</i>.




helper function for nifti #2 return relevant xform string names from codes (qform and sform)


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L65-L68)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.resample_to_image' href='#MedImage3D.resample_to_image'>#</a>&nbsp;<b><u>MedImage3D.resample_to_image</u></b> &mdash; <i>Function</i>.




given two MedImage objects and a Interpolator enum value return the moving MedImage object resampled to the fixed MedImage object images should have the same orientation origin and spacing; their pixel arrays should have the same shape It require multiple steps some idea of implementation is below
1. check origin of both images as for example in case origin of the moving image is not in the fixed image we need to return zeros
  
2. we should define a grid on the basis of locations of the voxels in the fixed image and interpolate voxels from the moving image to the grid using for example GridInterpolations
  


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Resample_to_target.jl#L19-L26)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.calculate_inverse_44_matrix' href='#MedImage3D.calculate_inverse_44_matrix'>#</a>&nbsp;<b><u>MedImage3D.calculate_inverse_44_matrix</u></b> &mdash; <i>Function</i>.




helper function for nifti calculates inverse of a 4x4 matrix


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L86-L89)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.formulate_string' href='#MedImage3D.formulate_string'>#</a>&nbsp;<b><u>MedImage3D.formulate_string</u></b> &mdash; <i>Function</i>.




helper function for nifti #1 return a concatenated string for encoded iterables


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L52-L55)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.interpolate_point' href='#MedImage3D.interpolate_point'>#</a>&nbsp;<b><u>MedImage3D.interpolate_point</u></b> &mdash; <i>Function</i>.




interpolate the point in the given space keep_begining_same - will keep unmodified first layer of each axis - usefull when changing spacing


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Utils.jl#L35-L38)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.Interpolator_enum' href='#MedImage3D.Interpolator_enum'>#</a>&nbsp;<b><u>MedImage3D.Interpolator_enum</u></b> &mdash; <i>Type</i>.




Definitions of basic interpolators


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/MedImage_data_struct.jl#L86-L88)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.scale_mi' href='#MedImage3D.scale_mi'>#</a>&nbsp;<b><u>MedImage3D.scale_mi</u></b> &mdash; <i>Function</i>.




given a MedImage object and a Tuple that contains the scaling values for each axis (x,y,z in order)

we are setting Interpolator by using Interpolator enum return the scaled MedImage object 


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Basic_transformations.jl#L235-L240)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.Mode_mi' href='#MedImage3D.Mode_mi'>#</a>&nbsp;<b><u>MedImage3D.Mode_mi</u></b> &mdash; <i>Type</i>.




Indicating do we want to change underlying pixel array spatial metadata or both


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/MedImage_data_struct.jl#L91-L93)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.scale' href='#MedImage3D.scale'>#</a>&nbsp;<b><u>MedImage3D.scale</u></b> &mdash; <i>Function</i>.




overwriting this function from Interpolations.jl becouse check_ranges giving error


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Resample_to_target.jl#L8-L11)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.Image_type' href='#MedImage3D.Image_type'>#</a>&nbsp;<b><u>MedImage3D.Image_type</u></b> &mdash; <i>Type</i>.




Defining image type enum


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/MedImage_data_struct.jl#L17-L19)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.cast_to_array_b_type' href='#MedImage3D.cast_to_array_b_type'>#</a>&nbsp;<b><u>MedImage3D.cast_to_array_b_type</u></b> &mdash; <i>Function</i>.




cast array a to the value type of array b


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Utils.jl#L16-L18)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.Image_subtype' href='#MedImage3D.Image_subtype'>#</a>&nbsp;<b><u>MedImage3D.Image_subtype</u></b> &mdash; <i>Type</i>.




Defining subimage type enum


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/MedImage_data_struct.jl#L31-L33)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.string_intent' href='#MedImage3D.string_intent'>#</a>&nbsp;<b><u>MedImage3D.string_intent</u></b> &mdash; <i>Function</i>.




helper function for nifti returns a string version for the specified intent code from nifti


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L191-L194)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.formulate_timing_scale_for_xyzt_time' href='#MedImage3D.formulate_timing_scale_for_xyzt_time'>#</a>&nbsp;<b><u>MedImage3D.formulate_timing_scale_for_xyzt_time</u></b> &mdash; <i>Function</i>.




helper function for nifti 


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L282-L284)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.set_direction_for_nifti_file' href='#MedImage3D.set_direction_for_nifti_file'>#</a>&nbsp;<b><u>MedImage3D.set_direction_for_nifti_file</u></b> &mdash; <i>Function</i>.




helper function for nifti setting the direction cosines (orientation) for a 3D nifti file


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L459-L462)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.get_pixel_data' href='#MedImage3D.get_pixel_data'>#</a>&nbsp;<b><u>MedImage3D.get_pixel_data</u></b> &mdash; <i>Function</i>.




helper function for dicom #2 returns an array of pixel data for unique ids within dicom files


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L28-L31)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.set_spacing_for_nifti_files' href='#MedImage3D.set_spacing_for_nifti_files'>#</a>&nbsp;<b><u>MedImage3D.set_spacing_for_nifti_files</u></b> &mdash; <i>Function</i>.




helper function for nifti setting spacing for 3D nifti filesd(4D nfiti file yet to be added)


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L403-L406)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.set_origin_for_nifti_file' href='#MedImage3D.set_origin_for_nifti_file'>#</a>&nbsp;<b><u>MedImage3D.set_origin_for_nifti_file</u></b> &mdash; <i>Function</i>.




helper function for nifti setting the origin for a 3D nifti file


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L470-L473)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.formulate_qto_xyz' href='#MedImage3D.formulate_qto_xyz'>#</a>&nbsp;<b><u>MedImage3D.formulate_qto_xyz</u></b> &mdash; <i>Function</i>.




helper function for nifti create a qform matrix from the quaterns


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L112-L115)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.check_sform_qform_similarity' href='#MedImage3D.check_sform_qform_similarity'>#</a>&nbsp;<b><u>MedImage3D.check_sform_qform_similarity</u></b> &mdash; <i>Function</i>.




helper function for nifti  checking similarity of s_transformation_matrix and q_transformation_matrix


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L424-L427)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.crop_mi' href='#MedImage3D.crop_mi'>#</a>&nbsp;<b><u>MedImage3D.crop_mi</u></b> &mdash; <i>Function</i>.




given a MedImage object and a Tuples that contains the location of the begining of the crop (crop_beg) and the size of the crop (crop_size) crops image It modifies both pixel array and metadata we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used) return the cropped MedImage object 


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Basic_transformations.jl#L161-L166)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.calculate_qfac' href='#MedImage3D.calculate_qfac'>#</a>&nbsp;<b><u>MedImage3D.calculate_qfac</u></b> &mdash; <i>Function</i>.




helper function nifti return qfac after calculation


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L76-L79)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.translate_mi' href='#MedImage3D.translate_mi'>#</a>&nbsp;<b><u>MedImage3D.translate_mi</u></b> &mdash; <i>Function</i>.




given a MedImage object translation value (translate_by) and axis (translate_in_axis) in witch to translate the image return translated image It is diffrent from pad by the fact that it changes only the metadata of the image do not influence pixel array we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used) return the translated MedImage object


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Basic_transformations.jl#L212-L217)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.formulate_nifti_image_struct' href='#MedImage3D.formulate_nifti_image_struct'>#</a>&nbsp;<b><u>MedImage3D.formulate_nifti_image_struct</u></b> &mdash; <i>Function</i>.




helper function for nifti creates a nifti_image struct which basically encapsulates all the necessary data, contains voxel data


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L293-L296)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.formulate_spacing_scale_for_xyzt_space' href='#MedImage3D.formulate_spacing_scale_for_xyzt_space'>#</a>&nbsp;<b><u>MedImage3D.formulate_spacing_scale_for_xyzt_space</u></b> &mdash; <i>Function</i>.




helper function for nifti calculating spacing scale from xyzt_units to space


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L271-L274)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.formulate_header_data_dict' href='#MedImage3D.formulate_header_data_dict'>#</a>&nbsp;<b><u>MedImage3D.formulate_header_data_dict</u></b> &mdash; <i>Function</i>.




helper function for nifti creates a data dictionary for header data can be used to create a new NIfTI.NIfTI1Header when saving to a file


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Load_and_save.jl#L210-L213)

</div>
<br>
<div style='border-width:1px; border-style:solid; border-color:black; padding: 1em; border-radius: 25px;'>
<a id='MedImage3D.pad_mi' href='#MedImage3D.pad_mi'>#</a>&nbsp;<b><u>MedImage3D.pad_mi</u></b> &mdash; <i>Function</i>.




given a MedImage object and a Tuples that contains the information on how many voxels to add in each axis (pad_beg) and on the end of the axis (pad_end) we are performing padding by adding voxels with value pad_val It modifies both pixel array and metadata we are setting Interpolator by using Interpolator enum (in basic implementation it will not be used) return the cropped MedImage object 


[source](https://github.com/divital-coder/MedImage.jl/blob/185b08bb8afbc110ffa367a54f0681b62f470481/src/Basic_transformations.jl#L184-L190)

</div>
<br>
