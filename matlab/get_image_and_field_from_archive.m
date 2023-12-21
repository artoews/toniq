function [image, field] = get_image_and_field_from_archive(file)
    [kspace_xyzbc, offsets] = extract(file);
    image_xyzb = combine_coils(file, kspace_xyzbc);
    image = combine_bins(image_xyzb, offsets);
    image = correct(file, image);
    field = estimate_field(image_xyzb, offsets);
    field = correct(file, field);
end