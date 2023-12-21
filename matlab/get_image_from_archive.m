function image = get_image_from_archive(file)
    [kspace_xyzbc, offsets] = extract(file);
    image_xyzb = combine_coils(file, kspace_xyzbc);
    image = combine_bins(image_xyzb, offsets);
    image = correct(file, image);
end