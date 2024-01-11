function field = get_field_from_archive(file)
    [kspace_xyzbc, offsets] = extract(file);
    image_xyzb = combine_coils(file, kspace_xyzbc);
    field = estimate_field(image_xyzb, offsets);
    field = correct_geometry(file, field);
end