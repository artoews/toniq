function [image_xyzb, offsets] = get_bins_from_archive(file)
    [kspace_xyzbc, offsets] = extract(file);    
    image_xyzb = combine_coils(file, kspace_xyzbc);
end