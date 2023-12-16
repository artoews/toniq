function [image_xyzb] = correct3D_bins(scanarchive_file, image_xyzb)
[nx, ny, nz, nb] = size(image_xyzb);
for ib=1:nb
    image_xyzb(:,:,:,ib) = correct3D(scanarchive_file, image_xyzb(:,:,:,ib));
end