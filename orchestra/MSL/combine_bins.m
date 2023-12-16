function [image_xyz] = combine_bins(image_xyzb, offsets)
[nx, ny, nz, nb] = size(image_xyzb);
image_xyz = zeros(nx, ny, nz);
for iz = 1:nz
    slc = squeeze(image_xyzb(:,:,iz,:));
    image_xyz(:, :, iz) = GERecon('Mavric.Combine', slc, iz, nz, offsets, nx);
end
end
