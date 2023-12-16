sa = '/Users/artoews/root/data/mri/231021/Series20/ScanArchive_415723SHMR18_20231021_205205589.h5';

[kspace_xyzbc, offsets] = extract(sa);
image_xyzb = combine_coils(sa, kspace_xyzbc);
% image_xyzb = correct3D_bins(sa, image_xyzb); % TODO try this instead of
% correct3D below. Careful, may need to separate gradwarp and orient so that just
% gradwarp is applied prior to bin correction, not sure.
image_xyz = combine_bins(image_xyzb, offsets');
image_xyz = correct3D(sa, image_xyz);

[nx, ny, nz] = size(image_xyz);

figure;
imshow(image_xyz(:,:,nz/2),[])
figure;
imshow(squeeze(image_xyz(:,ny/2,:)),[]);