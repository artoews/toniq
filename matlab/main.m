exam_dir = '/Users/artoews/root/data/mri/231021/';

sa = [exam_dir 'Series20/ScanArchive_415723SHMR18_20231021_205205589.h5']; % plastic
% sa = [exam_dir 'Series47/ScanArchive_415723SHMR18_20231021_222010247.h5']; % metal

% NOTE
% if MATLAB crashes, check that you're not using single-float precision.
% it seems that some Orchestra functions require double-float precsion.

[kspace_xyzbc, offsets] = extract(sa);
image_xyzb = combine_coils(sa, kspace_xyzbc);
image_xyz = combine_bins(image_xyzb, offsets);
image_xyz = correct(sa, image_xyz);
field = estimate_field(image_xyzb, offsets);
field = correct(sa, field);

[nx, ny, nz] = size(image_xyz);
figure(11); imshow(image_xyz(:,:,nz/2),[]); colorbar;
figure(12); imshow(field(:,:,nz/2),[]); colorbar;
figure(13); imshow(squeeze(image_xyz(:,ny/2,:)),[]); colorbar; daspect([1 2 1]);
figure(14); imshow(squeeze(field(:,ny/2,:)),[]); colorbar; daspect([1 2 1]);