exam_dir = '/Users/artoews/root/data/mri/231021/';

% plastic
sa = [exam_dir 'Series20/ScanArchive_415723SHMR18_20231021_205205589.h5'];
field_plastic = get_field_from_archive(sa);

% metal
sa = [exam_dir 'Series47/ScanArchive_415723SHMR18_20231021_222010247.h5'];
field_metal = get_field_from_archive(sa);

field = field_metal - field_plastic;
save('field_map_hz', 'field');

[nx, ny, nz] = size(field);
figure(5); imshow(field(:, :, nz/2), []); colorbar;
figure(6); imshow(squeeze(field(:, ny/2, :)), []); colorbar; daspect([1 2 1]);
