function [image_xyzb] = combine_coils(scanarchive_file, kspace_xyzbc)
%% transform coil-bin k-space into bin images

archive = GERecon('Archive.Load', scanarchive_file);

[nx, ny, nz, nb, nc] = size(kspace_xyzbc);
nx = 256; % TODO why does arc.synthesize go to this bigger size?
ny = 256;
image_xyzb = zeros(nx, ny, nz, nb);
image_xyc = zeros(nx, ny, nc);

for ib = 1:nb
    fprintf('working on bin %d of %d\n', ib, nb)
    %k_xyzc = squeeze(kspace_xyzbc(:,1:yRes,:,ib,:)); 
    k_xyzc = squeeze(kspace_xyzbc(:,:,:,ib,:)); % TODO look into the yRes thing
    k_xyzc = GERecon('Arc.Synthesize', k_xyzc);
    h_xyzc = ifft(k_xyzc, [], 3);
    for iz = 1:nz
        for ic = 1:nc
            image_xy = GERecon('Transform', h_xyzc(:,:,iz,ic));           
            image_xyc(:,:,ic) = image_xy;
        end
        image_xyzb(:,:,iz,ib) = GERecon('SumOfSquares', image_xyc);  
    end
end