function [kspace_xyzbc, offsets] = extract(scanarchive_file)
%% read k-space data from scan archive, zero-filling missing array entries

archive = GERecon('Archive.Load', scanarchive_file);

nx = archive.DownloadData.rdb_hdr_rec.rdb_hdr_da_xres;
ny = archive.DownloadData.rdb_hdr_rec.rdb_hdr_da_yres - 1;
nz = archive.SlicesPerPass(1);
stop = archive.DownloadData.rdb_hdr_rec.rdb_hdr_dab.stop_rcv;
start = archive.DownloadData.rdb_hdr_rec.rdb_hdr_dab.start_rcv;
nc = stop - start + 1;
numEchoes = archive.DownloadData.rdb_hdr_rec.rdb_hdr_nechoes;
numPasses = archive.DownloadData.rdb_hdr_rec.rdb_hdr_npasses;
nb = numEchoes * numPasses;

num_passes = archive.DownloadData.rdb_hdr_rec.rdb_hdr_npasses;
kspace_xyzbc = zeros(nx, ny, nz, nb, nc, 'single');
mask = zeros(ny, nz, nb, 'logical');
pass = 1;
bins_per_pass = nb / num_passes;

offsets = archive.DownloadData.rdb_hdr_rec.rdb_hdr_mavric_b0_offset;
offsets= offsets(1:nb); % frequency offsets are stored as a 1x40 vector. Only 24 needed. Others are stored as 1000, which should be removed.
bin_order = floor(offsets/1000 + nb/2);
if min(bin_order) == 0
    bin_order = bin_order + 1;
end

% Loop through each control, sorting frames if applicable
fprintf('Beginning control loop with %d iterations\n', archive.ControlCount);

tic
for i = 1:archive.ControlCount
    control = GERecon('Archive.Next', archive);

    % Sort only programmable packets in range
    if(control.opcode == 1)
        iy = control.viewNum;
        iz = control.sliceNum + 1;
        bin_index = bin_order(control.echoNum + (pass - 1) * bins_per_pass + 1);
        data = squeeze(control.Data);
%         if mod(iz, 2) == 0
%             data = -data;  % half-FOV shift in z - why is this necessary?
%         end
        kspace_xyzbc(:, iy, iz, bin_index, :) = data;
        mask(iy, iz, bin_index) = true;
    elseif(control.opcode == 0) % end of pass and/or scan
        if(pass < num_passes)
            pass = pass + 1;
        end
    end

    if (mod(i, 10000) == 0)
        fprintf('At control %d. ', i);
        toc
    end

end
disp('Finished control loop.')
toc

GERecon('Archive.Close', archive);
disp('Closed Archive')

end