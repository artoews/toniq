function [ kspace_xyzcb, mask_kacq, control_table, yRes ] = load_mavric_scanarchive_2( filename )
%% Code to reconstruct MAVRIC SL ScanArchives 
% S.S. Kaushik
% GE Healthcare Waukesha WI

    archive = GERecon('Archive.Load', filename);

    % Scan parameters
    xRes = archive.DownloadData.rdb_hdr_rec.rdb_hdr_da_xres;
    yRes = archive.DownloadData.rdb_hdr_rec.rdb_hdr_da_yres - 1;
    stop = archive.DownloadData.rdb_hdr_rec.rdb_hdr_dab.stop_rcv;
    start = archive.DownloadData.rdb_hdr_rec.rdb_hdr_dab.start_rcv;
    nChannels = stop - start + 1;
    numEchoes = archive.DownloadData.rdb_hdr_rec.rdb_hdr_nechoes;
    numPasses = archive.DownloadData.rdb_hdr_rec.rdb_hdr_npasses;
    numBins = numEchoes * numPasses;
    
    offsets = archive.DownloadData.rdb_hdr_rec.rdb_hdr_mavric_b0_offset;
    offsets= offsets(1:numBins); % frequency offsets are stored as a 1x40 vector. Only 24 needed. Others are stored as 1000, which shoul
d be removed. 

    df = 1000;
    binOrder = floor(offsets/df + numBins/2);    
    
    if min(binOrder) == 0
        binOrder = binOrder + 1;
    end

    % Keep track of the current pass
    pass = 1;
    zRes = archive.SlicesPerPass(pass);
    % Allocate K-space for each pass separately - conserve memory
    kspace = (zeros(xRes, yRes, nChannels, zRes,numBins));
    mask_kacq = zeros(yRes, zRes, numBins);
    control_table = [];

    % Loop through each control, sorting frames if applicable
    for i = 1:archive.ControlCount

        control = GERecon('Archive.Next', archive);

        % Sort only programmable packets in range
        if(control.opcode == 1 && ...
           control.viewNum > 0 && ...
           control.viewNum <= yRes && ...
           control.sliceNum < zRes)
            
        
            bin_index = binOrder(control.echoNum + (pass - 1) * numBins/numPasses + 1);
            kspace(:,control.viewNum,:,control.sliceNum+1, bin_index) = squeeze(control.Data);

           mask_kacq(control.viewNum, control.sliceNum+1, bin_index) = 1;

        elseif(control.opcode == 0) % end of pass and/or scan

            if(pass < numPasses)
                pass = pass + 1;
            end

        end
    end

    %% Does GE Recon
%    ks = permute(kspace,[1 2 4 3 5]);    
%     mavBinsCombined = GenerateImages(filename, ks,xRes, yRes, zRes, nChannels, numBins);
%     for i=1:zRes
%         combinedImage = GERecon('Mavric.Combine',mavBinsCombined(:,:,:,i),i,zRes,offsets', xRes);
%         % Get corners and orientation for this slice location
%         sliceInfo = GERecon('Archive.Info', archive, i);
%         corners = sliceInfo.Corners;
%         orientation = sliceInfo.Orientation;   
% 
%         % Apply Gradwarp
%         gradwarpedImage = GERecon('Gradwarp', combinedImage, corners);
% 
%         % Orient the image
%         finalImage = GERecon('Orient', gradwarpedImage, orientation);
%         finalImages(:,:,i) = finalImage;
%     end
%     
    %%
    hnover = archive.DownloadData.rdb_hdr_rec.rdb_hdr_hnover;
    
    if(hnover > 0)
        yResFinal = (yRes - hnover) * 2;
    else
        yResFinal = yRes;
    end    
    mask_kacq = padarray(mask_kacq, [(yResFinal - yRes) 0 0], 0, 'post');
    kspace_xyczb = padarray(kspace, [0 (yResFinal - yRes) 0 0 0], 'post'); % pad to resolve partial fourier
    kspace_xyzcb = permute(kspace_xyczb, [1 2 4 3 5]); % permute dimensions
    
    GERecon('Archive.Close', archive);
end