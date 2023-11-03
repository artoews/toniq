function Arc2DRecon(pfilePath, samplingPatternPath)
%% Arc2DRecon - Reconstruct a 2D ARC Pfile
%
% Copyright 2019 General Electric Company. All rights reserved.
% GE Proprietary and Confidential Information. Only to be distributed with
% permission from GE. Resulting outputs are not for diagnostic purposes.
%
% Arc2DRecon(pfilePath, samplingPatternPath)
% will reconstruct the 2D ARC data in the given pfile. The sampling pattern
% file (kacqXXX.txt) must be specified to describe the acquisition of the
% K-Space.
%
% Limitations: intensity correction

    % Load Pfile
    pfile = GERecon('Pfile.Load', pfilePath);

    % Load Arc Sampling Pattern (kacq_yz.txt)
    GERecon('Arc.LoadKacq', samplingPatternPath);

    % Load KSpace
    kSpace = zeros(pfile.xRes, pfile.yRes, pfile.slices, pfile.channels);

    for s = 1:pfile.slices
        for c = 1:pfile.channels
            
            % Load K-Space
            kSpace(:,:,s,c) = GERecon('Pfile.KSpace', s, 1, c);
            
        end
    end

    % Loop for each slice/channel to create a magnitude image
    for s = 1:pfile.slices
        
        % Synthesize KSpace
        kSpace(:,:,s,:) = GERecon('Arc.Synthesize', kSpace(:,:,s,:));
            
        for c = 1:pfile.channels
        
            % Transform K-Space
            channelImage = GERecon('Transform', kSpace(:,:,s,c));

            channelImages(:,:,c) = channelImage;
        end

        % Get corners and orientation for this slice location
        corners = GERecon('Pfile.Corners', s);
        orientation = GERecon('Pfile.Orientation', s);
            
        % Apply Channel Combination
        combinedImage = GERecon('SumOfSquares', channelImages);

        % Create Magnitude Image
        magnitudeImage = abs(combinedImage);
        
        % Apply Gradwarp
        gradwarpedImage = GERecon('Gradwarp', magnitudeImage, corners);

        % Orient the image
        finalImage = GERecon('Orient', gradwarpedImage, orientation);

        % Display
        imagesc(finalImage);
        
        % Display
        title(['Slice: ' num2str(s)]);

        % Save DICOMs
        filename = ['DICOMs/image' num2str(s) '.dcm'];
        GERecon('Dicom.Write', filename, finalImage, s, orientation, corners);

        pause(0.1);
    end
end
