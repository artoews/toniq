function [image_xyz] = correct(scanarchive_file, image_xyz)
archive = GERecon('Archive.Load', scanarchive_file);
sliceInfoStart = GERecon('Archive.Info', archive, 1);
cornerStart = sliceInfoStart.Corners;
nz = archive.SlicesPerPass(1);
sliceInfoEnd = GERecon('Archive.Info', archive, nz);
cornerEnd = sliceInfoEnd.Corners;
corners = [cornerStart cornerEnd];
image_xyz = GERecon('Gradwarp', image_xyz, corners, 'HRMW');
% TODO see if you can extract coefs directly from archive, instead of
% hard-coding the gradient type as above ('HRMW')
% image_xyz = GERecon('Gradwarp', image_xyz, corners, 'SphericalHarmonicCoefficients', coefs);
for iz=1:nz
    sliceInfo = GERecon('Archive.Info', archive, iz);
    orientation = sliceInfo.Orientation;
    image_xyz(:,:,iz) = GERecon('Orient', image_xyz(:,:,iz), orientation);
end
end