function [field] = estimate_field(image_xyzb, offsets)
%% Estimate a field map from MAVRIC-SL bin images and their offset frequencies
%% Use center-of-mass method

total_mass = sum(abs(image_xyzb), 4);
normalized_mass = abs(image_xyzb) ./ total_mass;
zero_mask = isinf(normalized_mass);
normalized_mass(zero_mask) = 0;
offsets = reshape(offsets, 1, 1, 1, []);
field = sum(normalized_mass .* offsets, 4);

end