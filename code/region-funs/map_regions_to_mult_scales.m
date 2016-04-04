function [ scaled_regions ] = map_regions_to_mult_scales(region_params, regions, image_size, scales )
% map_regions_to_mult_scales: given a set of regions that comes from an
% image of size image_size, it maps each of them to the appropriate image 
% scale from set of scales given from the parameter scales.
% 
% INPUTS:
% 1) region_params: (type struct) the region pooling parameters. Some of 
% its fields are:
%   a) sz_conv_standard: (scalar value) the last convolution size
%   b) step_standard: (scalar value) is the stride in pixels of the output 
%   of the last convolutional layer of the network where the regions are 
%   going to be projected (for VGG16 is equal to 16).
% 2) regions: is a N x 8 array with the N input regions. Each region is 
% represented by 8 values [xo0, yo0, xo1, yo1, xi0, yi0, xi1, yi1] that 
% correspond to its outer rectangle [xo0, yo0, xo1, yo1] and its inner 
% rectangle [xi0, yi0, xi1, yi1]. 
% 3) image_size: 2 x 1 or 1 x 2 array with the size of the original image
% 4) scales: NS x 1 or 1 x NS array with the image scales that are used. NS
% is the number of images. The i-th value is the size in pixels of the
% smallest dimension of the image in the i-th scale.
% 
% OUTPUT:
% 1) scaled_regions: is a N x 9 array with the N output regions. Each region is 
% represented by 9 values [scale_id, xo0, yo0, xo1, yo1, xi0, yi0, xi1, yi1]  
% that correspond to its outer rectangle [xo0, yo0, xo1, yo1] and its inner 
% rectangle [xi0, yi0, xi1, yi1]. scale_id corresponds to the scale id to
% which each region is mapped.
%
% This file is part of the code that implements the following paper:
% Title      : "LocNet: Improving Localization Accuracy for Object Detection"
% Authors    : Spyros Gidaris, Nikos Komodakis
% Institution: Universite Paris Est, Ecole des Ponts ParisTech
% ArXiv link : http://arxiv.org/abs/1511.07763
% code       : https://github.com/gidariss/LocNet
%
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2016 Spyros Gidaris
% 
% Title     : "LocNet: Improving Localization Accuracy for Object Detection"
% ArXiv link: http://arxiv.org/abs/1511.07763
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------

boxes_out = regions(:,1:4);
boxes_in  = regions(:,5:8);

min_img_sz = min(image_size(1:2));

if length(scales) > 1
    box_areas      = (boxes_out(:,3) - boxes_out(:, 1) + 1) .* (boxes_out(:,4) - boxes_out(:,2) + 1);
    expected_scale = region_params.sz_conv_standard * region_params.step_standard * min_img_sz ./ sqrt(box_areas);
    expected_scale = round(expected_scale(:));
    [~, best_scale_ids] = min(abs(bsxfun(@minus, scales, expected_scale(:))), [], 2);   
else
    best_scale_ids = ones(size(boxes_out, 1), 1);
end
    
boxes_scales     = scales(best_scale_ids(:));
scaled_boxes_out = bsxfun(@times, (boxes_out - 1), (boxes_scales(:) - 1)) / (min_img_sz - 1) + 1;
scaled_boxes_in  = bsxfun(@times, (boxes_in  - 1), (boxes_scales(:) - 1)) / (min_img_sz - 1) + 1;
scaled_regions   = [best_scale_ids(:), scaled_boxes_out, scaled_boxes_in];
end