function scaled_regions = map_boxes_to_regions_mult_scales(region_params, bboxes, image_size, scales)
% map_boxes_to_regions_mult_scales: given a set of bounding boxes, it
% creates for each of them a region (acording to the region_params provided)
% and it maps each of those regions from the original image with size 
% image_size to the appropriate image scale from set of scales given from 
% the parameter scales.
%
% 1) region_params: (type struct) the region pooling parameters. Some of 
% its fields are:
%   a) sz_conv_standard: (scalar value) the last convolution size
%   b) step_standard: (scalar value) is the stride in pixels of the output 
%   of the last convolutional layer of the network where the regions are 
%   going to be projected (for VGG16 is equal to 16).
% 2) boxes: a N x 4 array with the bounding box coordinates in the form of
% [x0,y0,x1,y1] (where (x0,y0) is the top-left corner and (x1,y1) the 
% bottom left corner)
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
regions = create_regions_from_boxes( region_params, bboxes );
[ scaled_regions ] = map_regions_to_mult_scales(region_params, regions, image_size, scales );
end