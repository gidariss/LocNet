function regions = create_regions_from_boxes( region_params, boxes )
% create_regions_from_boxes: given a set of bounding boxes it creates the
% regions that will be fed to the recognition/localization network
% 
% INPUTS:
% 1) region_params: (type struct) the region pooling parameters. Some of 
% its fields are:
%   a) scale_inner: scalar value with the scaling factor of the inner
%   rectangle of the region. In case this value is 0 then actually no inner
%   rectangle is used
%   b) scale_outer: scalar value with the scaling factor of the outer
%   rectangle of the region. 
%   c) half_bbox: intiger value in the range [1,2,3,4]. If this parameter 
%   is set to 1, 2, 3, or 4 then each bounding box will be reshaped to its
%   left, right, top, or bottom half part correspondingly. This action is
%   performed prior to scaling the box according to the scale_inner and
%   scale_outer params. If this parameter is missing or it is empty then 
%   the action of taking the half part of bounding box is NOT performed.
% 2) boxes: a N x 4 array with the bounding box coordinates in the form of
% [x0,y0,x1,y1] (where (x0,y0) is the top-left corner and (x1,y1) the 
% bottom left corner)
%
% OUTPUT:
% 1) regions: is a N x 8 array that contains the region coordinates of each 
% of the N bounding boxes. Note that each region is represented by 8 values 
% [xo0, yo0, xo1, yo1, xi0, yi0, xi1, yi1] that correspond to its outer 
% rectangle [xo0, yo0, xo1, yo1] and its inner rectangle 
% [xi0, yi0, xi1, yi1]. 
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

assert(all(region_params.scale_outer >= region_params.scale_inner));

boxes      = transform_bboxes(region_params, boxes);
boxes_out  = scale_bboxes(boxes, region_params.scale_outer);
boxes_in   = scale_bboxes(boxes, region_params.scale_inner);
regions    = [boxes_out, boxes_in];
end

function boxes = transform_bboxes(region_params, boxes)

if isfield(region_params, 'half_bbox') && ~isempty(region_params.half_bbox) && region_params.half_bbox > 0
    boxes = get_half_bbox( boxes, region_params.half_bbox );
end
end

function [ bboxes ] = get_half_bbox( bboxes, half_bbox )

assert(half_bbox >= 1 && half_bbox <= 4);

switch half_bbox
    case 1 % left half 
        bboxes_half_width = floor((bboxes(:,3) - bboxes(:,1)+1)/2);
        bboxes(:,3) = bboxes(:,1) + bboxes_half_width;
    case 2 % right half
        bboxes_half_width = floor((bboxes(:,3) - bboxes(:,1)+1)/2);
        bboxes(:,1) = bboxes(:,3) - bboxes_half_width;
    case 3 % up half
        bboxes_half_height = floor((bboxes(:,4) - bboxes(:,2)+1)/2);
        bboxes(:,4) = bboxes(:,2) + bboxes_half_height;
    case 4 % down half
        bboxes_half_height = floor((bboxes(:,4) - bboxes(:,2)+1)/2);
        bboxes(:,2) = bboxes(:,4) - bboxes_half_height;
end
bboxes = round(bboxes);
end
