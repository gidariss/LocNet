function [ bboxes ] = scale_bboxes( bboxes, scale_factor )
% scale_bboxes: it scales the set bounding boxes bboxes by the scale_factor
% factor.
%
% INPUT:
% 1) bboxes: a N x 4 array with the input bounding box coordinates in the 
% form of [x0,y0,x1,y1] (where (x0,y0) is the top-left corner and (x1,y1)  
% the bottom left corner)
% 2) scale_factor: a 1 x 1 or 2 x 1 array with the scaling factor of the
% bounding boxes. If scale_factor is a 1 x 1 array then the same scaling
% factor will be applied on both the x and y axis. If scale_factor is a
% 2 x 1 (or 1 x 2) array then the bounding boxes will be scaled across the
% y dimension by scale_factor(1) and across the x dimension by
% scale_factor(2).
% 
% OUTPUT:
% 1) bboxes: a N x 4 array with the output bounding box coordinates in the 
% form of [x0,y0,x1,y1] (where (x0,y0) is the top-left corner and (x1,y1)  
% the bottom left corners.
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

if numel(scale_factor) == 1, scale_factor(2) = scale_factor(1); end
assert(numel(scale_factor) == 2);
scale_factor = single(scale_factor);


bboxes_center      = [(bboxes(:,1)+bboxes(:,3)), (bboxes(:,2)+bboxes(:,4))]/2;
bboxes_width_half  = (bboxes(:,3) - bboxes(:,1))/2;
bboxes_width_half  = bboxes_width_half * scale_factor(2);

bboxes_height_half = (bboxes(:,4) - bboxes(:,2))/2;
bboxes_height_half = bboxes_height_half * scale_factor(1);

bboxes = round([bboxes_center(:,1) - bboxes_width_half, ...
                bboxes_center(:,2) - bboxes_height_half, ...
                bboxes_center(:,1) + bboxes_width_half, ...
                bboxes_center(:,2) + bboxes_height_half]);
end