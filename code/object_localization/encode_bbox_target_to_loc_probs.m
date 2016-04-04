function [loc_prob_vectors, bbox_target_quantized, bbox_target_in_region] = encode_bbox_target_to_loc_probs(...
    bbox_in, bbox_target, conf)
% encode_bbox_target_to_loc_probs: given input bounding boxes (bbox_in),
% target bounding boxes and the configuration parameters of LocNet (conf) 
% it computes the target probability vectors that the LocNet should predict
% for each input bounding box.
% 
% INPUTS:
% 1) bbox_in: a N x 4 array with the input bounding box coordinates in the 
% form of [xi0,yi0,xi1,yi1] where (xi0,yi0) is the tot-left corner and  
% (xi1,yi1) is the bottom-right corner. N is the number of bounding boxes.
% 2) bbox_target: a N x 5 array with the target bounding box coordinates in the 
% form of [xt0,yt0,xt1,yt1,c] where (xt0,yt0) is the tot-left corner, (xt1,yt1)
% is the bottom-right corner, and c is the category id of the bounding box.
% 3) conf: struct with the configuration parameters of LocNet:
%   3.1) conf.scale_ratio: (scalar value) the scaling factor of the search region
%        w.r.t. the input bounding boxes. Specifically, given an input 
%        bounding boxes, in order for LocNet to localize the target 
%        bounding box it "look" in a search region that is obtained by scaling
%        the input bounding box by a scaling factor. The scaling factor is 
%        given by the scale_ratio parameter.
%   3.2) conf.resolution: scalar value; In order for the LocNet to localize 
%   the target bounding boxes inside the search region, it considers a division 
%   of the search region in M horizontal stripes (rows) and M vertical stripes 
%   (columns). The value of M is given by the parameter resolution.
%   3.3) conf.num_classes: (scalar value) number of categories
%   3.4) conf.loc_type: string wiht the type of the localization model that  
%   the LocNet implements. The possible options are: 'inout','borders', or 'combined'
%
% OUTPUTS:
% 1) loc_prob_vectors: a M x K x C x N array with the predicted probability
% vectors of each input bounding box; N is the number of bounding boxes, C
% is the number of categoris, M is the resolution of the target
% probabilities specified by the input parameter conf.resolution and K is
% the number of target probability vectors per bounding box and per
% category. For the InOut model K = 2 (1 vector for each axis), for the
% Border model K = 4 (2 vectors for each axis; e.g. for the x axis we have 
% 1 probability vector for the left border and 1 probability vector for the 
% right bordr), and for the Combined model K = 6 (3 vectors for each axis; 
% e.g. for the x axis we have 1 probability vector for the in-out elements, 
% 1 probability vector for the left border, and 1 probability vector 
% for the right border).
% 2) bbox_target_quantized: a N x 5 array with the target bounding box  
% coordinates in the form of [xtq0,ytq0,xtq1,ytq1] where (xtq0,ytq0) is 
% the tot-left corner, (xtq1,ytq1) is the bottom-right corner, and c is the
% category id of the bounding box. The coordinates are w.r.t. the 
% corresponding search regions and quantized in discrite values between 1 
% and resolution.
% 3) bbox_target_in_region: a N x 5 array with the target bounding box  
% coordinates in the form of [xtr0,ytr0,xtr1,ytr1] where (xtr0,ytr0) is 
% the tot-left corner, (xtr1,ytr1) is the bottom-right corner, and c is the
% category id of the bounding box. The coordinates are w.r.t. the 
% corresponding search regions.
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

scale_ratio = conf.scale_ratio;
resolution  = conf.resolution;
num_classes = conf.num_classes;
loc_type    = conf.loc_type;

% get the search regions
bbox_search_region = scale_bboxes(bbox_in, scale_ratio);

% get the coordinates of the target bounding boxes w.r.t. the search
% regions and quantized in discrite values between 1 and resolution (in 
% the paper the resolution parameter is referred as M)
[bbox_target_quantized, bbox_target_in_region] = map_target_bbox_to_search_region(bbox_search_region, bbox_target, resolution);

% compute the target probability vectors
switch loc_type
    case 'inout'
        loc_prob_vectors = encode_target_bbox_in_out(  bbox_target_quantized, resolution, num_classes);
    case 'borders'
        loc_prob_vectors = encode_target_bbox_borders( bbox_target_quantized, resolution, num_classes);
    case 'combined'
        loc_prob_vectors = encode_target_bbox_combined(bbox_target_quantized, resolution, num_classes);
    otherwise
        error('Invalid localization type %s',loc_type)
end

end

function [bbox_target_quantized, bbox_target_in_region] = map_target_bbox_to_search_region(...
    bbox_search_region, bbox_target, resolution)
region_width  = bbox_search_region(:,3)-bbox_search_region(:,1)+1;
region_height = bbox_search_region(:,4)-bbox_search_region(:,2)+1;

% the target bounding box coordinates w.r.t. the search region bbox_search_region
bbox_target_in_region = bbox_target;
bbox_target_in_region(:,1) = max(1,min(region_width,   bbox_target_in_region(:,1)-bbox_search_region(:,1)+1));
bbox_target_in_region(:,3) = max(1,min(region_width,   bbox_target_in_region(:,3)-bbox_search_region(:,1)+1));
bbox_target_in_region(:,2) = max(1,min(region_height,  bbox_target_in_region(:,2)-bbox_search_region(:,2)+1));
bbox_target_in_region(:,4) = max(1,min(region_height,  bbox_target_in_region(:,4)-bbox_search_region(:,2)+1));

% quantize the target bounding box coordinates w.r.t. the search region bbox_search_region
% in discrite values between 1 and resolution (in the paper the resolution parameter is referred as M)
bbox_target_quantized      = bbox_target_in_region;
bbox_target_quantized(:,1) = round((bbox_target_quantized(:,1)-0.5) .* (resolution./region_width))  + 1;
bbox_target_quantized(:,2) = round((bbox_target_quantized(:,2)-0.5) .* (resolution./region_height)) + 1;
bbox_target_quantized(:,3) = round((bbox_target_quantized(:,3)-0.5) .* (resolution./region_width))  + 1;
bbox_target_quantized(:,4) = round((bbox_target_quantized(:,4)-0.5) .* (resolution./region_height)) + 1;

bbox_target_quantized = single(bbox_target_quantized);
end

function loc_prob_vectors = encode_target_bbox_in_out(bbox_target_quantized, resolution, num_classes)
class_indices = bbox_target_quantized(:,5);
bbox_target_quantized = max(1,min(resolution, bbox_target_quantized(:,1:4)));

num_bboxes = size(bbox_target_quantized,1);
loc_prob_vectors   = zeros([resolution, 2, num_classes, num_bboxes], 'single');

bbox_target_quantized = bbox_target_quantized';
for b = 1:num_bboxes
    class_idx = class_indices(b);
    % IN-OUT TARGET PROBABILITIES FOR THE X AXIS
    loc_prob_vectors(bbox_target_quantized(1,b):bbox_target_quantized(3,b),1,class_idx,b) = 1;
    % IN-OUT TARGET PROBABILITIES FOR THE Y AXIS
    loc_prob_vectors(bbox_target_quantized(2,b):bbox_target_quantized(4,b),2,class_idx,b) = 1;
end

end

function loc_prob_vectors = encode_target_bbox_combined(bbox_target_quantized, resolution, num_classes)
class_indices = bbox_target_quantized(:,5);
bbox_target_quantized = max(1,min(resolution, bbox_target_quantized(:,1:4)));
num_bboxes = size(bbox_target_quantized,1);
loc_prob_vectors = zeros([resolution, 6, num_classes, num_bboxes], 'single'); 

bbox_target_quantized = bbox_target_quantized';
for b = 1:num_bboxes
    class_idx = class_indices(b);
    % IN-OUT TARGET PROBABILITIES FOR THE X AXIS
    loc_prob_vectors(bbox_target_quantized(1,b):bbox_target_quantized(3,b),1,class_idx,b) = 1;
    % IN-OUT TARGET PROBABILITIES FOR THE Y AXIS
    loc_prob_vectors(bbox_target_quantized(2,b):bbox_target_quantized(4,b),4,class_idx,b) = 1;
    
    % BORDERS TARGET PROBABILITIES LEFT 
    loc_prob_vectors(bbox_target_quantized(1,b),2,class_idx,b) = 1;
    % BORDERS TARGET PROBABILITIES RIGHT
    loc_prob_vectors(bbox_target_quantized(3,b),3,class_idx,b) = 1;
    % BORDERS TARGET PROBABILITIES TOP
    loc_prob_vectors(bbox_target_quantized(2,b),5,class_idx,b) = 1;
    % BORDERS TARGET PROBABILITIES BOTTOM
    loc_prob_vectors(bbox_target_quantized(4,b),6,class_idx,b) = 1;     
end
end

function loc_prob_vectors = encode_target_bbox_borders(bbox_target_quantized, resolution, num_classes)
class_indices = bbox_target_quantized(:,5);
bbox_target_quantized = max(1,min(resolution, bbox_target_quantized(:,1:4)));
num_bboxes = size(bbox_target_quantized,1);
loc_prob_vectors = zeros([resolution, 4, num_classes, num_bboxes], 'single'); 

bbox_target_quantized = bbox_target_quantized';
for b = 1:num_bboxes
    class_idx = class_indices(b);
      
    % BORDERS TARGET PROBABILITIES LEFT 
    loc_prob_vectors(bbox_target_quantized(1,b),1,class_idx,b) = 1;
    % BORDERS TARGET PROBABILITIES RIGHT
    loc_prob_vectors(bbox_target_quantized(3,b),2,class_idx,b) = 1;
    % BORDERS TARGET PROBABILITIES TOP
    loc_prob_vectors(bbox_target_quantized(2,b),3,class_idx,b) = 1;
    % BORDERS TARGET PROBABILITIES BOTTOM
    loc_prob_vectors(bbox_target_quantized(4,b),4,class_idx,b) = 1;  
end
end