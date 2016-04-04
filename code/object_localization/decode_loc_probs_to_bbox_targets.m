function [bbox_pred, bbox_pred_in_region, bbox_pred_in_region_quantized] = decode_loc_probs_to_bbox_targets(...
    bbox_in, class_indices, loc_prob_vectors, conf)
% decode_loc_probs_to_bbox_targets: given the input bounding boxes (bbox_in),
% the category ids of each bounding box (class_indices), and the predicted
% probability vectors (loc_prob_vectors) of each input bounding box it 
% returns the location of the predicted bounding boxes.
% 
% INPUTS:
% 1) bbox_in: a N x 4 array with the input bounding box coordinates in the 
% form of [xi0,yi0,xi1,yi1] where (xi0,yi0) is the tot-left corner and  
% (xi1,yi1) is the bottom-right corner. N is the number of bounding boxes.
% 2) class_indices: a N x 1 array with the category id of each input
% bounding box.
% 3) loc_prob_vectors: a M x K x C x N array with the predicted probability
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
% 4) conf: struct with the configuration parameters of LocNet:
%   4.1) conf.scale_ratio: (scalar value) the scaling factor of the search region
%        w.r.t. the input bounding boxes. Specifically, given an input 
%        bounding boxes, in order for LocNet to localize the target 
%        bounding box it "look" in a search region that is obtained by scaling
%        the input bounding box by a scaling factor. The scaling factor is 
%        given by the scale_ratio parameter.
%   4.2) conf.resolution: scalar value; In order for the LocNet to localize 
%   the target bounding boxes inside the search region, it considers a division 
%   of the search region in M horizontal stripes (rows) and M vertical stripes 
%   (columns). The value of M is given by the parameter conf.resolution.
%   4.3) conf.num_classes: (scalar value) number of categories
%   4.4) conf.loc_type: string wiht the type of the localization model that  
%   the LocNet implements. The possible options are: 'inout','borders', or 'combined'
%
% OUTPUTS:
% 1) bbox_pred: a N x 5 array with the predicted bounding box coordinates in the 
% form of [xt0,yt0,xt1,yt1,c] where (xt0,yt0) is the tot-left corner, (xt1,yt1)
% is the bottom-right corner, and c is the category id of the bounding box.
% 3) bbox_pred_in_region: a N x 5 array with the predicted bounding box  
% coordinates in the form of [xtr0,ytr0,xtr1,ytr1] where (xtr0,ytr0) is 
% the tot-left corner, (xtr1,ytr1) is the bottom-right corner, and c is the
% category id of the bounding box. The coordinates are w.r.t. the 
% corresponding search regions.
% 2) bbox_target_quantized: a N x 5 array with the predicted bounding box  
% coordinates in the form of [xtq0,ytq0,xtq1,ytq1] where (xtq0,ytq0) is 
% the tot-left corner, (xtq1,ytq1) is the bottom-right corner, and c is the
% category id of the bounding box. The coordinates are w.r.t. the 
% corresponding search regions and quantized in discrite values between 1 
% and resolution.
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

resolution  = conf.resolution;
scale_ratio = conf.scale_ratio;
loc_type    = conf.loc_type;

assert(size(loc_prob_vectors,1) == conf.resolution);

% get the coordinates of the search regions
bbox_search_region = scale_bboxes(bbox_in, scale_ratio);

% given the predicted probability vectors and the category ids, estimate 
% the most likely bounding box location of the objects of interest; the 
% predicted bounding boxes locations are w.r.t. the search region 
% bbox_search_region and quantized in discrite values between 1 and 
% resolution (in the paper the resolution parameter is referred as M)
switch loc_type
    case 'inout'
        bbox_pred_in_region_quantized = decode_in_out_probabilities(loc_prob_vectors, class_indices);
    case 'borders'
        bbox_pred_in_region_quantized = decode_borders_probabilities(loc_prob_vectors, class_indices);    
    case 'combined'
        bbox_pred_in_region_quantized = decode_combined_probabilities(loc_prob_vectors, class_indices);
    otherwise
        error('Invalid localization type %s',loc_type)
end

% get the coordinates of the predicted bounding boxes w.r.t. the original
% image and in pixel units. 
[bbox_pred, bbox_pred_in_region] = decode_quantized_location(...
    bbox_pred_in_region_quantized, resolution, bbox_search_region);
bbox_pred = [bbox_pred, class_indices]; 
end

function [real_coord, real_coord_region] = decode_quantized_location(quantized, resolution, bbox_search_region)
search_width  = bbox_search_region(:,3)-bbox_search_region(:,1)+1;
search_height = bbox_search_region(:,4)-bbox_search_region(:,2)+1;

% transform the coordinates of the predicted bounding boxes in pixel units
real_coord_region      = quantized;
real_coord_region(:,1) = (quantized(:,1)-1).*(search_width /resolution) + 0.5;
real_coord_region(:,2) = (quantized(:,2)-1).*(search_height/resolution) + 0.5;
real_coord_region(:,3) = (quantized(:,3)-1).*(search_width /resolution) + 0.5;
real_coord_region(:,4) = (quantized(:,4)-1).*(search_height/resolution) + 0.5;

% get the coordinates of the predicted bounding boxes w.r.t. the image
real_coord      = real_coord_region;
real_coord(:,1) = real_coord_region(:,1)+bbox_search_region(:,1);
real_coord(:,3) = real_coord_region(:,3)+bbox_search_region(:,1);
real_coord(:,2) = real_coord_region(:,2)+bbox_search_region(:,2);
real_coord(:,4) = real_coord_region(:,4)+bbox_search_region(:,2);
end

function best_location = decode_in_out_probabilities(loc_maps_per_cls, class_indices)

% keep from each input bounding box the probability vectors that correspond
% to its category id
loc_maps_size = size(loc_maps_per_cls);
assert(max(class_indices) <= loc_maps_size(3));
loc_maps_size(3) = 1;
loc_prob_vectors = zeros(loc_maps_size,'single');
unique_class_indices = unique(class_indices);
for i = 1:length(unique_class_indices)
    class_idx   = unique_class_indices(i);
    is_this_cls = class_idx == class_indices;
    loc_prob_vectors(:,:,1,is_this_cls) = loc_maps_per_cls(:,:,class_idx,is_this_cls);
end
loc_prob_vectors = squeeze(loc_prob_vectors);

% given the predicted probability vectors, for each input bounding box, 
% estimate the most likely location (by minimizing the negative log likelihood)
% of bounding box of the actual object independently for the x and y axis.
[best_locationsX] = minimizeNegLogLikelihoodInOut(loc_prob_vectors(:,1,:)); % x axis
[best_locationsY] = minimizeNegLogLikelihoodInOut(loc_prob_vectors(:,2,:)); % y axis
best_location = single([best_locationsX(:,1),best_locationsY(:,1),best_locationsX(:,2),best_locationsY(:,2)]);
end

function best_location = decode_combined_probabilities(loc_maps_per_cls, class_indices)

% keep from each input bounding box the probability vectors that correspond
% to its category id
loc_maps_size = size(loc_maps_per_cls);
assert(max(class_indices) <= loc_maps_size(3));
loc_maps_size(3) = 1;
loc_prob_vectors = zeros(loc_maps_size,'single');
unique_class_indices = unique(class_indices);
for i = 1:length(unique_class_indices)
    class_idx   = unique_class_indices(i);
    is_this_cls = class_idx == class_indices;
    loc_prob_vectors(:,:,1,is_this_cls) = loc_maps_per_cls(:,:,class_idx,is_this_cls);
end
loc_prob_vectors = squeeze(loc_prob_vectors);

% given the predicted probability vectors, for each input bounding box, 
% estimate the most likely location (by minimizing the negative log likelihood)
% of bounding box of the actual object independently for the x and y axis.
[best_locationsX] = minimizeNegLogLikelihoodCombined(loc_prob_vectors(:,1:3,:)); % x axis
[best_locationsY] = minimizeNegLogLikelihoodCombined(loc_prob_vectors(:,4:6,:)); % y axis
best_location = single([best_locationsX(:,1),best_locationsY(:,1),best_locationsX(:,2),best_locationsY(:,2)]);
end

function best_location = decode_borders_probabilities(loc_maps_per_cls, class_indices)

% keep from each input bounding box the probability vectors that correspond
% to its category id
loc_maps_size = size(loc_maps_per_cls);
assert(max(class_indices) <= loc_maps_size(3));
loc_maps_size(3) = 1;
loc_prob_vectors = zeros(loc_maps_size,'single');
unique_class_indices = unique(class_indices);
for i = 1:length(unique_class_indices)
    class_idx   = unique_class_indices(i);
    is_this_cls = class_idx == class_indices;
    loc_prob_vectors(:,:,1,is_this_cls) = loc_maps_per_cls(:,:,class_idx,is_this_cls);
end
loc_prob_vectors = squeeze(loc_prob_vectors);

% given the predicted probability vectors, for each input bounding box, 
% estimate the most likely location (by minimizing the negative log likelihood)
% of bounding box of the actual object independently for the x and y axis.
[best_locationsX] = minimizeNegLogLikelihoodBorders(loc_prob_vectors(:,1:2,:)); % x axis
[best_locationsY] = minimizeNegLogLikelihoodBorders(loc_prob_vectors(:,3:4,:)); % y axis
best_location = single([best_locationsX(:,1),best_locationsY(:,1),best_locationsX(:,2),best_locationsY(:,2)]);
end

function [best_location, MinNegLogLikelihood, NegLogLikelihoodPerLocation, all_locations] = minimizeNegLogLikelihoodCombined(Probs)
% Given the predicted Combined probability vectors for a single dimension
% (either the x or y axis), estimate the most likely location of the actual
% bounding box in the corresponding axis by minimizing the negative
% log-likelihood. For the mimization we use exaustive search.


min_prob         = 0.0001;
PositiveProbs = max(  Probs,min_prob);
NegativeProbs = max(1-Probs,min_prob);

LogInsideProbs   = -log(squeeze(PositiveProbs(:,1,:)));
LogBordersProbs0 = -log(squeeze(PositiveProbs(:,2,:)))';
LogBordersProbs1 = -log(squeeze(PositiveProbs(:,3,:)))';

LogOutsideProbs     = -log(squeeze(NegativeProbs(:,1,:)));
LogNonBordersProbs0 = -log(squeeze(NegativeProbs(:,2,:)))';
LogNonBordersProbs1 = -log(squeeze(NegativeProbs(:,3,:)))';

resolution            = size(LogInsideProbs,1);
num_elems             = size(LogOutsideProbs,2);
LogInsideProbsCumSum  = cumsum([zeros(1,num_elems,'single');LogInsideProbs], 1)';
LogOutsideProbsCumSum = cumsum([zeros(1,num_elems,'single');LogOutsideProbs],1)';

NegLogLikelihoodPerLocation = zeros([num_elems, resolution*resolution],'single');
all_locations = zeros([resolution*resolution, 2],'single');
c = 0;
% compute the negative log-likelihood of each possible location of the
% bounding box
for a = 1:(resolution-1) % the double loop iterates over all possible (discrete) locations of the bounding box
    for b = (a+1):resolution
        c = c + 1;
        % NegLogLikelihoodPerLocation(:,c) is the negative log-likelihood of 
        % the bounding box with coordines on [a, b] in the x (or y) axis. 
        % For instance, in the x axis, a is the location of the left border
        % of the bounding box and b is the location of the right border of 
        % the bounding box.   
        NegLogLikelihoodPerLocation(:,c) = (LogInsideProbsCumSum(:,b+1)-LogInsideProbsCumSum(:,a))-...
            (LogOutsideProbsCumSum(:,b+1)-LogOutsideProbsCumSum(:,a)) + ...
            (LogBordersProbs0(:,a) + LogBordersProbs1(:,b)) - ...
            (LogNonBordersProbs0(:,a) + LogNonBordersProbs1(:,b));          
        all_locations(c,1) = a;
        all_locations(c,2) = b;
    end
end
all_locations = all_locations(1:c,:);
NegLogLikelihoodPerLocation = NegLogLikelihoodPerLocation(:,1:c);

% pick the location with the minimum negative log-likelihood
[MinNegLogLikelihood, min_idx] = min(NegLogLikelihoodPerLocation,[],2);
best_location = all_locations(min_idx,:);
end

function [best_location, MinNegLogLikelihood, NegLogLikelihoodPerLocation, all_locations] = minimizeNegLogLikelihoodInOut(Probs)
% Given the predicted InOut probability vectors for a single dimension
% (either the x or y axis), estimate the most likely location of the actual
% bounding box in the corresponding axis by minimizing the negative
% log-likelihood. For the mimization we use exaustive search. 

min_prob         = 0.0001;
InsideProbs      = max(  Probs,min_prob);
OutsideProbs     = max(1-Probs,min_prob);
LogInsideProbs   = -log(squeeze(InsideProbs(:,1,:)));
LogOutsideProbs  = -log(squeeze(OutsideProbs(:,1,:)));

resolution            = size(LogInsideProbs,1);
num_elems             = size(LogOutsideProbs,2);
LogInsideProbsCumSum  = cumsum([zeros(1,num_elems,'single');LogInsideProbs], 1)';
LogOutsideProbsCumSum = cumsum([zeros(1,num_elems,'single');LogOutsideProbs],1)';

NegLogLikelihoodPerLocation = zeros([num_elems, resolution*resolution],'single');
all_locations = zeros([resolution*resolution, 2],'single');
c = 0;
% compute the negative log-likelihood of each possible location of the
% bounding box
for a = 1:(resolution-1) % the double loop iterates over all possible (discrete) locations of the bounding box
    for b = (a+1):resolution
        c = c + 1;
        % NegLogLikelihoodPerLocation(:,c) is the negative log-likelihood of 
        % the bounding box with coordines on [a, b] in the x (or y) axis. 
        % For instance, in the x axis, a is the location of the left border
        % of the bounding box and b is the location of the right border of 
        % the bounding box.        
        NegLogLikelihoodPerLocation(:,c) = (LogInsideProbsCumSum(:,b+1)-LogInsideProbsCumSum(:,a))-...
            (LogOutsideProbsCumSum(:,b+1)-LogOutsideProbsCumSum(:,a));
        all_locations(c,1) = a;
        all_locations(c,2) = b;
    end
end
all_locations = all_locations(1:c,:);
NegLogLikelihoodPerLocation = NegLogLikelihoodPerLocation(:,1:c);

% pick the location with the minimum negative log-likelihood
[MinNegLogLikelihood, min_idx] = min(NegLogLikelihoodPerLocation,[],2);
best_location = all_locations(min_idx,:);
end

function [best_location, MinNegLogLikelihood, NegLogLikelihoodPerLocation, all_locations] = minimizeNegLogLikelihoodBorders(Probs)
% Given the predicted Borders probability vectors for a single dimension
% (either the x or y axis), estimate the most likely location of the actual
% bounding box in the corresponding axis by minimizing the negative
% log-likelihood. For the mimization we use exaustive search.

min_prob         = 0.0001;
PositiveProbs    = max(  Probs,min_prob);
LogBordersProbs0 = -log(squeeze(PositiveProbs(:,1,:)))';
LogBordersProbs1 = -log(squeeze(PositiveProbs(:,2,:)))';

resolution = size(LogBordersProbs0,2);
num_elems = size(LogBordersProbs0,1);

NegLogLikelihoodPerLocation = zeros([num_elems, resolution*resolution],'single');
all_locations = zeros([resolution*resolution, 2],'single');
c = 0;
% compute the negative log-likelihood of each possible location of the
% bounding box
for a = 1:(resolution-1) % the double loop iterates over all possible (discrete) locations of the bounding box
    for b = (a+1):resolution
        c = c + 1;
        % NegLogLikelihoodPerLocation(:,c) is the negative log-likelihood of 
        % the bounding box with coordines on [a, b] in the x (or y) axis. 
        % For instance, in the x axis, a is the location of the left border
        % of the bounding box and b is the location of the right border of 
        % the bounding box.
        NegLogLikelihoodPerLocation(:,c) = (LogBordersProbs0(:,a) + LogBordersProbs1(:,b));        
        all_locations(c,1) = a;
        all_locations(c,2) = b;
    end
end
all_locations = all_locations(1:c,:);
NegLogLikelihoodPerLocation = NegLogLikelihoodPerLocation(:,1:c);

% pick the location with the minimum negative log-likelihood
[MinNegLogLikelihood, min_idx] = min(NegLogLikelihoodPerLocation,[],2);
best_location = all_locations(min_idx,:);
end