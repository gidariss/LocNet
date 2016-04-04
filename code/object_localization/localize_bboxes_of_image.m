function bboxes_out = localize_bboxes_of_image(model, image, bboxes_in)
% localize_bboxes_of_image given a localization model, an image and a set
% bounding boxes with the category id of each of them, it predicts a new
% location for each box such that the new ones will be closer (i.e. better
% localized) on the actual objects of the given categories.
%
% INPUTS:
% 1) model:  (type struct) the bounding box localization model
% 2) image:  a [Height x Width x 3] uint8 matrix with the image 
% 3) bboxes_in: a N x 5 array with the bounding box coordinates in the 
% form of [x0,y0,x1,y1,c] where (x0,y0) is the tot-left corner, (x1,y1) is 
% the bottom-right corner, and c is the category id of the bounding box. N 
% is the number of bounding boxes
%
% OUTPUT:
% 1) bboxes_out : a N x 5 array with the refined bounding boxes. It has the
% same format as bboxes_in
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

if isempty(bboxes_in)
    bboxes_out = zeros(0,5,'single');
    return;
end

% apply on the candidate bounding boxes and on the image the region-based 
% CNN network.
[outputs, out_blob_names] = run_region_based_net_on_img(model, image, bboxes_in);
% get the output blob that corresponds on the predicted localization values
idx = find(strcmp(out_blob_names,model.preds_loc_out_blob));
assert(numel(idx) == 1);
pred_localization_values = outputs{idx};

% decode the predicted localization values (pred_localization_values) to
% actual bounding box locations
loc_params = model.loc_params;
class_indices = bboxes_in(:,5);
switch loc_params.loc_type
    case 'bboxreg'
        % permute from [(4*num_classes) x NumBoxes] -> [NumBoxes x (4*num_classes)] 
        pred_localization_values = single(permute(pred_localization_values, [2, 1]));        
        % reshape from [NumBoxes x (4*num_classes)] -> [NumBoxes x 4 x num_classes]
        pred_localization_values = reshape(pred_localization_values, ...
            [size(pred_localization_values,1), 4, loc_params.num_classes]);
        bboxes_out = decode_reg_vals_to_bbox_targets(...
            bboxes_in(:,1:4), pred_localization_values, class_indices);
    case {'inout','borders','combined'}
        % decode the predicted localization values (pred_localization_values)
        % to the actual bounding box locations
        bboxes_out = decode_loc_probs_to_bbox_targets(...
            bboxes_in(:,1:4), class_indices, pred_localization_values, loc_params);       
    otherwise
        error('Invalid localization type %s',loc_type)
end

% post-process the predicted bounding boxes
img_size          = [size(image,1), size(image,2)];
bboxes_out(:,1:4) = clip_bbox_inside_the_img(bboxes_out(:,1:4), img_size);
bboxes_out(:,1:4) = check_box_coords(bboxes_out(:,1:4));
end

function bboxes = clip_bbox_inside_the_img(bboxes, img_size)
bboxes(:,1:4:end) = max(1,           bboxes(:,1:4:end));
bboxes(:,2:4:end) = max(1,           bboxes(:,2:4:end));
bboxes(:,3:4:end) = min(img_size(2), bboxes(:,3:4:end));
bboxes(:,4:4:end) = min(img_size(1), bboxes(:,4:4:end));
end

function bboxes = check_box_coords(bboxes)
assert(size(bboxes,2) == 4);
ind = bboxes(:,1) > bboxes(:,3);
if any(ind), bboxes(ind,[3,1]) = bboxes(ind,[1,3]); end
ind = bboxes(:,2) > bboxes(:,4);
if any(ind), bboxes(ind,[4,2]) = bboxes(ind,[2,4]); end
end