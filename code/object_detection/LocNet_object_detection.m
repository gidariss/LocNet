function [ bbox_detections ] = LocNet_object_detection( ...
    img, model_obj_rec, model_obj_loc, bbox_proposals, conf )
% LocNet_object_detection: given an image, a recognition model for 
% scoring candidate detection boxes, a bounding box localilization model
% (e.g. LocNet or CNN-based bounding box regression) for refining the 
% bounding box coordinates, and an initial set of class-agnostic bounding 
% box proposals, it performs the object detection task by implementing the 
% object detection pipeline that is described on the paper:
% "LocNet: Improving Localization Accuracy for Object Detection"
% http://arxiv.org/abs/1511.07763
% 
% INPUT:
% 1) img: a H x W x 3 uint8 matrix that contains the image pixel values
% 2) model_obj_rec: a struct with the object recognition model
% 3) model_obj_loc: a struct with the bounding box localization model
% 4) bbox_proposals: is a NB X 4 array that contains the candidate boxes 
% that will be given as input to the object detection pipeline. The i-th  
% row of it contains the cordinates [x0, y0, x1, y1] of the i-th candidate 
% box, where the (x0,y0) and (x1,y1) are coorindates of the top-left and 
% bottom-right corners correspondingly. NB is the number of box proposals.
% 5) conf: a struct that must contain the following fields:
%    a) conf.nms_iou_thrs: scalar value with the IoU threshold that will be
%       used during the non-max-suppression step that is applied at post
%       processing time.
%    b) conf.thresh: is a C x 1 array, where C is the number of categories.
%       It must contain the threshold per category that will be used for 
%       removing candidate boxes with low confidence prior to applying the 
%       non-max-suppression step at post processing time.
%    c) conf.num_iterations: scalar value with the number of iterations
%       that the iterative localization scheme is performed. 
%    d) conf.thresh_init  is a C x 1 array, where C is the number of categories.
%       It must contain the threshold per category that will be used in
%       order to prune the candidate boxes with low confidence only at the
%       first iteration of the object detection pipeline (in order to 
%       remove the computation cost of subsequent steps). 
%    e) conf.nms_iou_thrs_init scalar value with the IoU threshold that 
%       will be used during the non-max-suppression step that is applied
%       at the first iteration of the object detection pipeline in order to
%       remove near duplicate box proposals (the typical value for this 
%       parameter is 0.95).
%       
% OUTPUT:
% 1) bbox_detections: is a C x 1 cell array with the object detection where
% C is the number of categories. The i-th element of bbox_detections is a
% ND_i x 5 matrix arrray with object detection of the i-th category. Each row
% contains the following values [x0, y0, x1, y1, scr] where the (x0,y0) and 
% (x1,y1) are coorindates of the top-left and bottom-right corners
% correspondingly. scr is the confidence score assigned to the bounding box
% detection and ND_i is the number of detections.
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

category_names = model_obj_rec.classes; % a C x 1 cell array with the name 
% of the categories that the detection system looks for. C is the number of
% categories.
num_categories = length(category_names);
max_per_image = 100;

%********************** ITERATIVE LOCALIZATION SCHEME *********************
bbox_cand_dets_per_iter = cell(conf.num_iterations,1);
for iter = 1:conf.num_iterations
    % score the bounding box proposals with the recognition model;
    % bboxes_scores will be a NB x C array, where NB is the number of box 
    % proposals and C is the number of categories.
    bboxes_scores = recognize_bboxes_of_image(model_obj_rec, img, bbox_proposals(:,1:4));
    if (iter == 1)
        % For each category prune the candidate detection boxes that have
        % low confidence score or are near duplicate candidate boxes
        [bbox_proposals, bboxes_scores] = prune_candidate_boxes_with_low_confidence(...
            bbox_proposals, bboxes_scores, conf.thresh_init, conf.nms_iou_thrs_init, max_per_image);  
        % After the above operation there will be a different set of 
        % candidate detection boxes per category. Hence, at this point 
        % bbox_proposals will be a NBB x 5 array where the first 4 columns 
        % contain the coordinates of the bounding boxes that survived the above 
        % pruning operation and the 5-th column contains the category ids of the bounding boxes.
        % bboxes_scores will a NBB x 1 array with the confidence score of each
        % bounding box w.r.t. its category.
    else % iter > 1
        % Note that for iter > 1 there is a different set of box proposals
        % per category
        category_indices = bbox_proposals(:,5); % the category id of each box proposal
        
        % get the confidence score of each box proposal w.r.t. its category
        bboxes_scores_per_class = cell(num_categories,1);
        for c = 1:num_categories
            bboxes_scores_per_class{c} = bboxes_scores(category_indices==c,c);
        end
        bboxes_scores = cell2mat(bboxes_scores_per_class); 
        % bboxes_scores is a NBB x 1 array with the confidence score of 
        % each box proposal w.r.t. its category.
    end
    
    bbox_cand_dets_per_iter{iter} = prepare_bbox_cand_dets(...
        bbox_proposals, bboxes_scores, num_categories);
    % bbox_cand_dets_per_iter{iter} is a C x 1 cell array with the candidate 
    % detection boxes of each of the C categories that were generated during 
    % the iter-th iteration. The i-th element of this cell array is NBB_i x 5
    % array with the candidate detections of the i-th category; the first 4 
    % columns contain the bouding box coordinates and the 5-th column
    % contains the confidence score of each bounding box with respect to
    % the i-th category. NBB_i is the number of candidate detections that
    % were generated during the iter-th iteration for the i-th category.
    
    if iter < conf.num_iterations
        % predict a new bounding box for each box proposal that ideally it will
        % be better localized on an object of the same category as the box proposal.
        bbox_refined = localize_bboxes_of_image(model_obj_loc, img, bbox_proposals);
        % bbox_refined is a NBB x 5 array where the first 4 columns contain the
        % refined bounding box coordinates and the 5-th column contains the
        % category id of each bounding box.
        
        bbox_proposals = bbox_refined;
    end
end
%**************************************************************************

%***************************** POST PROCESSING ****************************
% For each category merge the candidate bounding box detections of each 
% iteration to a single set
bbox_cand_dets  = merge_bbox_cand_dets_of_all_iters(bbox_cand_dets_per_iter);

% Apply the non-max-suppression with box voting step
conf.do_bbox_voting     = true; % do bounding box voting
conf.box_ave_iou_thresh = 0.5; % the IoU threshold for the neighboring bounding boxes
conf.add_val            = 1.5; % this value is added to the bounding box scores 
                               % before they are used as weight during the box voting step                              
bbox_detections = post_process_candidate_detections( bbox_cand_dets, ...
    'thresholds', conf.thresh, 'nms_iou_thrs', conf.nms_iou_thrs, ...
    'max_per_image', max_per_image, 'do_bbox_voting', conf.do_bbox_voting, ...
    'box_ave_iou_thresh', conf.box_ave_iou_thresh, 'add_val', conf.add_val,...
    'use_gpu',true);
%**************************************************************************

end

function [bbox_proposals, bboxes_scores] = prune_candidate_boxes_with_low_confidence(...
    bbox_proposals, bboxes_scores, thresh_init, nms_over_thrs_init, max_per_image)
bbox_cand_dets = single([bbox_proposals(:,1:4), bboxes_scores]);
% For each category prune the candidate detection boxes with low 
% confidence score or near duplicate detection boxes (IoU > nms_over_thrs_init)
bbox_cand_dets = post_process_candidate_detections( bbox_cand_dets, ...
    'thresholds', thresh_init, 'nms_iou_thrs', nms_over_thrs_init, ...
    'max_per_image', max_per_image);   
% After the above operation there will be a different set of 
% candidate detection boxes per category. Hence, bbox_cand_dets
% will be a C x 1 cell array with the candidate detection boxes of 
% each of the C categories.

% reformulate the bbox_cand_dets cell array
[bbox_proposals, bboxes_scores] = merge_bbox_cand_of_all_classes(bbox_cand_dets);
% bbox_proposals is a NB x 5 array with the candidate detection
% boxes of all the categories together. The first 4 columns contain 
% the coordinates of the bounding boxes and the 5-th column contains
% the category id of each bounding box. 
% bboxes_scores is a NB x 1 array with the confidence score of each
% bounding box w.r.t. its category id.
end

function bbox_cand_dets = merge_bbox_cand_dets_of_all_iters(bbox_cand_dets_per_iter)
num_iterations = length(bbox_cand_dets_per_iter);
num_categories = length(bbox_cand_dets_per_iter{1});
bbox_cand_dets = cell(num_categories,1);
for j = 1:num_categories
    bbox_cand_dets_per_iter_this_cls = cell(num_iterations, 1);
    for iter = 1:num_iterations
        bbox_cand_dets_per_iter_this_cls{iter} = bbox_cand_dets_per_iter{iter}{j};
    end
    bbox_cand_dets{j} = cell2mat(bbox_cand_dets_per_iter_this_cls);
end
end

function [bbox_proposals, bbox_scores] = merge_bbox_cand_of_all_classes(bbox_cand_dets)
bbox_cand_dets = bbox_cand_dets(:);
bbox_proposals_per_class = cellfun(@(x) x(:,1:4), bbox_cand_dets, 'UniformOutput', false);
class_indices = [];
for c = 1:length(bbox_cand_dets)
    class_indices = [class_indices; ones(size(bbox_cand_dets{c},1),1,'single')*c];
end
bbox_scores_per_class = cellfun(@(x) x(:,5:end), bbox_cand_dets, 'UniformOutput', false);


num_bbox_per_class  = cellfun(@(x) size(x,1), bbox_proposals_per_class,  'UniformOutput', true);
bbox_proposals      = cell2mat(bbox_proposals_per_class(num_bbox_per_class>0));
bbox_scores         = cell2mat(bbox_scores_per_class(num_bbox_per_class>0));
bbox_proposals      = [bbox_proposals, class_indices];
end

function bbox_cand_dets = prepare_bbox_cand_dets(bbox_coordinates, bbox_scores, num_classes)
class_indices      = bbox_coordinates(:,5);
bbox_cand_dets_tmp = single([bbox_coordinates(:,1:4), bbox_scores]);
bbox_cand_dets     = cell(num_classes,1);
for c = 1:num_classes
    bbox_cand_dets{c} = bbox_cand_dets_tmp(class_indices==c,:);
    if isempty(bbox_cand_dets{c}), bbox_cand_dets{c} = zeros(0,5,'single'); end
end
end