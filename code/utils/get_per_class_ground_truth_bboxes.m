function [all_bbox_gt_per_class, all_bbox_ids_per_class] = get_per_class_ground_truth_bboxes(...
    all_bbox_gt, num_classes)
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

num_imgs = numel(all_bbox_gt);
all_bbox_gt_per_class  = cell(num_classes, 1);
all_bbox_ids_per_class = cell(num_classes, 1);
for class_idx = 1:num_classes
    all_bbox_gt_per_class{class_idx} = cell(num_imgs, 1);
    for img_idx = 1:num_imgs
        is_this_class_mask  = find(all_bbox_gt{img_idx}(:,5) == class_idx); 
        all_bbox_gt_per_class{class_idx}{img_idx}  = single(all_bbox_gt{img_idx}(is_this_class_mask,1:4));
        all_bbox_ids_per_class{class_idx}{img_idx} = single(is_this_class_mask(:));
    end
end
end