function all_bbox_gt = remove_difficult_ground_truth_bboxes(all_bbox_gt)
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
for img_idx = 1:num_imgs
    if size(all_bbox_gt{img_idx},2) >= 6
        is_easy_mask  = all_bbox_gt{img_idx}(:,6) == 0; 
        all_bbox_gt{img_idx} = all_bbox_gt{img_idx}(is_easy_mask,:);
    end
end
end