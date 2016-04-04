function [mean_recall_per_IoU, mean_ave_recall, thresholds] = compute_recall_per_IoU_of_bboxes(all_bbox_gt, abbox, classes)
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

num_classes = length(classes);

if num_classes > 1
    all_bbox_gt           = remove_difficult_ground_truth_bboxes(all_bbox_gt);
    all_bbox_gt_per_class = get_per_class_ground_truth_bboxes(all_bbox_gt, num_classes);
    ave_recall_per_class  = zeros(num_classes, 1);
    recall_per_class      = cell(1,num_classes);

    % compute the recall per IoU threshold and the average recall of the 
    % set of candidate bounding boxes abbox for each class separately 
    for j = 1:num_classes
        [ave_recall_per_class(j), recall_per_class{j}, thresholds] = ...
            compute_ave_recall_of_bbox(abbox{j}, all_bbox_gt_per_class{j});
        recall_per_class{j} = recall_per_class{j}(:);
    end
    fprintf('Average Recall per class:\n')
    printAPResults(classes, ave_recall_per_class);
    mean_recall_per_IoU = mean(cell2mat(recall_per_class),2); % average over classes
    mean_ave_recall = mean(ave_recall_per_class); % average over classes
    fprintf('Average Recall: %.3f -- Recall(IoU>=0.4): %.3f -- Recall(IoU>=0.5): %.3f -- Recall(IoU>=0.7): %.3f\n', ...
        mean_ave_recall, mean_recall_per_IoU(thresholds == 0.4), mean_recall_per_IoU(thresholds == 0.5), mean_recall_per_IoU(thresholds == 0.7));
else
    % compute the recall per IoU threshold and the average recall of the 
    % set of candidate bounding boxes abbox.
    [ave_recall, recall_per_IoU, thresholds] = compute_ave_recall_of_bbox(abbox, all_bbox_gt);
    mean_recall_per_IoU = recall_per_IoU(:);  
    mean_ave_recall = ave_recall;
    fprintf('Average Recall: %.3f -- Recall(IoU>=0.4): %.3f -- Recall(IoU>=0.5): %.3f -- Recall(IoU>=0.7): %.3f\n',...
        mean_ave_recall, mean_recall_per_IoU(thresholds == 0.4), mean_recall_per_IoU(thresholds == 0.5), mean_recall_per_IoU(thresholds == 0.7));
end

end