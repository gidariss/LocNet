function [ap_results05,ap_results07,ap_resultsCOCO,mean_recall_per_IoU] = print_detection_evaluation_metrics(...
    all_bbox_gt, abbox_dets, classes)
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

% evaluate the average precision for IoU >=0.5 
ap_results05 = evaluate_average_precision_pascal(all_bbox_gt, abbox_dets, classes );
fprintf('AP with IoU >= 0.5\n');
printAPResults(classes, ap_results05);
% evaluate the average precision for IoU >=0.7
ap_results07 = evaluate_average_precision_pascal(all_bbox_gt, abbox_dets, classes, 'minoverlap', 0.7);
fprintf('AP with IoU >= 0.7\n');
printAPResults(classes, ap_results07);
% evaluate the coco style average precision
ap_resultsCOCO = evaluate_average_precision_pascal(all_bbox_gt, abbox_dets, classes, 'coco_style', true); 
fprintf('AP with coco style\n');
printAPResults(classes, ap_resultsCOCO);


num_classes = length(classes);
all_bbox_gt_no_difficults = remove_difficult_ground_truth_bboxes(all_bbox_gt);
all_bbox_gt_per_class = get_per_class_ground_truth_bboxes(all_bbox_gt_no_difficults, num_classes);
    
ave_recall_per_class = zeros(num_classes, 1);
recall_per_class     = cell(1,num_classes);

% compute the recall per IoU threshold and the average recall of the set of
% detections for each class separately 
for j = 1:num_classes
    [ave_recall_per_class(j), recall_per_class{j}, thresholds] = ...
        compute_ave_recall_of_bbox(abbox_dets{j}, all_bbox_gt_per_class{j});
    recall_per_class{j} = recall_per_class{j}(:);
    recall_per_class{j} = recall_per_class{j}(thresholds>=0.5);
end
thresholds = thresholds(thresholds>=0.5);
fprintf('Average Recall of detections :\n');
printAPResults(classes, ave_recall_per_class);
fprintf('Recall per IoU of detections (averaged over the classes):\n');
mean_recall_per_IoU = mean(cell2mat(recall_per_class),2); % average over the classes
thresholds_string = strsplit(num2str(thresholds));
assert(length(mean_recall_per_IoU) == length(thresholds_string));
printAPResults(thresholds_string, mean_recall_per_IoU);
end