function [ all_results, all_results_per_thr ] = evaluate_average_precision_pascal( ...
    all_bbox_gt, all_detected_bbox, classes, varargin)
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

%************************** OPTIONS *************************************
ip = inputParser;
ip.addParamValue('minoverlap', 0.5,   @isnumeric);
ip.addParamValue('coco_style', false, @islogical);
ip.addParamValue('penalize_duplicates', true, @islogical);

ip.parse(varargin{:});
opts = ip.Results;

minoverlap = opts.minoverlap;
all_results_per_thr = [];
if opts.coco_style
    minoverlap_list = 0.5:0.05:0.95;
    for m = 1:length(minoverlap_list)
        all_results_tmp = compute_average_precision_of_detection(...
            all_bbox_gt, all_detected_bbox, minoverlap_list(m), opts.penalize_duplicates);
        if m == 1
            all_results = all_results_tmp;
        else
            for i = 1:length(all_results)
                all_results(i).ap = [all_results(i).ap, all_results_tmp(i).ap];
            end
        end
    end
   all_results_per_thr = all_results;
   for i = 1:length(all_results)
       all_results(i).ap = mean(all_results(i).ap);
   end
else
    all_results = compute_average_precision_of_detection(...
        all_bbox_gt, all_detected_bbox, minoverlap, opts.penalize_duplicates);
end

% fprintf('\n~~~~~~~~~~~~~~~~~~~~\n');
% fprintf('Results:\n');
% aps = [all_results(:).ap]' * 100;
% disp(mean(aps));
% fprintf('~~~~~~~~~~~~~~~~~~~~\n');
end

function [false_positives, true_positives] = find_detection_labels(bbox_detections, bbox_gt, is_difficult, minoverlap)
false_positives = zeros(size(bbox_detections,1), 1);
true_positives  = zeros(size(bbox_detections,1), 1);
overlap         = zeros(size(bbox_detections,1), size(bbox_gt,1));
num_bbox_gt     = size(bbox_gt,1);

for j = 1:num_bbox_gt
    overlap(:,j) = boxoverlap(bbox_detections(:,1:4), bbox_gt(j,1:4));
end

overlap(overlap==0) = -inf;
[max_overlap, jmax] = max(overlap,[],2);

does_overlap = max_overlap >= minoverlap;

false_positives(~does_overlap) = 1; % false positive

does_overlap = does_overlap & ~is_difficult(jmax); % dont care about the difficult ones

if any(does_overlap) 
    bbox_indices = find(does_overlap);
    jmax         = jmax(does_overlap);

    for j = 1:num_bbox_gt
        overlap_j = bbox_indices(jmax == j);
        if ~isempty(overlap_j)
            true_positives(overlap_j(1)) = 1; % true positive
            if numel(overlap_j) > 1
                false_positives(overlap_j(2:end)) = 1; % false positive - multiple detections
            end
        end      
    end
end
end

function [false_positives, true_positives] = find_detection_labels_no_loc(bbox_detections, bbox_gt, is_difficult, minoverlap)
false_positives = zeros(size(bbox_detections,1), 1);
true_positives  = zeros(size(bbox_detections,1), 1);
overlap         = zeros(size(bbox_detections,1), size(bbox_gt,1));
num_bbox_gt     = size(bbox_gt,1);

for j = 1:num_bbox_gt
    overlap(:,j) = boxoverlap(bbox_detections(:,1:4), bbox_gt(j,1:4));
end

overlap(overlap==0) = -inf;
[max_overlap, jmax] = max(overlap,[],2);

does_overlap = max_overlap >= minoverlap;

false_positives(~does_overlap) = 1; % false positive

does_overlap = does_overlap & ~is_difficult(jmax); % dont care about the difficult ones

if any(does_overlap) 
    bbox_indices = find(does_overlap);
    jmax         = jmax(does_overlap);

    for j = 1:num_bbox_gt
        overlap_j = bbox_indices(jmax == j);
        if ~isempty(overlap_j)
            true_positives(overlap_j(1)) = 1; % true positive
            if numel(overlap_j) > 1
                false_positives(overlap_j(2:end)) = 0; % multiple detections
            end
        end      
    end
end

end

function all_results = compute_average_precision_of_detection(all_bbox_gt, all_detected_bbox, minoverlap, penalize_duplicates)
num_imgs       = length(all_bbox_gt);
num_classes    = length(all_detected_bbox);
true_positives   = cell(num_classes,1);
false_positives  = cell(num_classes,1);
detection_scores = cell(num_classes,1);

for class_idx = 1:num_classes
    true_positives{class_idx}   = cell(num_imgs,1);
    false_positives{class_idx}  = cell(num_imgs,1);
    detection_scores{class_idx} = cell(num_imgs,1);
end

num_positives = zeros(num_classes, 1);
for img_idx = 1:num_imgs
    for class_idx = 1:num_classes
        ground_truth_idx = all_bbox_gt{img_idx}(:,5) == class_idx;
        bbox_gt          = all_bbox_gt{img_idx}(ground_truth_idx,1:4);
        is_difficult     = all_bbox_gt{img_idx}(ground_truth_idx,6) > 0;
        num_positives(class_idx) = num_positives(class_idx) + sum(~is_difficult);

        bbox_detections = all_detected_bbox{class_idx}{img_idx};
        num_detections  = size(bbox_detections, 1);
        if num_detections > 0
            if isempty(bbox_gt)
                false_positives{class_idx}{img_idx} = ones(num_detections, 1);
                true_positives{class_idx}{img_idx}  = zeros(num_detections, 1);
            else
                [~, order]      = sort(bbox_detections(:,5), 'descend');
                bbox_detections = bbox_detections(order, :);

                if penalize_duplicates
                    [false_positives{class_idx}{img_idx},...
                        true_positives{class_idx}{img_idx}] = ...
                        find_detection_labels(bbox_detections(:,1:4), ...
                        bbox_gt, is_difficult, minoverlap);
                else
                    [false_positives{class_idx}{img_idx},...
                        true_positives{class_idx}{img_idx}] = ...
                        find_detection_labels_no_loc(bbox_detections(:,1:4), ...
                        bbox_gt, is_difficult, minoverlap);                    
                end
            end
            detection_scores{class_idx}{img_idx} = double(bbox_detections(:,5));
        end
    end
end

all_results = compute_average_precision(true_positives, false_positives, ...
    detection_scores, 1:num_classes, num_positives);
end

function res = compute_average_precision(true_positives, false_positives, detection_scores, class_indices, num_positives)
for class_idx = class_indices
    all_true_positives  = cell2mat(true_positives{class_idx}(:));
    all_false_positives = cell2mat(false_positives{class_idx}(:));
    all_scores          = cell2mat(detection_scores{class_idx}(:));
    [all_scores, order] = sort(all_scores, 'descend');
    all_true_positives  = all_true_positives(order);
    all_false_positives = all_false_positives(order);

    % compute precision/recall
    all_false_positives = cumsum(all_false_positives);
    all_true_positives  = cumsum(all_true_positives);

    recall    = all_true_positives  /  num_positives(class_idx);
    precision = all_true_positives ./ (all_false_positives + all_true_positives);
    f1_score  = 2 * (recall.*precision) ./ (recall + precision + eps);
    [f1_score, max_idx] = max(f1_score);
    f1_thresh    = all_scores(max_idx);
    
    % compute average precision
    average_precion = 0;
    for t=0:0.1:1
        p = max(precision(recall>=t));
        if ~isempty(p)
            average_precion = average_precion + p/11;
        end
    end
    ap_auc = xVOCap(recall, precision);
    res(class_idx).recall    = recall;
    res(class_idx).precision = precision;
    res(class_idx).ap        = average_precion;
    res(class_idx).ap_auc    = ap_auc;
    res(class_idx).f1_score  = f1_score;
    res(class_idx).f1_thresh = f1_thresh;
%     fprintf('!!! %s : %.4f %.4f\n', classes{class_idx}, average_precion, ap_auc);
end
end

function ap = xVOCap(rec,prec)

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
end
