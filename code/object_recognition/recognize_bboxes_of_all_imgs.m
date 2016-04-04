function [abbox_scores] = recognize_bboxes_of_all_imgs(...
    model, image_paths, all_bbox_proposals, dst_directory, image_set_name, varargin)
% recognize_bboxes_of_all_imgs: given a bounding box recognition model and 
% a set of images with their input candidate bounding boxes, for each 
% image it assigns a confidence score to each candidate bounding box that
% represents the likelihood to tightly enclose an object of interest.
%
% INPUTS:
% 1) model: (type struct) the bounding box recognition model
% 2) image_paths: NI x 1 cell array with the image paths of the set in the
% form of strings. NI is the number of images.
% 3) all_bbox_proposals: a cell array with the candidate bounding
% boxes that are going to be processed. It can be of two forms:
%   a) (the flag is_per_class must being set to false) a NI x 1 cell array 
%   where NI is the number of images. The all_bbox_proposals{i} element 
%   contains the candidate bounding boxes of the i-th image in the form of a 
%   NB_i x 4 array where NB_i is the number of candidate bounding boxes of  
%   the i-th image. The 4 columns of this array contain the bounding box  
%   coordinates in the form of [x0,y0,x1,y1] (where the (x0,y0) and (x1,y1)  
%   are the top-left and bottom-right corners).
%   b) (the flag is_per_class must being set to true) a C x 1 cell array 
%   where C is the number of categories. The all_bbox_proposals{j} element 
%   in this case is NI x 1 cell array, where NI is the number of images, 
%   with the candidate bounding boxes of the j-th category for each image. 
%   The element all_bbox_proposals{j}{i} is a NB_{i,j} x 4 array, where 
%   NB_{i,j} is the number candidate bounding box of the i-th 
%   image for the j-th category. The 4 columns of this array are the 
%   bounding boxes in the form of [x0,y0,x1,y1] (where the (x0,y0) and (x1,y1) 
%   are the top-left and bottom-right corners). 
% 4) dst_directory: string with the path of the destination directory where
% the results are going to be cached. 
% 5) image_set_name: string with the name of the image set that is being
% processed. It will be used on the name of the file where the results will
% be saved.
% 6) The rest input arguments are given in the form of Name,Value pair
% arguments and are:
% 'is_per_class': boolean value that if set to false, then the 3.a) form
% of the all_bbox_proposals input parameter will be expected; otherwise, if set to 
% true then the 3.b) form of the all_bbox_proposals input parameter will be
% expected. Default value: false
% 'suffix': string with a suffix that it will be used on the name of the 
% file where the results will be saved. Default value: ''
% 'all_bbox_gt': (optional) a NI cell array with the ground truth bounding
% boxes of each image; it will be used (if given) for printing the mAP on
% regular (see param 'checkpoint_step') step during processing the image set
% 'checkpoint_step': scalar value; during processing the NI images, after each
% checkpoint_step number of images the results are cached and the mAP
% results are printed. Default value: 500
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


ip = inputParser;
ip.addParamValue('is_per_class', false, @islogical);
ip.addParamValue('suffix', '', @ischar);
ip.addParamValue('all_bbox_gt', {},     @iscell);
ip.addParamValue('checkpoint_step', 500, @isnumeric);

ip.parse(varargin{:});
opts = ip.Results;

mkdir_if_missing(dst_directory);
filepath_results     = [dst_directory, filesep, 'scores', '_boxes_', opts.suffix, image_set_name, '.mat'];
in_progress_filepath = [dst_directory, filesep, 'scores', '_boxes_', opts.suffix, image_set_name, '_in_progress.mat'];

timestamp       = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_file        = fullfile(dst_directory, 'output', ...
    ['log_file_recognition',opts.suffix, image_set_name, '_', timestamp, '.txt']);
mkdir_if_missing(fileparts(log_file));

t_start = tic();
try
    abbox_scores = load_bboxes_scores(filepath_results);
catch
    diary(log_file);
    abbox_scores = process_images(...
        model, image_paths, all_bbox_proposals, opts.all_bbox_gt, ...
        opts.is_per_class, in_progress_filepath, opts.checkpoint_step);
    save_bboxes_scores(filepath_results, abbox_scores);
    diary off;
    delete(in_progress_filepath);
end
fprintf('Recognize bounding box proposals in %.4f minutes.\n', toc(t_start)/60);
end

function save_bboxes_scores(filename, bboxes_scores)
save(filename, 'bboxes_scores', '-v7.3');
end

function bboxes_scores = load_bboxes_scores(filename)
load(filename, 'bboxes_scores');
end

function all_bboxes_out = process_images(...
    model, image_paths, all_bbox_proposals, all_bbox_gt, ...
    is_per_class, in_progress_filepath, checkpoint_step)

num_imgs      = length(image_paths);
nms_iou_thrs  = 0.3; 
max_per_set   = 5 * num_imgs;
max_per_image = 100;
num_classes   = length(model.classes);
in_progress_filepath_prev = [in_progress_filepath, '.prev'];

try
    % load (if any) cached subset of results from images that have being
    % already processed from a previous run of the script that was stopped
    % before finish processing all the images.
    try
        [fist_img_idx, aboxes, all_bboxes_out, thresh] = load_progress(in_progress_filepath);
    catch
        [fist_img_idx, aboxes, all_bboxes_out, thresh] = load_progress(in_progress_filepath_prev);
    end
    mAP   = check_progress_on_mAP(aboxes, all_bbox_gt, fist_img_idx-1, model.classes);
    mAP_i = fist_img_idx-1;
catch exception
    fprintf('Exception message %s\n', getReport(exception));
    aboxes = cell(num_classes, 1);
    
    for i = 1:num_classes, aboxes{i} = cell(num_imgs, 1); end
    if is_per_class
        all_bboxes_out = cell(num_classes,1);
        for i = 1:num_classes, all_bboxes_out{i} = cell(num_imgs, 1); end
    else
        all_bboxes_out = cell(num_imgs,1);
    end
    thresh = -3.5 * ones(num_classes, 1);
    fist_img_idx = 1; mAP = 0; mAP_i = 0;
end

total_el_time = 0;
model.max_rois_num_in_gpu = 1000; %find_max_rois_num_in_gpu(model); 
num_chars = 0;
for i = fist_img_idx:num_imgs
    th = tic;
    % get the bounding box proposals of this image
    bbox_proposals = get_this_img_bbox_proposals(all_bbox_proposals, i, is_per_class);
    image          = get_image(image_paths{i}); % read the image
    % score the bounding box proposals with the recognition model
    bboxes_scores  = recognize_bboxes_of_image(model, image, bbox_proposals(:,1:4));
    recognition_time = toc(th); 
    
    th = tic;
    % prepare the candidate boundng box detections of this image
    bbox_cand_dets = prepare_bbox_cand_dets(bbox_proposals, bboxes_scores, num_classes, is_per_class);
    % prepare the output for this image
    all_bboxes_out = prepare_this_img_output(all_bboxes_out, i, is_per_class, bbox_cand_dets);
    % perform the post-processing step of non-max-suppresion on the 
    % candidate boundng box detections of this image
    bbox_detections = post_process_candidate_detections(bbox_cand_dets, ...
        'thresholds',thresh, 'nms_iou_thrs',nms_iou_thrs,'use_gpu',true,...
        'max_per_image',max_per_image);

    for j = 1:num_classes, aboxes{j}{i} = bbox_detections{j}; end
    postprocessing_time = toc(th);
    
    if mod(i, checkpoint_step) == 0
        th = tic;
        for j = 1:num_classes, [aboxes{j}, thresh(j)] = keep_top_k(aboxes{j}, i, max_per_set, thresh(j)); end
        disp(thresh(:)');
        
        % save the till now progress
        save_progress(aboxes, all_bboxes_out, thresh, i, in_progress_filepath, in_progress_filepath_prev);
        
        % evaluate the mAP of the images 1 till i
        mAP = check_progress_on_mAP(aboxes, all_bbox_gt, i, model.classes);
        mAP_i = i;
        diary; diary; % flush diary

        checkpoint_time = toc(th);
        fprintf('check_point time %.2fs\n',checkpoint_time)
        num_chars = 0;
    end
    
    elapsed_time = recognition_time + postprocessing_time;
    [total_el_time, avg_time, est_rem_time] = timing_process(elapsed_time, total_el_time, fist_img_idx, i, num_imgs);
    fprintf(repmat('\b',[1, num_chars]));
    
    num_chars = fprintf('%s: bbox rec %d/%d:| ET %.3fs + %.3fs | AT: %.3fs | TT %.4fmin | ERT %.4fmin | mAP[%d/%d] = %.4f\n', ...
        procid(), i, num_imgs, recognition_time, postprocessing_time, avg_time, ...
        total_el_time/60, est_rem_time/60, mAP_i, num_imgs, mAP);
    
end

delete(in_progress_filepath_prev);
end

function [total_el_time, ave_time, est_rem_time] = timing_process(...
    elapsed_time, total_el_time, fist_img_idx, i, num_imgs)

total_el_time   = total_el_time + elapsed_time;
ave_time        = total_el_time / (i-fist_img_idx+1);
est_rem_time    = ave_time * (num_imgs - i);
end

function bbox_proposals = get_this_img_bbox_proposals(all_bboxes_in, img_id, is_per_class)
if is_per_class
    bbox_proposals_per_class = cellfun(@(x) x{img_id}(:,1:4), all_bboxes_in, 'UniformOutput', false);
    num_bbox_per_class = cellfun(@(x) size(x,1), bbox_proposals_per_class,  'UniformOutput', true);
    bbox_proposals = cell2mat(bbox_proposals_per_class(num_bbox_per_class>0));
    
    class_indices = single([]);
    for c = 1:length(num_bbox_per_class)
        class_indices = [class_indices; ones(num_bbox_per_class(c),1,'single')*c];
    end

    bbox_proposals = single([bbox_proposals, class_indices]);
    if isempty(bbox_proposals), bbox_proposals = zeros(0,5,'single'); end
else
    bbox_proposals = all_bboxes_in{img_id}(:,1:4);
end
end

function bbox_cand_dets = prepare_bbox_cand_dets(bbox_proposals, bbox_scores, num_classes, is_per_class)
if is_per_class
    class_indices  = bbox_proposals(:,5);
    bbox_cand_dets = cell(num_classes,1);
    for c = 1:num_classes
        this_cls_mask = class_indices==c;
        bbox_cand_dets{c} = single([bbox_proposals(this_cls_mask,1:4),bbox_scores(this_cls_mask,c)]);
        if isempty(bbox_cand_dets{c}), bbox_cand_dets{c} = zeros(0,5,'single'); end
    end
else
    bbox_cand_dets = single([bbox_proposals(:,1:4), bbox_scores]);
end
end

function all_bboxes_out = prepare_this_img_output(all_bboxes_out, img_idx, is_per_class, bbox_this_img)
if is_per_class
    for j = 1:length(bbox_this_img), all_bboxes_out{j}{img_idx} = bbox_this_img{j}; end
else
    all_bboxes_out{img_idx} = bbox_this_img;
end
end

function [fist_img_idx, aboxes, abbox_scores, thresh] = load_progress(in_progress_filepath)
ld = load(in_progress_filepath); 
fist_img_idx = ld.progress_state.img_idx + 1;
aboxes       = ld.progress_state.aboxes;
abbox_scores = ld.progress_state.abbox_scores;
thresh       = ld.progress_state.thresh;  
end

function save_progress(aboxes, abbox_scores, thresh, img_idx, in_progress_filepath, in_progress_filepath_prev)
progress_state              = struct;
progress_state.img_idx      = img_idx;
progress_state.aboxes       = aboxes;
progress_state.abbox_scores = abbox_scores;
progress_state.thresh       = thresh;

if exist(in_progress_filepath, 'file')
    % in case it crash during updating the in_progress_filepath file 
    copyfile(in_progress_filepath, in_progress_filepath_prev); 
end
save(in_progress_filepath, 'progress_state', '-v7.3');
end

function mAP = check_progress_on_mAP(aboxes, all_bbox_gt, img_idx, classes)
mAP = 0;
if ~isempty(all_bbox_gt)
    aboxes      = cellfun(@(x) x(1:img_idx), aboxes, 'UniformOutput', false);
    mAP_result  = evaluate_average_precision_pascal( all_bbox_gt(1:img_idx), aboxes, classes );
    printAPResults(classes, mAP_result);
    mAP = mean([mAP_result(:).ap]');
end
end

function [boxes, thresh] = keep_top_k(boxes, end_at, top_k, thresh)
% ------------------------------------------------------------------------
% Keep top K
X = cat(1, boxes{1:end_at});
if isempty(X), return; end

scores = sort(X(:,end), 'descend');
thresh = scores(min(length(scores), top_k));
for image_index = 1:end_at
    bbox = boxes{image_index};
    keep = find(bbox(:,end) >= thresh);
    boxes{image_index} = bbox(keep,:);
end

end