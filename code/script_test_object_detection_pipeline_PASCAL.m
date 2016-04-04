function script_test_object_detection_pipeline_PASCAL(model_rec_dir_name, model_loc_dir_name, varargin)
% script_test_object_detection_pipeline_PASCAL(model_rec_dir_name, model_loc_dir_name, varargin)
% given a recognition model (model_rec_dir_name) and a localization model
% (model_loc_dir_name) it performs the object detection task on the
% specified PASCAL VOC set of images using the aforementioned models. 
% This script must always be called from the installation directory of the
% code.
% 
% INPUTS:
% 1) model_rec_dir_name: string with the directory name that contains the
% recognition model. The full path to the model directory is:
% "{installation-directory-path}/models-exps/{model_rec_dir_name}/"
% 2) model_loc_dir_name: string with the directory name that contains the
% recognition model. The full path to model the directory is:
% "{installation-directory-path}/models-exps/{model_loc_dir_name}/"
% 3) The rest input arguments are given in the form of Name,Value pair
% arguments and are:
% 'gpu_id': scalar value with gpu id (one-based index) that will be used for 
% running the experiments. If a non positive value is given then the CPU
% will be used instead. Default value: 0
% 'image_set_test': string with PASCAL VOC image set to which the object
% detection task will be run, e.g. 'test', 'trainval', or 'val'. Default value:'test'
% 'voc_year_test': string with the PASCAL VOC challenge year to which the 
% object detection task will be run, e.g. '2007' or '2012'. Default value:'2007'
% 'bbox_proposals': string with the class agnostic object proposals
% algorithm that will be used as input to the detection system. The options
% that are supported are 'edge_boxes' and 'selective_search'. Default value:'edge_boxes'
% 'use_detection_svms': boolean value that if set to true then the last layer 
% of the recognition model is a set of class-specific detection svms trained with
% hard negative mining. Default value: true
% 'num_iterations': scalar value with the number of iterations that the 
% iterative localization scheme of the object detection pipeline will be 
% applied. Default value: 4
%
% (other arguments)
% 'nms_thr': scalar value with the IoU threshold that is being used on the 
% non-maximum-suppression step at post-processing time. Default value: 0.3
% 'score_thr': scalar value with score threshold that is being applied to
% candidate detections prior to the non-max-suppression step. Default value: -4
% If this value is low enough then it doesn't affect the result of the mAP
% metric and it's purpose is to speed up the nms step and the evalution of 
% the mAP metrics.
% 'ave_per_image_init': scalar value with the desired average number of
% candidate boxes per image and per category that should remain after
% pruning the candidate bounding box detections with low confidence. This
% pruning operation is applied after the first time that the recognition
% model is applied on the bounding box proposals with the purpose of
% reducing the computational complexity of the subsequent steps of the
% iterative localization scheme (since they will process fewer boxes). 
% Default value: 15
% 'score_thr_init': the threshold that will be used in order to prune the 
% candidate boxes with low confidence after the first iteration of the 
% iterative localization scheme (see above). Default value -4. Actually,
% if this value is small enough it doesn't affect the output, and the
% threshold is actually defined from the 'ave_per_image_init' parameter value above.
% 'nms_init': scalar value with the IoU threshold that will be used during 
% the non-max-suppression step that is applied after the first iteration of
% the iterative localization scheme in order to remove near duplicate box 
% proposals (with the purpose of reducing the computational complexity of 
% subsequent steps). Default value: 0.95.
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

%****************************** OPTIONS ***********************************
ip = inputParser;
ip.addParamValue('gpu_id', 0, @isscalar);

ip.addParamValue('image_set_test', 'test')
ip.addParamValue('voc_year_test',  '2007')
ip.addParamValue('bbox_proposals', 'edge_boxes');

ip.addParamValue('use_detection_svms',  true,       @islogical);
ip.addParamValue('num_iterations', 4, @isscalar);

ip.addParamValue('nms_thr',      0.3, @isnumeric);
ip.addParamValue('score_thr',     -4, @isnumeric);

ip.addParamValue('nms_init',         0.95,  @isscalar);
ip.addParamValue('score_thr_init',     -4,  @isscalar);
ip.addParamValue('ave_per_image_init', 15,  @isscalar);

% other
ip.addParamValue('startIdx',             [], @isnumeric);
ip.addParamValue('stopIdx',              [], @isnumeric);
ip.addParamValue('checkpoint_step',     500, @isnumeric);
ip.addParamValue('cache_dir',      '', @ischar);
ip.addParamValue('cache_dir_name', '', @ischar);
ip.addParamValue('bbox_loc_suffix',   '', @ischar);

ip.parse(varargin{:});
opts = ip.Results;

if opts.use_detection_svms
    opts.model_rec_mat_name = 'detection_model_svm.mat';
else
    opts.model_rec_mat_name = 'detection_model_softmax.mat';
end
opts.model_loc_mat_name = 'localization_model.mat';
bbox_voting_add_val = 0;
if opts.use_detection_svms, bbox_voting_add_val = 1.5; end
gpu_id = opts.gpu_id;

full_model_rec_dir  = fullfile(pwd, 'models-exps', model_rec_dir_name);
full_model_rec_path = fullfile(full_model_rec_dir, opts.model_rec_mat_name);
assert(exist(full_model_rec_dir,'dir')>0);
assert(exist(full_model_rec_path,'file')>0);

full_model_loc_dir  = fullfile(pwd, 'models-exps', model_loc_dir_name);
full_model_loc_path = fullfile(full_model_loc_dir, opts.model_loc_mat_name);
assert(exist(full_model_loc_dir,'dir')>0);
assert(exist(full_model_loc_path,'file')>0);

voc_year_test   = opts.voc_year_test;
image_set_test  = opts.image_set_test;
bbox_proposals  = opts.bbox_proposals; 
        

%**************************************************************************

%*************************** LOAD DATASET *********************************
% load image dataset
image_db_test = load_image_dataset(...
    'image_set', image_set_test, ...
    'voc_year',  voc_year_test, ...
    'proposals_method', bbox_proposals);

if (~isempty(opts.startIdx) && ~isempty(opts.stopIdx))
    image_db_test = get_image_subset(image_db_test, opts.startIdx, opts.stopIdx);
end

image_paths_test      = image_db_test.image_paths;
all_regions_test      = image_db_test.all_regions;
all_bbox_gt_test      = image_db_test.all_bbox_gt;
proposals_suffix_test = image_db_test.proposals_suffix;
image_set_name_test   = image_db_test.image_set_name;

% compute and display stats related to the cantidate bounding boxes that
% are given as input to the object detection pipeline
compute_recall_per_IoU_of_bboxes(all_bbox_gt_test, all_regions_test, {'all'});
%**************************************************************************

%***************************** LOAD MODELS ********************************
% load object recognition model
ld = load(full_model_rec_path, 'model');
model_obj_rec = ld.model; 
clear ld; 

% load localization model
ld = load(full_model_loc_path, 'model');
model_bbox_loc = ld.model; 
clear ld;
%**************************************************************************

%************************* PREPARE CACHE DIRECTORY ************************
% set the cache directory where intermediate results and the final results
% will be saved
cache_dir         = resolve_cache_dir(model_obj_rec, full_model_rec_dir, opts);
results_dir_name  = sprintf('res_%s', proposals_suffix_test);
dst_rec_directory = fullfile(cache_dir, results_dir_name);
mkdir_if_missing(dst_rec_directory);

if isempty(opts.bbox_loc_suffix), opts.bbox_loc_suffix = model_loc_dir_name; end

results_dir_loc_name = sprintf('object_localization_nms%.2f_thr%.2f_ave%d_%s_Dataset_%s', ...
    opts.nms_init, opts.score_thr_init, opts.ave_per_image_init, opts.bbox_loc_suffix, image_set_name_test); 

dst_loc_directory = fullfile(dst_rec_directory, results_dir_loc_name);
mkdir_if_missing(dst_loc_directory);
fprintf('cache dirname loc: %s\n',results_dir_loc_name);
fprintf('cache dir loc:\n%s\n',   dst_loc_directory);
%**************************************************************************

%***************************** TEST MODELS ********************************
classes = model_obj_rec.classes;
save_file_data = fullfile(dst_loc_directory, 'iterative_object_localization_data.mat');
try 
    % if the results are already computed load them from the disk
    ld = load(save_file_data);
    abbox_det_cands_rescore = ld.abbox_det_cands_rescore;
    assert(length(abbox_det_cands_rescore) >= opts.num_iterations);
catch exception
    fprintf('Exception message %s\n', getReport(exception));
    
    caffe_set_device( gpu_id );
    abbox_det_cands_rescore  = cell(opts.num_iterations,1);  
    
    dst_directory    = dst_rec_directory;
    suffix_this_iter = '';
    is_per_class     = false;
    
    %********************** OBJECT DETECTION PIPELINE *********************
    % apply the iterative object detection pipeline on the specified set of
    % images
    for iter = 1:opts.num_iterations

        %************************ SCORE CAND. BBOXES **********************
        % score the candidate bounding box proposals of each image with the
        % recognition model
        caffe.reset_all();
        model_obj_rec = load_object_recognition_model_on_caffe(...
            model_obj_rec, opts.use_detection_svms, full_model_rec_dir);
        abbox_det_cands_rescore{iter} = recognize_bboxes_of_all_imgs(...
            model_obj_rec, image_paths_test, all_regions_test, dst_directory,...
            image_set_name_test, 'all_bbox_gt', all_bbox_gt_test, ...
            'checkpoint_step', opts.checkpoint_step, 'is_per_class', is_per_class, ...
            'suffix', suffix_this_iter);
        caffe.reset_all();
        %******************************************************************  
        
        if iter == 1
            %************* PRUNE LOW CONFIDENCE CAND. BBOXES **************
            % Prune candidate boxes with low confidence score and remove
            % near duplicate boxes in order to reduce the computational
            % cost of the subsequent steps. 
            abbox_det_cands_rescore{iter} = post_process_candidate_detections_all_imgs(...
                abbox_det_cands_rescore{iter}, 'nms_iou_thrs', opts.nms_init, 'threshold', opts.score_thr_init, ...
                'ave_per_image', opts.ave_per_image_init, 'use_gpu',gpu_id>0,'is_per_class', is_per_class);
            %**************************************************************   
            is_per_class  = true;
            dst_directory = dst_loc_directory; 
        end
        
%         %************************** EVALUATE mAP ****************************
%         fprintf('Object Localization %d Iterations - Intermediate results:\n', iter);
%         abbox_dets_i = post_process_candidate_detections_all_imgs(...
%             abbox_det_cands_rescore{iter},'nms_iou_thrs', opts.nms_thr,'threshold',opts.score_thr,...
%             'is_per_class', is_per_class,'do_bbox_voting',true,'add_val',bbox_voting_add_val,...
%             'box_ave_iou_thresh',0.5,'use_gpu',gpu_id>0);
%         print_detection_evaluation_metrics(all_bbox_gt_test, abbox_dets_i, classes);
%         %**************************************************************************
        
        suffix_this_iter = sprintf('_iter_%d', iter);
        fprintf('%s\n', suffix_this_iter);
        if (iter < opts.num_iterations)
            %********** LOCNET: REFINE LOCATION OF CAND. BBOXES ***********
            % refine the location of the candidate bounding boxes with the
            % LocNet localization model
            model_bbox_loc = load_bbox_loc_model_on_caffe(model_bbox_loc, full_model_loc_dir);
            all_regions_test = localize_bboxes_of_all_imgs(...
                model_bbox_loc, image_paths_test, abbox_det_cands_rescore{iter}, ...
                dst_directory, image_set_name_test, 'all_bbox_gt', all_bbox_gt_test, ...
                'checkpoint_step', opts.checkpoint_step, 'is_per_class', is_per_class, ...
                'suffix', suffix_this_iter);
            caffe.reset_all();
            
            [~, mean_ave_recall] = compute_recall_per_IoU_of_bboxes(all_bbox_gt_test, all_regions_test, classes);
            fprintf('Iter %d mAR = %.5f\n', iter, mean_ave_recall);            
            %**************************************************************           
        end
    end
    %**********************************************************************
    save(save_file_data, 'abbox_det_cands_rescore', '-v7.3');
end

for num_iter = opts.num_iterations
    %*************************** POST-PROCESSING ******************************
    % merge the candidate detection boxes of each iteration
    abbox_dets_cands_all = merge_detected_bboxes(abbox_det_cands_rescore(1:num_iter));
    abbox_dets_cands_all = abbox_dets_cands_all{1};
    % apply the post-processing steps of non-max-suppression and box voting
    abbox_dets = post_process_candidate_detections_all_imgs(...
        abbox_dets_cands_all,'nms_iou_thrs', opts.nms_thr,'threshold',opts.score_thr,...
        'is_per_class', true,'do_bbox_voting',true,'add_val',bbox_voting_add_val,...
        'box_ave_iou_thresh',0.5,'use_gpu',gpu_id>0);
    %**************************************************************************

    %************************** EVALUATE mAP FINAL ****************************
    fprintf('Object Localization %d Iterations - Final results:\n', num_iter);
    print_detection_evaluation_metrics(all_bbox_gt_test, abbox_dets, classes);
    %**************************************************************************
end
end

function image_db = get_image_subset(image_db, start_idx, stop_idx)
num_imgs = length(image_db.image_paths);

start_idx = max(1, start_idx);
stop_idx  = min(num_imgs, stop_idx);

image_db.image_paths    = image_db.image_paths(start_idx:stop_idx);
image_db.all_regions    = image_db.all_regions(start_idx:stop_idx);
image_db.all_bbox_gt    = image_db.all_bbox_gt(start_idx:stop_idx);
image_db.image_set_name = [image_db.image_set_name, sprintf('_start%d_stop%d',start_idx, stop_idx)];
end