function demo_LocNet_object_detection_pipeline(varargin)
% demo of the object detection pipeline using as a localization model
% either a LocNet model or a CNN-based bounding box regression model
%
% The input arguments are given in the form of Name,Value pair arguments 
% and are:
% 'gpu_id': scalar value with gpu id (one-based index) that will be used for 
% running the demo. If a non positive value is given then the CPU will be 
% used instead. Default value: 1
% 'rec_dir_name': the directory name of the recognition model that will be
% used for running the demo. The possible options (if you download and 
% properly place the pre-trained models) are 'VGG16_Reduced_MRCNN' for the 
% Reduced MR-CNN model and 'VGG16_FastRCNN' for the (re-implemented by us) 
% Fast RCNN model. Default value: 'VGG16_Reduced_MRCNN'
% 'loc_dir_name': the directory name of the localization model that will be
% used for running the demo. The possible options (if you download and 
% properly place the pre-trained models) are:
% 1) 'VGG16_LocNet_InOut':    LocNet-InOut localization model
% 2) 'VGG16_LocNet_Borders':  LocNet-Borders localization model
% 3) 'VGG16_LocNet_Combined': LocNet-Combined localization model
% 4) 'VGG16_BBoxReg':         CNN-based bounding box regression (baseline)
% Default value: 'VGG16_LocNet_InOut'
% 'box_method': string with the box proposals algorithm that will be used 
% in order to generate the set of candidate boxes that will be given as 
% input to the object detection pipeline. Currently the possible choises 
% are: 'edge_boxes', 'selective_search', or 'dense_boxes_10k'. Default
% value: 'edge_boxes'.
% 'img_path': path to the image that will be used for the demo. Note that
% if you use other than the default image for running the demo you will 
% need to first install the box proposal algorithms (read the README.md
% file).
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
ip.addParamValue('gpu_id',      1,                                  @isscalar);
ip.addParamValue('box_method',  'edge_boxes',                       @ischar);
ip.addParamValue('img_path',    './code/examples/images/000084.jpg',@ischar);
ip.addParamValue('rec_dir_name','VGG16_Reduced_MRCNN',              @ischar);
ip.addParamValue('loc_dir_name','VGG16_LocNet_InOut',               @ischar);
ip.parse(varargin{:});
opts = ip.Results;

%******************************* OPTIONS **********************************
% The GPU that will be used for running the demo. gpu_id is a one-based 
% index; if a non positive value is given then the CPU will be used instead.  
gpu_id = opts.gpu_id; 

% Set the directory name of bounding box recognition model; this could be 
% the 'VGG16_Reduced_MRCNN' for the Reduced MR-CNN model and 
% 'VGG16_FastRCNN' for the (re-implemented by us) Fast RCNN model.
model_rec_dir_name = opts.rec_dir_name;

% Set the directory name of the LocNet localization model or the CNN-based 
% bounding box regression model); the possible options (after downloading
% the models) could be:
% 1) 'VGG16_LocNet_InOut':    LocNet-InOut localization model
% 2) 'VGG16_LocNet_Borders':  LocNet-Borders localization model
% 3) 'VGG16_LocNet_Combined': LocNet-Combined localization model
% 4) 'VGG16_BBoxReg':         CNN-based bounding box regression (baseline)
model_loc_dir_name = opts.loc_dir_name;

img_path   = opts.img_path; % path to the image

box_method = opts.box_method; % string with the box proposals algorithm that
% will be used in order to generate the set of candidate boxes that will be
% given as input to the object detection pipeline. Currently the possible
% choises are: 'edge_boxes', 'selective_search', or 'dense_boxes_10k' 

% number of iterations that the iterative localization scheme is performed.
num_iterations = 4; 

% maximum number of regions/bboxes that can be given in one go in the 
% caffe networks such that everything can fit in the GPU memory; 
% For a GPU with 6 Gbytes memory:
model_obj_rec_max_rois_num_in_gpu = 200; 
model_obj_loc_max_rois_num_in_gpu = 200; 
%**************************************************************************

%***************************** LOAD MODELS ********************************
caffe_set_device( gpu_id );
caffe.reset_all();
% Normaly, the loading of the models needs to performed only once before
% processing a bunch of images
fprintf('Loading detection models... '); th = tic;

% set the path of the bounding box recognition model for object detection
full_model_rec_dir  = fullfile(pwd, 'models-exps', model_rec_dir_name); % full path to the model's directory
use_detection_svms  = true;
model_rec_mat_name  = 'detection_model_svm.mat'; % model's matlab filename
full_model_rec_path = fullfile(full_model_rec_dir, model_rec_mat_name); % full path to the model's matlab file
assert(exist(full_model_rec_dir,'dir')>0);
assert(exist(full_model_rec_path,'file')>0);

% Load the recognition moddel
ld = load(full_model_rec_path, 'model');
model_obj_rec = ld.model; 
clear ld; 
model_obj_rec = load_object_recognition_model_on_caffe(...
    model_obj_rec, use_detection_svms, full_model_rec_dir);

% Set the path of the LocNet localization model that will be used 
full_model_loc_dir  = fullfile(pwd, 'models-exps', model_loc_dir_name);
model_loc_mat_name  = 'localization_model.mat'; % model's matlab filename
full_model_loc_path = fullfile(full_model_loc_dir, model_loc_mat_name);  % full path to the model's matlab file
assert(exist(full_model_loc_dir,'dir')>0);
assert(exist(full_model_loc_path,'file')>0);

% Load the LocNet model
ld = load(full_model_loc_path, 'model');
model_obj_loc = ld.model; 
clear ld;
model_obj_loc = load_bbox_loc_model_on_caffe(model_obj_loc, full_model_loc_dir);
fprintf(' %.3f sec\n', toc(th));

% maximum number of regions/bboxes that can be given in one go in the 
% caffe networks such that everything can fit in the GPU memory
model_obj_rec.max_rois_num_in_gpu = model_obj_rec_max_rois_num_in_gpu; 
model_obj_loc.max_rois_num_in_gpu = model_obj_loc_max_rois_num_in_gpu;
%**************************************************************************

%************************** CONFIGURATION *********************************
category_names = model_obj_rec.classes; % a C x 1 cell array with the name 
% of the categories that the detection system looks for. C is the number of
% categories.
num_categories = length(category_names);

conf = struct;
conf.thresh = -3 * ones(num_categories,1); % It contains the 
% threshold per category that will be used for removing scored boxes with 
% low confidence prior to applying the non-max-suppression step.
conf.nms_iou_thrs = 0.3; % IoU threshold for the non-max-suppression step
conf.box_method = box_method; % string with the box proposals algorithm that
% will be used in order to generate the set of candidate boxes that will be
% given as input to the object detection pipeline. Currently the possible
% choises are: 'edge_boxes', 'selective_search', or 'dense_boxes_10k' 
conf.num_iterations = num_iterations; % number of iterations that the iterative localization scheme is performed.
conf.thresh_init = -2.5 * ones(num_categories,1); % the threshold per category 
% that will be used in order to prune the candidate boxes with low 
% confidence only at the first iteration of the object detection pipeline 
% (in order to remove the computation cost of subsequent steps). Normally
% this threshold is chosen such as the average number of candidate boxes
% per image and per category to be around 18.
conf.nms_iou_thrs_init = 0.95; 
if strcmp(conf.box_method,'dense_boxes_10k'), conf.nms_iou_thrs_init = 0.85; end
% nms_iou_thrs_init: the IoU threshold that will be used during the 
% non-max-suppression step that is applied at the first iteration of the 
% object detection pipeline in order to remove near duplicate box proposals 
%**************************************************************************

%********************* RUN THE DETECTION PIPELINE *************************
img = imread(img_path); % load image
% extract the box proposals that will be given as input to the object
% detection pipeline
fprintf('Extracting bounding box proposals... '); th = tic;
[a,b] = fileparts(img_path);
proposals_files = fullfile(a, [b, '_',box_method, '_box_proposals.mat']);
try 
    ld = load(proposals_files);
    bbox_proposals = ld.bbox_proposals;
catch
    bbox_proposals = extract_object_proposals( img, conf );
%     save(proposals_files,'bbox_proposals');
end
fprintf(' %.3f sec\n', toc(th));

% run the LocNet object detection pipeline in the image
fprintf('LocNet object detection pipeline... '); th = tic;
bbox_detections = LocNet_object_detection( img, model_obj_rec, ...
    model_obj_loc, bbox_proposals, conf);
fprintf(' %.3f sec\n', toc(th));

caffe.reset_all(); % free the memory occupied by the caffe models
%**************************************************************************

%************************ VISUALIZE DETECTIONS ****************************
% visualize the bounding box detections.
score_thresh = 0 * ones(num_categories, 1); % the minimum score threshold per 
% category for keeping or discarding a detection. For the purposes of this
% demo we set the thresholds to 0 value (which is not however, the
% optimal value).
display_bbox_detections( img, bbox_detections, score_thresh, category_names );
%**************************************************************************
end
