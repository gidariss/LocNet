function script_train_LocNet_PASCAL(model_dir_name, varargin)
% script_train_LocNet_PASCAL(model_dir_name, ...): it trains a LocNet CNN 
% network for the task of object localization in the PASCAL VOC detection
% challenge. The trained model as well as intermediate results are placed
% in the created by the function directory "./models-exps/{model_dir_name}".
%
% For training the PASCAL dataset is used. By default the current script 
% trains the LocNet network on the union of the PASCAL VOC2007 train+val 
% and VOC2012 train+val datasets using both the selective search and the 
% edge box proposals and flipped version of the images.
% 
%
% INPUTS:
% 1) model_dir_name: string with the name of the directory where the
% trained LocNet CNN network willl be saved. The directory will be
% created in the following location "./models-exps/{model_dir_name}"
% 2) The rest input arguments are given in the form of Name,Value pair 
% arguments and are:
% ****************** LOCALIZATION/SEARCH REGION PARAMS ********************
% 'scale_ratio': (scalar value) the scaling factor of the search region
% w.r.t. the input bounding boxes. Specifically, given an input bounding 
% boxes, in order for LocNet to localize the target bounding box it "look" 
% in a search region that is obtained by scaling the input bounding box by 
% a scaling factor. This scaling factor is given by the scale_ratio. 
% parameter. Default value: 1.8. Note that this parameter is set to 1.8 for
% the LocNet models andto 1.3 for the CNN-based bounding box regression
% models. The aforementioned parameters gave the best results in each case.
% 'resolution': conf.resolution: scalar value; In order for the LocNet to localize 
% the target bounding boxes inside the search region, it considers a division 
% of the search region in M horizontal stripes (rows) and M vertical stripes 
% (columns). The value of M is given by the parameter resolution. Default
% value: 28. For other resolution values, new .prototxt definition files
% with the architecture of the network must be provided. 
% 'loc_type': string wiht the type of the localization model that the LocNet 
% implements. The possible options are: 'inout','borders','combined', and
% 'bboxreg'. If the 'bboxreg' option is given then, instead of a LocNet,
% the typical CNN-based bounding box network is trained. Default value: 'combined'
% *************************** MODEL PARAMS ********************************
% 'scales': NumScales x 1 or 1 x NumScales vector with the images scales
% that will be used. The i-th value should be the size in pixels of the
% smallest dimension of the image in the i-th scale. Default value: [600] 
% 'max_size': scalar value with the maximum size in pixels of the biggest
% dimension of the image (in any scale). Default value: [1000]
% ************************* TRAINING PARAMS *******************************
% 'ims_per_batch': scalar value with the number of training images per
% batch. Default value: 2.
% 'batch_size': scalar value with the number of training regions per bathc.
% Default value: 128
% 'fg_threshold_bbox_loc': scalar value with the minimum IoU threshold; 
% only the box proposals that overlap with a ground truth bounding box
% above this threhold (in the IoU metric) will be used during both training
% and validation. Default value: 0.4
% ************************** TRAINING SET *********************************
% 'train_set': a Ts x 1 or 1 x Ts cell array with the PASCAL VOC image set
% names that are going to be used for training. 
% Default value: {'trainval','trainval'}
% 'voc_year_train': a Ts x 1 or 1 x Ts cell array with the PASCAL VOC 
% challenge years (in form of strings) that are going to be used for 
% training. Examples:
%   - train_set = {'trainval'}; voc_year_train = {'2007'};
%   the training will be performed on VOC2007 train+val dataset
%   - train_set = {'trainval'}; voc_year_train = {'2012'};
%   the training will be performed on VOC2012 train+val dataset
%   - train_set = {'trainval','trainval'}; voc_year_train = {'2007','2012'};
%   the training will be performed on the union of VOC2007 train+val plus 
%   VOC2012 train+val datasets.
% 'proposals_method_train': a Tp x  1 or 1 x Tp cell array with object
% proposals that will be used for training, e.g. {'edge_boxes'}, 
% {'selective_search'}, or {'selective_search','edge_boxes'}.
% Default value: {'selective_search','edge_boxes'}
% 'train_use_flips': a boolean value that if set to true then flipped
% versions of the images are going to be used during training. Default value: true
% 
% Briefly, by default the current script trains the region adaptation 
% module on the union of the PASCAL VOC2007 train+val and VOC2012 train+val
% datasets using both the selective search and edge box proposals and 
% flipped version of the images.
% *************************************************************************
% ************************** VALIDATION SET *******************************
% 'val_set': similar to 'train_set'
% 'voc_year_val': similar to 'voc_year_train'
% 'proposals_method_val': similar to 'proposals_method_train'
% 'val_use_flips': similar to 'train_use_flips'
% 
% Briefly, by default the current script uses as validation set the PASCAL 
% VOC2007 test dataset using both the selective search proposals and NO 
% flipped version of the images.
% *************************************************************************
% OTHER:
% 'solverstate': string with the caffe solverstate filename in order to resume
% training from there. For example by setting the parameter 'solverstate'
% to 'model_iter_30000' the caffe solver will resume to training from the
% 30000-th iteration; the solverstate file is being assumed to exist on the
% location: "./models-exps/{model_dir_name}/model_iter_30000.solverstate".
% 'gpu_id': scalar value with gpu id (one-based index) that will be used for 
% running the experiments. If a non positive value is given then the CPU
% will be used instead. Default value: 0
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

% ************************** OPTIONS **************************************
ip = inputParser;
% training set
ip.addParamValue('train_set', {'trainval', 'trainval'})
ip.addParamValue('voc_year_train', {'2007', '2012'})
ip.addParamValue('proposals_method_train', {'selective_search', 'edge_boxes'});
ip.addParamValue('train_use_flips', true, @islogical);

% validation set
ip.addParamValue('val_set', {'test'})
ip.addParamValue('voc_year_val', {'2007'})
ip.addParamValue('proposals_method_val', {'selective_search'});
ip.addParamValue('val_use_flips', false, @islogical);

% localization params
ip.addParamValue('scale_ratio', 1.8, @isnumeric);
ip.addParamValue('resolution',  28, @isnumeric);
ip.addParamValue('loc_type', 'combined', @ischar); % options: 'inout','borders','combined',or 'bboxreg'

% model params
ip.addParamValue('scales', 600, @isnumeric);
ip.addParamValue('max_size', 1000, @isnumeric);

% training params
ip.addParamValue('batch_size', 128, @isnumeric);
ip.addParamValue('ims_per_batch', 2, @isnumeric);
ip.addParamValue('fg_threshold_bbox_loc', 0.4, @isnumeric);

% other
ip.addParamValue('solverstate', '', @ischar)
ip.addParamValue('test_only', false, @islogical)
ip.addParamValue('gpu_id', 0, @isscalar);
ip.addParamValue('finetuned_modelname', '', @ischar);

ip.parse(varargin{:});
opts = ip.Results;
% *************************************************************************

clc;

% configuration file with the region pooling parameters
opts.vgg_pool_params_def    = fullfile(pwd, 'model-defs/vgg_region_config.m');
% location of the model directory where the results of training the region
% adaptation module will be placed
opts.finetune_rst_dir       = fullfile(pwd, 'models-exps', model_dir_name);
mkdir_if_missing(opts.finetune_rst_dir);

opts.save_mat_model_only = false;
if ~isempty(opts.finetuned_modelname)
    % if the parameter finetuned_modelname is non-empty then no training is
    % performed and the current script only creates a .mat file that contains
    % the region adaptation module model that uses as weights/parameters
    % those of the file opts.finetuned_modelname
    opts.save_mat_model_only = true;
end


% the network weights file that will be used for initialization
opts.net_file = fullfile(pwd,'data/vgg16_pretrained/VGG_ILSVRC_16_layers.caffemodel'); 
assert(exist(opts.net_file,'file')>0);

% the solver definition file that will be used for training
switch opts.loc_type
    case 'bboxreg'
        def_solver_filename = 'VGG16_bbox_reg_solver.prototxt';
        def_deploy_filename = 'VGG16_bbox_reg_deploy.prototxt';
    case 'inout'
        def_solver_filename = 'VGG16_LocNet_InOut_M28_solver.prototxt';
        def_deploy_filename = 'VGG16_LocNet_InOut_M28_deploy.prototxt';
    case 'borders'
        def_solver_filename = 'VGG16_LocNet_Borders_M28_solver.prototxt';
        def_deploy_filename = 'VGG16_LocNet_Borders_M28_deploy.prototxt';
    case 'combined'
        def_solver_filename = 'VGG16_LocNet_Combined_M28_solver.prototxt'; 
        def_deploy_filename = 'VGG16_LocNet_Combined_M28_deploy.prototxt';
    otherwise
        error('Invalid loc_type = %s.\nSupported localization types %s, %s, %s, and %s.', ...
            opts.loc_type, 'inout', 'borders', 'combined', 'bboxreg');
end
opts.finetune_net_def_file = fullfile(pwd, 'model-defs', def_solver_filename);
assert(exist(opts.finetune_net_def_file,'file')>0);

% parse the solver file
[solver_file, ~, ~, opts.max_iter, opts.snapshot_prefix] = ...
    parse_copy_finetune_prototxt(...
    opts.finetune_net_def_file, opts.finetune_rst_dir);

opts.finetune_net_def_file = fullfile(opts.finetune_rst_dir, solver_file);
assert(exist(opts.finetune_net_def_file,'file')>0)

disp(opts)

% set the training dataset options
image_db_train.dataset   = 'pascal';
image_db_train.image_set = opts.train_set;
image_db_train.voc_year  = opts.voc_year_train;
image_db_train.proposals_method= opts.proposals_method_train;
image_db_train.use_flips = opts.train_use_flips;

% set the validation dataset options
image_db_val.dataset   = 'pascal';
image_db_val.image_set = opts.val_set;
image_db_val.voc_year  = opts.voc_year_val;
image_db_val.proposals_method= opts.proposals_method_val;
image_db_val.use_flips = opts.val_use_flips;

voc_path      = [pwd, '/datasets/VOC%s/'];
voc_path_year = sprintf(voc_path, '2007');
VOCopts       = initVOCOpts(voc_path_year,'2007');
classes       = VOCopts.classes;

opts.display_step   = 250; % display training progress every 250 iterations
opts.test_iter      = 2 * opts.display_step; % 500 iterations
opts.test_interval  = 20 * opts.display_step; % 5000 iterations. It must be the same as the test_interval parameter in the solver file
opts.num_classes    = length(classes); % number of categories of pascal dataset

% create struct pooler that contains the pooling parameters and the region type
pooler_loc = load_pooling_params(opts.vgg_pool_params_def, ...
    'scale_inner', 0, 'scale_outer', opts.scale_ratio, 'half_bbox', []);

model = struct;
model.pooler_loc = pooler_loc;
% mean pixel value per color channel for image pre-processing before 
% feeding it to the VGG16 convolutional layers.
model.mean_pix   = [103.939, 116.779, 123.68]; 
model.scales     = opts.scales;   % image scales that will be used during training
model.max_size   = opts.max_size; % the maximum size of the image

% localization parameters
model.loc_params = struct;
model.loc_params.loc_type    = opts.loc_type;
model.loc_params.resolution  = opts.resolution;
model.loc_params.scale_ratio = opts.scale_ratio;
model.loc_params.num_classes = opts.num_classes;

if ~isempty(opts.solverstate) 
    % set the full solverstate file path
    opts.solver_state_file = fullfile(opts.finetune_rst_dir, [opts.solverstate, '.solverstate']);
    assert(exist(opts.solver_state_file,'file')>0);
end
if opts.save_mat_model_only
    finetuned_model_path = fullfile(opts.finetune_rst_dir, [opts.finetuned_modelname,'.caffemodel']);
else
    caffe.reset_all();
    caffe_set_device( opts.gpu_id );
    
    % start training the LocNet model (or the CNN-based bounding box 
    % regression model in case the loc_type is 'bboxreg')
    finetuned_model_path = train_LocNet_model(...
        image_db_train, image_db_val, model, opts);
    
    diary off;
    caffe.reset_all();  
end


% prepare mat file of model
assert(exist(finetuned_model_path,'file')>0);
[~,filename,ext]     = fileparts(finetuned_model_path);
finetuned_model_path = ['.',filesep,filename,ext];

deploy_def_deploy_file_src  = fullfile(pwd, 'model-defs', def_deploy_filename);
deploy_def_deploy_file_dst  = fullfile(opts.finetune_rst_dir, 'localization_model.prototxt');
copyfile(deploy_def_deploy_file_src, deploy_def_deploy_file_dst);

[~,b,c] = fileparts(deploy_def_deploy_file_dst);
deploy_def_deploy_file_dst = ['./',b,c];

model.net_def_file     = deploy_def_deploy_file_dst;
model.net_weights_file = {finetuned_model_path};
model.preds_loc_out_blob = 'preds_loc'; % name of the output blob of the localization network
model.pooler           = model.pooler_loc;
model                  = rmfield(model,'pooler_loc');
model.classes          = classes;
model_filename         = fullfile(opts.finetune_rst_dir, 'localization_model.mat');
save(model_filename, 'model');

end
