function finetuned_model_path = train_LocNet_model(...
        image_db_train, image_db_val, model, conf)
% train_LocNet_model: trains the LocNet network
% 
% INPUTS:
% 1) image_db_train: struct with the specifiers of the training dataset
% 2) image_db_val: struct with the specifiers of the testing dataset
% 3) model: struct with the localization model
% 4) conf: struct with the configuration options of the training process
%
% OUTPUT:
% 1) finetuned_model_path: string with the path to the trained caffe model
% produced from the training procedure.
% 
% This file is part of the code that implements the following paper:
% Title      : "LocNet: Improving Localization Accuracy for Object Detection"
% Authors    : Spyros Gidaris, Nikos Komodakis
% Institution: Universite Paris Est, Ecole des Ponts ParisTech
% ArXiv link : http://arxiv.org/abs/1511.07763
% code       : https://github.com/gidariss/LocNet
% 
%
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2016 Spyros Gidaris
% 
% Title     : "LocNet: Improving Localization Accuracy for Object Detection"
% ArXiv link: http://arxiv.org/abs/1511.07763
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------

mkdir_if_missing(conf.finetune_rst_dir);
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
log_file  = fullfile(conf.finetune_rst_dir, 'output', ['LocNet_' timestamp, '.txt']);
mkdir_if_missing(fileparts(log_file));
finetuned_model_path = '';
diary(log_file);
    
image_db_train_file = fullfile(conf.finetune_rst_dir,'train_file.mat');
image_db_val_file   = fullfile(conf.finetune_rst_dir,'val_file.mat');

% load training dataset
try
    load(image_db_train_file,'image_db_train');
catch
    image_db_train = load_image_dataset(...
        'dataset',          image_db_train.dataset, ...
        'image_set',        image_db_train.image_set, ...
        'voc_year',         image_db_train.voc_year, ...
        'proposals_method', image_db_train.proposals_method,...
        'use_flips',        image_db_train.use_flips);
    image_db_train.cache_regions_dir = ...
        fullfile(pwd,'cached_regions', ...
        ['LocNet_model_', image_db_train.image_set_name, '_', image_db_train.proposals_suffix,filesep]);
    mkdir_if_missing(image_db_train.cache_regions_dir);

    fprintf('Prepare training data:\n')
    image_db_train = setup_image_db(image_db_train, conf);
    fprintf('Done\n')
    save(image_db_train_file,'image_db_train','-v7.3');
end

% load testing dataset
try
    load(image_db_val_file,'image_db_val');
catch
    image_db_val = load_image_dataset(...
        'dataset',          image_db_val.dataset, ...
        'image_set',        image_db_val.image_set, ...
        'voc_year',         image_db_val.voc_year, ...
        'proposals_method', image_db_val.proposals_method,...
        'use_flips',        image_db_val.use_flips);
    
    image_db_val.cache_regions_dir = ...
        fullfile(pwd,'cached_regions', ...
        ['LocNet_model_', image_db_val.image_set_name, '_', image_db_val.proposals_suffix,filesep]);
    mkdir_if_missing(image_db_val.cache_regions_dir);

    fprintf('Prepare validation data:\n')
    image_db_val = setup_image_db(image_db_val, conf);
    fprintf('Done\n')
    save(image_db_val_file,'image_db_val','-v7.3');
end

fprintf('conf:\n');
disp(conf);
fprintf('model:\n');
disp(model);
fprintf('pooler:\n');
disp(model.pooler_loc);
fprintf('model.loc_params:\n');
disp(model.loc_params);

current_dir = pwd;
cd(fileparts(conf.finetune_net_def_file))
% initialize caffe solver
[solver, iter_] = InitializeSolver(conf);

fprintf('Starting iteration %d\n', iter_);

prev_rng = seed_rand();
last_finetuned_model_prev_path = '';

rng(prev_rng);

iter_ = solver.iter();
mean_error_test = TestNet(conf, model, solver, image_db_val);
ShowState(iter_, NaN, mean_error_test);
if conf.test_only
    mean_error_test = TestNet(conf, model, solver, image_db_val);
    ShowState(iter_, NaN, mean_error_test);
    return; 
end

% start training
shuffled_inds   = {};
while(iter_ < conf.max_iter)
    % train the localization model (for conf.test_interval number of iterations)
    [mean_error_train, shuffled_inds] = TrainNet(conf, model, solver, image_db_train, shuffled_inds);
    % test the localization model (for conf.test_iter number of iterations)
    mean_error_test = TestNet( conf, model, solver, image_db_val);
    
    iter_ = solver.iter();
    ShowState(iter_, mean_error_train, mean_error_test);
    diary; diary; % flush diary
        
    last_finetuned_model_path = fullfile(conf.finetune_rst_dir, ...
        sprintf('%s_iter_%d.caffemodel', conf.snapshot_prefix, iter_));
    assert(exist(last_finetuned_model_path,'file')>0);
    fprintf('last save as %s\n', last_finetuned_model_path);
        
    % delete the previously saved model
    deletePreviousModel(last_finetuned_model_prev_path);
    last_finetuned_model_prev_path = last_finetuned_model_path;
end
cd(current_dir);
finetuned_model_path = last_finetuned_model_path;
fprintf('Optimization is finished\n')
end

function [solver, iter_] = InitializeSolver(conf)
solver = caffe.Solver(conf.finetune_net_def_file);
iter_   = 1;

if isfield(conf,'solver_state_file') && ~isempty(conf.solver_state_file)
    solver.restore(conf.solver_state_file);
    [directory,filename] = fileparts(conf.solver_state_file);
    model_file = [directory,filesep, filename, '.caffemodel'];
    solver.net.copy_from(model_file);
    solver.test_nets(1).copy_from(model_file);
    iter_   = solver.iter();    
    rng('shuffle')
else
    solver.net.copy_from(conf.net_file);
    solver.test_nets(1).copy_from(conf.net_file);
end
end

function deletePreviousModel(finetuned_model_prev_path)
if exist(finetuned_model_prev_path, 'file')
	delete(finetuned_model_prev_path);
end
[directory,filename] = fileparts(finetuned_model_prev_path);
finetuned_model_prev_path_solver_state = [directory,filesep, filename,'.solverstate'];
if exist(finetuned_model_prev_path_solver_state, 'file')
    delete(finetuned_model_prev_path_solver_state);
end
end

function ShowState(iter, train_error, test_error)
fprintf('\n------------------------- Iteration %d -------------------------\n', iter);
fprintf('Error - Training : %.4f - Testing : %.4f\n', train_error, test_error);
end

function [mean_loc_error] = TestNet(conf, model, solver, image_db_val)
fprintf('Testing: \n'); th = tic;


% sample testing images and testing regions
conf.save_data_test_file = fullfile(conf.finetune_rst_dir, ...
    sprintf('test_file_ims_per_batch%d_batch_size%d_test_iter%d.mat', ...
    conf.ims_per_batch, conf.batch_size, conf.test_iter));
try 
    ld = load(conf.save_data_test_file);
    shuffled_inds_val    = ld.shuffled_inds_val;
    windows_bbox_loc_val = ld.windows_bbox_loc_val;
catch
    % sample test images
    shuffled_inds_val    = GenerateRandomMinibatch([], image_db_val, conf.ims_per_batch);
    shuffled_inds_val    = shuffled_inds_val(randperm(length(shuffled_inds_val), conf.test_iter));
    windows_bbox_loc_val = cell(length(shuffled_inds_val),1);
    % sample test regions from the test images
    for i = 1:length(shuffled_inds_val)
        windows_bbox_loc_val{i} = SampleImageRegionsForLoc(conf, image_db_val.images(shuffled_inds_val{i}));
    end
    save(conf.save_data_test_file, 'shuffled_inds_val', 'windows_bbox_loc_val');
end

mean_loc_error = 0; 
mean_loc_loss  = 0;

bbox_inits_loc  = cell(length(shuffled_inds_val),1);
predictions_loc = cell([1,1,1,length(shuffled_inds_val)]);
bbox_target_loc = cell(length(shuffled_inds_val),1);
for i = 1:length(shuffled_inds_val)
    image_roidb_this_minibath = image_db_val.images(shuffled_inds_val{i}); % validation image
    windows_bbox_loc_this = windows_bbox_loc_val{i}; % sampled regions in the validation image
    
    % get image caffe blobs
    [im_blob, im_scales, image_sizes] = GetImageBlob(model, image_roidb_this_minibath);
    
    % get regions caffe blobs and target blobs
    [loc_blobs, bbox_target_loc{i}, bbox_inits_loc{i}] = ...
        GetLocNetMinibatchData(model.loc_params, model, image_roidb_this_minibath, ...
        windows_bbox_loc_this, im_scales, image_sizes);
    % set network input
    net_inputs = [{im_blob}, loc_blobs];
    
    caffe_reshape_net_as_input(solver.test_nets(1), net_inputs);
    caffe_set_input(solver.test_nets(1), net_inputs);
    solver.test_nets(1).forward_prefilled();
    
    % get output
    mean_loc_loss        = mean_loc_loss + solver.test_nets(1).blobs('loss_loc').get_data();
    predictions_loc{i}   = solver.test_nets(1).blobs('preds_loc').get_data();
    loc_accuracy         = get_loc_accuracy(model.loc_params, predictions_loc{i}, loc_blobs);
    mean_loc_error       = mean_loc_error + (1-loc_accuracy);
end


[mAR_before, mAR_after, mAR_unbound] = ...
    get_mean_Average_Recall_stats(bbox_inits_loc, bbox_target_loc, predictions_loc, model.loc_params);
mean_loc_error = mean_loc_error / length(shuffled_inds_val);
mean_loc_loss  = mean_loc_loss  / length(shuffled_inds_val);

fprintf('elapsed time %.2f sec | localization: loss %.4f error %.4f mAR %.2f --> mAR %.2f (upper bound %.2f)\n', ...
    toc(th), mean_loc_loss, mean_loc_error, ...
    mAR_before*100, mAR_after*100, mAR_unbound*100);
end

function [loc_accuracy] = get_loc_accuracy(conf, predictions_loc, loc_blobs)
% compute the accuracy of the predicted probability vectors w.r.t. the 
% target probability vectors (only for the LocNet models).

switch conf.loc_type
    case {'inout','borders','combined'}
        target_vectors = loc_blobs{2};
        target_weights = loc_blobs{3};
        mask           = target_weights > 0;
        preds_vectors  = single(predictions_loc > 0.5);
        loc_accuracy   = sum(target_vectors(mask) == preds_vectors(mask)) / sum(mask(:));         
    otherwise
        loc_accuracy = 0;
end

end

function [mAR_before, mAR_after, mAR_unbound] = ...
    get_mean_Average_Recall_stats(bbox_inits_loc, bbox_target_loc, predictions_loc, conf)

bbox_target_loc = cell2mat(bbox_target_loc);
bbox_inits_loc  = cell2mat(bbox_inits_loc);
class_indices   = bbox_inits_loc(:,5);
bbox_inits_loc  = bbox_inits_loc(:,1:4);

switch conf.loc_type
    case 'bboxreg'
        predictions_loc = squeeze(cell2mat(predictions_loc(:)'));

        % permute from [(4*num_classes) x NumBoxes] -> [NumBoxes x (4*num_classes)] 
        predictions_loc = single(permute(predictions_loc, [2, 1]));        
        % reshape from [NumBoxes x (4*num_classes)] -> [NumBoxes x 4 x num_classes]
        predictions_loc = reshape(predictions_loc, ...
            [size(predictions_loc,1), 4, conf.num_classes]);

        pred_bbox_loc = decode_reg_vals_to_bbox_targets(...
            bbox_inits_loc, predictions_loc, class_indices);
        
        if nargout >= 3
            % find the ideal predictions... 
            target_reg_values = encode_bbox_targets_to_reg_vals(...
                bbox_inits_loc, bbox_target_loc, conf.num_classes);

            pred_bbox_loc_ideal = decode_reg_vals_to_bbox_targets(...
                bbox_inits_loc, target_reg_values, class_indices);

            % upper bound on the mAR with the CNN-based bounding box
            % regression model. Normaly mAR = 1.0.
            mAR_unbound = compute_mAR_of_bboxes(bbox_inits_loc, pred_bbox_loc_ideal(:,1:4), ...
                bbox_target_loc(:,1:4), class_indices, conf.num_classes); 
        end        
    case {'inout','borders','combined'}
        predictions_loc = squeeze(cell2mat(predictions_loc));
        
        pred_bbox_loc = decode_loc_probs_to_bbox_targets(...
            bbox_inits_loc, class_indices, predictions_loc, conf);

        
        if nargout >= 3
            % find the ideal predictions... 
            target_loc_prob_vectors = encode_bbox_target_to_loc_probs(...
                bbox_inits_loc, bbox_target_loc, conf);

            pred_bbox_loc_ideal = decode_loc_probs_to_bbox_targets(...
                bbox_inits_loc, class_indices, target_loc_prob_vectors, conf);

            % upper bound on the mAR with LocNet model. Because of the
            % quantization of the search region in M horizontal stripes and
            % M vertical stripes the mAR < 1.0
            mAR_unbound = compute_mAR_of_bboxes(bbox_inits_loc, pred_bbox_loc_ideal(:,1:4), ...
                bbox_target_loc(:,1:4), class_indices, conf.num_classes); 
        end
    otherwise
        error('Invalid localization type %s',conf.loc_type)
end

mAR_before = compute_mAR_of_bboxes(bbox_inits_loc, bbox_inits_loc, ...
            bbox_target_loc(:,1:4), class_indices, conf.num_classes);
        
mAR_after = compute_mAR_of_bboxes(bbox_inits_loc, pred_bbox_loc(:,1:4), ...
            bbox_target_loc(:,1:4), class_indices, conf.num_classes);
end

function [mean_loc_error, shuffled_inds] = TrainNet(conf, model, solver, image_db_train, shuffled_inds)
mean_loc_error = 0; 
mean_loc_loss = 0; 
fprintf('Training:\n'); th = tic;
for i = 1:conf.test_interval
    % sample training images
    [shuffled_inds, sub_db_inds] = GenerateRandomMinibatch(...
        shuffled_inds, image_db_train, conf.ims_per_batch);
    image_roidb_this_minibath = image_db_train.images(sub_db_inds);
    
    % get the training image blobs that will be fed to caffe  
    [im_blob, im_scales, image_sizes] = GetImageBlob(model, image_roidb_this_minibath);
    
    % sample training regions for localization
    windows_bbox_loc = SampleImageRegionsForLoc(conf, image_roidb_this_minibath);
    
    % get training regions caffe blobs and the target blobs
    loc_blobs = GetLocNetMinibatchData(model.loc_params, model, ...
        image_roidb_this_minibath, windows_bbox_loc, im_scales, image_sizes);
    % set network input
    net_inputs = [{im_blob}, loc_blobs];
    
    caffe_reshape_net_as_input(solver.net, net_inputs);
    caffe_set_input(solver.net, net_inputs);
    solver.step(1);

    % get output
    mean_loc_loss   = mean_loc_loss + solver.net.blobs('loss_loc').get_data();
    predictions_loc = solver.net.blobs('preds_loc').get_data();
    loc_accuracy    = get_loc_accuracy(model.loc_params, predictions_loc, loc_blobs);
    mean_loc_error  = mean_loc_error + (1-loc_accuracy); 
    
    if (mod(i, conf.display_step) == 0)
        fprintf('%2d/%d:\telapsed time %.2f sec --- error %.4f loss %.4f\n', ...
            i, conf.test_interval, toc(th), mean_loc_error/i, mean_loc_loss/i);
        th = tic;
        diary; diary; % flush diary
    end
end
mean_loc_error = mean_loc_error/i;
end

function [mAR, AR] = compute_mAR_of_bboxes(all_bbox_init, all_bbox_pred, all_bbox_targets, class_ids, num_classes)
% compute the mAR of the predicted bounding boxes w.r.t. the target
% bounding boxes.

AR = zeros(1,num_classes);
fprintf('AveRecall: ')
for c = 1:num_classes
    is_this_cls  = class_ids==c;
    bbox_pred    = all_bbox_pred(is_this_cls,:);
    bbox_targets = all_bbox_targets(is_this_cls,:);
    is_not_gt    = ~all(all_bbox_init(is_this_cls,:) == bbox_targets, 2);
    bbox_pred    = bbox_pred(is_not_gt,:);
    bbox_targets = bbox_targets(is_not_gt,:);
    AR(c)        = compute_ave_recall_of_bbox(bbox_pred, bbox_targets);
    fprintf('[%d]=%.2f(%d) ',c, AR(c),size(bbox_targets,1));
end

fprintf('\n')
mAR = mean(AR);
end

function [shuffled_inds, sub_inds] = GenerateRandomMinibatch(...
    shuffled_inds, image_db, ims_per_batch)
% This function comes from the Faster R-CNN code and the matlab
% re-implementation of Fast-RCNN (https://github.com/ShaoqingRen/faster_rcnn)

% shuffle training data per batch
if isempty(shuffled_inds)
    fprintf('!!!!!!!!!!!!!!!!!!!!!!!!New Epoch!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
    % make sure each minibatch, only has horizontal images or vertical
    % images, to save gpu memory
    hori_image_inds = image_db.image_sizes(:,2) >= image_db.image_sizes(:,1);
    vert_image_inds = ~hori_image_inds;
    hori_image_inds = single(find(hori_image_inds));
    vert_image_inds = single(find(vert_image_inds));

    % random perm
    lim = floor(length(hori_image_inds) / ims_per_batch) * ims_per_batch;
    hori_image_inds = hori_image_inds(randperm(length(hori_image_inds), lim));
    lim = floor(length(vert_image_inds) / ims_per_batch) * ims_per_batch;
    vert_image_inds = vert_image_inds(randperm(length(vert_image_inds), lim));

    % combine sample for each ims_per_batch 
    hori_image_inds = reshape(hori_image_inds, ims_per_batch, []);
    vert_image_inds = reshape(vert_image_inds, ims_per_batch, []);

    shuffled_inds = [hori_image_inds, vert_image_inds];
    shuffled_inds = shuffled_inds(:, randperm(size(shuffled_inds, 2)));
    shuffled_inds = num2cell(shuffled_inds, 1);
end

if nargout > 1
    % generate minibatch training data
    sub_inds = shuffled_inds{1};
    assert(length(sub_inds) == ims_per_batch);
    shuffled_inds(1) = [];
end    

end

function sampled_regions_loc = SampleImageRegionsForLoc(conf, image_roidb)

num_images = length(image_roidb);
assert(mod(conf.batch_size, num_images) == 0, ...
    sprintf('num_images %d must divide BATCH_SIZE %d', num_images, conf.batch_size));

rois_per_image  = conf.batch_size / num_images; % regions per image
num_images  = length(image_roidb);
sampled_regions_loc = cell(num_images, 1); 
for i = 1:num_images
    num_regions = size(image_roidb(i).windows_bbox_loc,1);
    % sample rois_per_image number of regions from this image
    indices = randperm(num_regions, min(rois_per_image, num_regions));
    windows_bbox_loc_this = image_roidb(i).windows_bbox_loc(indices,:);
    region_inds_loc = windows_bbox_loc_this(:,1); 
    % load the entire set of training regions of this image (cached in disk)
    regions = load_img_regions(image_roidb(i).regions); 
    % keep only the regions of this minibatch
    bboxes = regions(region_inds_loc,1:4);
    sampled_regions_loc{i} = single([bboxes,windows_bbox_loc_this(:,2:3)]);
end
end

function [im_blob, im_scales, image_sizes, im_scaled_size] = GetImageBlob(model, image_roidb)
% This function comes from the Faster R-CNN code and the matlab
% re-implementation of Fast-RCNN (https://github.com/ShaoqingRen/faster_rcnn)

num_images     = length(image_roidb);
random_scale_inds = randi(length(model.scales), num_images, 1);
processed_imgs = cell(num_images, 1);
im_scales      = nan(num_images, 1);
image_sizes    = nan(num_images, 2);
im_scaled_size = nan(num_images, 2); 
for i = 1:num_images
    img = get_image(image_roidb(i).image_path);
    image_sizes(i,1) = size(img,1);
    image_sizes(i,2) = size(img,2);
    target_size = model.scales(random_scale_inds(i));
    [processed_imgs{i}, im_scales(i), im_scaled_size(i,:)] = prepare_img_blob(img, ...
        target_size, model.mean_pix, model.max_size);
end

im_blob = img_list_to_blob(processed_imgs);
end

function image_db = setup_image_db(image_db, conf)
num_imgs = length(image_db.image_paths);
image_db.all_windows_bbox_loc = cell(num_imgs, 1);
num_regions_fg_loc = zeros(num_imgs, 1);

cache_regions_dir = image_db.cache_regions_dir;
all_region_files  = strcat(cache_regions_dir, getImageIdsFromImagePaths( image_db.image_paths ),'.mat');

for i = 1:num_imgs
    % find for each image its training bounding boxes and their
    % corresponding target bounding boxes
    [image_db.all_regions{i}, image_db.all_windows_bbox_loc{i}, num_regions_fg_loc(i)] = ...
        setup_img_data(image_db.all_bbox_gt{i}, image_db.all_regions{i}, conf);

    % cache the training bounding boxes in the hard disk (for large 
    % datasets, such as MSCOCO, is in-efficient memory wise to keep the
    % training boxes in the memory)
    save_img_regions(image_db.all_regions{i}, all_region_files{i});
    image_db.all_regions{i} = [];
    tic_toc_print('setup img data %d / % d (%.2f)\n', i, num_imgs, 100*i/num_imgs);
end

image_db.all_regions = [];
image_db.all_regions = all_region_files;

is_valid = (num_regions_fg_loc > 0);

fprintf('valid: %d/%d (%.3f) \n', sum(is_valid),length(is_valid),sum(is_valid)/length(is_valid));

image_db.image_paths          = image_db.image_paths(is_valid);
image_db.all_regions          = image_db.all_regions(is_valid);
image_db.all_windows_bbox_loc = image_db.all_windows_bbox_loc(is_valid);
image_db.all_bbox_gt          = image_db.all_bbox_gt(is_valid);
num_regions_fg_loc            = num_regions_fg_loc(is_valid);

image_db.images = cell2struct(...
    [image_db.image_paths, image_db.all_regions, ...
     image_db.all_windows_bbox_loc, image_db.all_bbox_gt], ...
    {'image_path', 'regions', 'windows_bbox_loc', 'bbox_gt'}, 2);

image_db = rmfield(image_db, 'image_paths');
image_db = rmfield(image_db, 'all_regions');
image_db = rmfield(image_db, 'all_windows_bbox_loc');
image_db = rmfield(image_db, 'all_bbox_gt');

fprintf('Total number of fg regions for localization: %d\n', sum(num_regions_fg_loc))

num_classes = length(image_db.classes);
fg_per_class = zeros(num_classes,1);
gt_per_class = zeros(num_classes,1);
labels_per_img = -ones(num_imgs,num_classes,'single');
for i = 1:num_imgs
    class_ids         = image_db.images(i).windows_bbox_loc(:,end);
    fg_per_class_this = accumarray(class_ids, 1, [num_classes, 1]);
    fg_per_class      = fg_per_class+fg_per_class_this;
    gt_per_class_this = accumarray(image_db.images(i).bbox_gt(:,5),1,[num_classes, 1]);
    gt_per_class      = gt_per_class + gt_per_class_this;
    
    unique_class_ids = unique(class_ids);
    labels_per_img(i,unique_class_ids) = 1;
end

% print the number of training boxes and grount truth bounding boxes per
% category
fprintf('Regions / GTs per class:\n');
for i = 1:num_classes
    fprintf('Category %d: FG %d GT %d R %.2f\n',i, fg_per_class(i), gt_per_class(i), fg_per_class(i)/gt_per_class(i));
end

image_db.gt_per_class   = gt_per_class;
image_db.fg_per_class   = fg_per_class;
image_db.labels_per_img = labels_per_img;
end

function [bbox_proposals, windows_bbox_loc, num_regions_fg_loc] = ...
    setup_img_data(bbox_gt, bbox_proposals, conf)

num_classes           = conf.num_classes;
fg_threshold_bbox_loc = conf.fg_threshold_bbox_loc;

bbox_gt_label = bbox_gt(:,5);
bbox_all = [bbox_gt(:,1:4); bbox_proposals(:,1:4)];
overlaps = zeros(size(bbox_all,1), num_classes, 'single');

windows_bbox_loc = cell(num_classes, 1);
for class_idx = 1:num_classes
    this_class_ind = find(bbox_gt_label == class_idx);

    windows_bbox_loc{class_idx} = zeros(0,3,'single');
    if ~isempty(this_class_ind)
        overlap_class = single(boxoverlap(bbox_all(:,1:4), bbox_gt(this_class_ind,1:4)));
        [overlaps(:, class_idx), gt_index] = max(overlap_class, [], 2);
        
        this_class_pr_index = find(overlaps(:, class_idx) >= fg_threshold_bbox_loc); 
        if ~isempty(this_class_pr_index)
            this_class_gt_index = this_class_ind(gt_index);
            this_class_gt_index = this_class_gt_index(this_class_pr_index);
            windows_bbox_loc{class_idx} = ...
                single([this_class_pr_index(:), this_class_gt_index(:), ...
                ones(length(this_class_pr_index),1)*class_idx]);
        end
    end
end
windows_bbox_loc = single(cell2mat(windows_bbox_loc)); % label, overlap, bbox_init, bbox_target
num_regions_fg_loc = size(windows_bbox_loc,1);

% remove unecesary bbox proposals...
pr_ind = windows_bbox_loc(:,1);
[pr_ind_u, ~, pr_ind_new] = unique(pr_ind);
bbox_proposals = bbox_all(pr_ind_u,1:4);
assert(all(all(bbox_proposals(pr_ind_new,1:4)==bbox_all(pr_ind,1:4))))
windows_bbox_loc(:,1) = pr_ind_new;
end

function save_img_regions(regions, regions_file)
if ~(exist(regions_file,'file')>0) 
    save(regions_file, 'regions', '-v7.3');
end
end

function regions = load_img_regions(regions_file)
load(regions_file, 'regions');
end