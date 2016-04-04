function [outputs, out_blob_names_total] = run_region_based_net_on_img(...
    model, image, bboxes, out_blob_names_extra)
% run_region_based_net_on_img: applies on the candidate bounding boxes
% (bboxes) and on the image the provided region-based CNN network (model).
% 
% INPUTS:
% 1) model:  (type struct) the bounding box recognition model
% 2) image:  a [Height x Width x 3] uint8 matrix with the image 
% 3) bboxes: a N x 4 array with the bounding box coordinates; each row is 
% the oordinates of a bounding box in the form of [x0,y0,x1,y1] where 
% (x0,y0) is tot-left corner and (x1,y1) is the bottom-right corner. N is 
% the number of bounding boxes.
% 4) out_blob_names_extra: (optional) a cell array with network blob names 
% (in the form of strings) of which the data will be returned from the 
% function. By default the function will always return the data of the
% network output blobs.
%
% OUTPUTS:
% 1) outputs: a NB x 1 cell array with the data of the inquired blobs; the
% i-th element outputs{i} has the data of the network blob with name 
% out_blob_names_total{i}.
% 2) out_blob_names_total: a NB x 1 cell array with the inquired blob name
% (in the form of strings)
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


if ~exist('out_blob_names_extra', 'var'), out_blob_names_extra = {}; end
assert(iscell(out_blob_names_extra));

% max_rois_num_in_gpu: maximum number of regions that will be given in one
% go in the network such that they could fit in the GPU memory
max_rois_num_in_gpu  = model.max_rois_num_in_gpu; 
out_blob_names       = model.net.outputs; % name of the network's output blobs
num_out_blobs        = length(out_blob_names); 
num_out_blobs_extra  = length(out_blob_names_extra);
num_out_blobs_total  = num_out_blobs + num_out_blobs_extra;
out_blob_names_total = [out_blob_names(:); out_blob_names_extra(:)];

image_size = size(image);
% get the image blob(s) that will be fed as input to the caffe network
[im_blob, im_scales] = get_image_blob(model, image);

% get the region blobs that will be given as input to caffe network
[rois_blob, scale_ids, ~, inv_order] = map_boxes_to_regions_wrapper(...
    model.pooler, bboxes, image_size, im_scales);

num_scales = length(model.scales);

% seperate the regions on their corresponding scale and for each scale
% divide its corresponding regions in chunks of maximum size of max_rois_num_in_gpu
[rois_blob_per_scale, num_chunks_per_scale, num_rois_per_scale] = ...
    divide_regions_in_chunks_per_scale(rois_blob, scale_ids, num_scales, max_rois_num_in_gpu);

total_num_chunks = sum(num_chunks_per_scale);

outputs = cell(num_out_blobs_total,1);
outputs = cellfun(@(x) cell([1, 1, 1, total_num_chunks]), outputs, 'UniformOutput', false);
c = 0;
for s = 1:num_scales

    % fed the image blob of the s-th scale and the regions that correspond
    % to this scale on the network and get the output
    outputs_this_scale = run_region_based_net_one_scale_effiently(model, ...
        out_blob_names_total, im_blob(s), rois_blob_per_scale{s}, ...
        num_rois_per_scale(s), num_chunks_per_scale(s), max_rois_num_in_gpu);    
    
    for j = 1:num_out_blobs_total
        outputs{j}(c +(1:num_chunks_per_scale(s))) = outputs_this_scale{j}(:);           
    end
    c = c + num_chunks_per_scale(s);
end

% format appropriately the output per blob
outputs = format_outputs(outputs, num_rois_per_scale, total_num_chunks, inv_order);
end

function [im_blob, im_scales] = get_image_blob(model, image)
[im_blob, im_scales] = arrayfun(@(x) ...
    prepare_img_blob(image, x, model.mean_pix, model.max_size), ...
    model.scales, 'UniformOutput', false);    
im_scales = cell2mat(im_scales);
end  

function caffe_reshape_net_as_this_input(net, inputs, num_rois)
input_size = cellfun(@(x) size(x), inputs, 'UniformOutput', false);

% a very stupid fix. deal with it in better way later.
if num_rois == 1
    for i = 2:length(input_size)
        input_size{i} = [input_size{i}, 1];
    end
end

input_blob_names = net.inputs;
for i = 1:length(input_size)
    size_this = net.blobs(input_blob_names{i}).shape();
    input_size_this = ones(size(size_this));
    input_size_this(1:length(input_size{i})) = input_size{i};
    input_size{i} = input_size_this;
end

caffe_reshape_net(net, input_size);
end

function [rois_blob, scale_ids, order, inv_order] = map_boxes_to_regions_wrapper(...
    region_params, bboxes, image_size, im_scales)

num_region_types = length(region_params);
rois_blob = cell(1,num_region_types);
scale_ids = cell(1,num_region_types);
% analyze each bounding box to one or more type of regions
for r = 1:num_region_types 
    rois_blob{r} = map_boxes_to_regions_mult_scales(...
        region_params(r), bboxes, image_size, im_scales);   
    scale_ids{r} = rois_blob{r}(:,1);
end

[scale_ids{1},order] = sort(scale_ids{1},'ascend');
[~,inv_order] = sort(order,'ascend');

for r = 1:num_region_types
    rois_blob{r}      = rois_blob{r}(order,:);
    scale_ids{r}      = rois_blob{r}(:,1);
    rois_blob{r}(:,1) = 1;
    rois_blob{r}      = rois_blob{r} - 1; % to c's index (start from 0)
    rois_blob{r}      = single(permute(rois_blob{r}, [3, 4, 2, 1]));
end
end

function [rois_blob_per_scale, num_chunks_per_scale, num_rois_per_scale] = ...
    divide_regions_in_chunks_per_scale(rois_blob, scale_ids, num_scales, max_rois_num_in_gpu)
num_region_types     = length(rois_blob);
rois_blob_per_scale  =  cell(num_scales,1);
num_chunks_per_scale = zeros(num_scales,1);
num_rois_per_scale   = zeros(num_scales,1);
for s = 1:num_scales
    this_scales_mask        = scale_ids{1} == s;
    num_rois_per_scale(s)   = sum(this_scales_mask);
    num_chunks_per_scale(s) = ceil(num_rois_per_scale(s) / max_rois_num_in_gpu);
    rois_blob_per_scale{s}  = cell(1,num_region_types);
    for r = 1:num_region_types
        rois_blob_per_scale{s}{r} = rois_blob{r}(:,:,:,this_scales_mask);
    end    
end
end

function outputs = run_region_based_net_one_scale_effiently(model, ...
    out_blob_names, im_blob, rois_blob, num_rois, num_chunks, max_rois_num_in_gpu)

outputs = cell(length(out_blob_names),1);
outputs = cellfun(@(x) cell([1, 1, 1, num_chunks]), outputs, 'UniformOutput', false);
for i = 1:num_chunks
    sub_ind_start = 1 + (i-1) * max_rois_num_in_gpu;
    sub_ind_end   = min(num_rois, i * max_rois_num_in_gpu);
    sub_num_rois  = sub_ind_end-sub_ind_start+1;
    sub_rois_blob = cell(1,length(rois_blob)); % the regions of this chunk that will be fed to the network
    for r = 1:length(rois_blob)
        sub_rois_blob{r} = rois_blob{r}(:, :, :, sub_ind_start:sub_ind_end);
    end

    % the inputs to the network for this chunk
    net_inputs = [im_blob, sub_rois_blob];
    caffe_reshape_net_as_this_input(model.net, net_inputs, sub_num_rois);
    if i == 1
        caffe_set_input(model.net, net_inputs);
        model.net.forward_prefilled();
    else
        input_blob_names = model.net.inputs; 
        % skip extracting the convolutional features of the same image
        % again.
        input_blob_names = input_blob_names(2:end);
        net_inputs       = net_inputs(2:end);
        caffe_set_blobs_data(model.net, net_inputs, input_blob_names);
        startLayerIdx = get_start_layer_idx_of_blobs(model.net, input_blob_names);
        model.net.forward_prefilled_from(startLayerIdx);
    end 

    for j = 1:length(out_blob_names)
        outputs{j}{i} = model.net.blobs(out_blob_names{j}).get_data();           
    end
end 
end

function outputs = format_outputs(outputs, num_rois_per_scale, total_num_chunks, inv_order)
% properly format the outputs

num_out_blobs_total = length(outputs);
for j = 1:num_out_blobs_total
    if total_num_chunks == 1
        outputs{j} = squeeze(cell2mat(outputs{j}));
    else
        chunk_sizes = cell2mat(cellfun(@(x) ...
            [size(x,1), size(x,2), size(x,3), size(x,4)], ...
            squeeze(outputs{j}(:)), 'UniformOutput', false));
        dim = find(any(chunk_sizes ~= 1,1),1,'last');
        if all(num_rois_per_scale <= 1)
            dim = dim + 1; 
        end
        shape = ones([1,dim]);
        shape(end)=total_num_chunks;
        outputs{j} = squeeze(cell2mat(reshape(outputs{j}, shape)));
        out_shape = size( outputs{j});
        if length(out_shape) == 1
            outputs{j} = outputs{j}(inv_order);
        elseif length(out_shape) == 2
            outputs{j} = outputs{j}(:,inv_order);
        elseif length(out_shape) == 3
            outputs{j} = outputs{j}(:,:,inv_order);
        elseif length(out_shape) == 4
            outputs{j} = outputs{j}(:,:,:,inv_order);
        end        
    end
end
end

function startLayerIdx = get_start_layer_idx_of_blobs(net, blob_names)
num_blobs = length(blob_names);
min_layer_id_per_blob = zeros(num_blobs,1);
for i = 1:num_blobs
    layer_ids = net.layer_ids_with_input_blob(blob_names{i});
    assert(~isempty(layer_ids));
    min_layer_id_per_blob(i) = min(layer_ids);
end
startLayerIdx = min(min_layer_id_per_blob);
end