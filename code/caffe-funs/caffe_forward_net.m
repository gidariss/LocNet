function [outputs, out_blob_names_total] = caffe_forward_net(net, input, out_blob_names_extra)
% 
% This file is part of the code that implements the following ICCV2015 accepted paper:
% title: "Object detection via a multi-region & semantic segmentation-aware CNN model"
% authors: Spyros Gidaris, Nikos Komodakis
% institution: Universite Paris Est, Ecole des Ponts ParisTech
% Technical report: http://arxiv.org/abs/1505.01749
% code: https://github.com/gidariss/mrcnn-object-detection
% c
% 
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2015 Spyros Gidaris
% 
% "Object detection via a multi-region & semantic segmentation-aware CNN model"
% Technical report: http://arxiv.org/abs/1505.01749
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------


if ~exist('out_blob_names_extra', 'var'), out_blob_names_extra = {}; end
assert(iscell(out_blob_names_extra));

input_size  = caffe_get_blobs_size(net, net.inputs);
assert(numel(input_size) == numel(input));
size_in     = size(input{1});
num_feats   = size_in(end);

out_blob_names       = net.outputs;
num_out_blobs        = length(out_blob_names);
num_out_blobs_extra  = length(out_blob_names_extra);
num_out_blobs_total  = num_out_blobs + num_out_blobs_extra;
out_blob_names_total = [out_blob_names(:); out_blob_names_extra(:)];

output_size  = caffe_get_blobs_size(net, out_blob_names_total);
outputs      = cell(num_out_blobs_total,1);
out_feat_dim = zeros(num_out_blobs_total,1);

batch_size  = input_size{1}(4);
num_batches = ceil(num_feats/batch_size);

for i = 1:num_out_blobs_total
    out_feat_dim(i) = prod(output_size{i}(1:(end-1)));
    if output_size{i}(end) == batch_size
        outputs{i} = zeros(out_feat_dim(i), num_feats, 'single');
    else
        outputs{i} = zeros(out_feat_dim(i), output_size{i}(end) * num_batches, 'single');
    end
end

for i = 1:num_batches
    start_idx = (i-1) * batch_size + 1;
    stop_idx  = min(i * batch_size, num_feats);
    batch_size_this = stop_idx - start_idx + 1;
    
    % forward propagate batch of region images
    out = net.forward(prepare_batch(input, input_size, start_idx, stop_idx));
    for j = (num_out_blobs+1):num_out_blobs_total
        out{j} = net.blobs(out_blob_names_total{j}).get_data; 
    end
    
    for j = 1:num_out_blobs_total
        out{j} = reshape(out{j}, [out_feat_dim(j), output_size{j}(end)]);
        if output_size{j}(end) == batch_size
            if (i == num_batches)
                out{j} = out{j}(:,1:batch_size_this); 
            end
            outputs{j}(:,start_idx:stop_idx) = out{j};
        else
            start_idx0 = (i-1) * output_size{j}(end) + 1;
            stop_idx0  = i * output_size{j}(end);
            outputs{j}(:,start_idx0:stop_idx0) = out{j};
        end
    end
end

end

function batch = prepare_batch(input, input_size, start_idx, stop_idx)
assert(numel(input_size) == numel(input));
batch_size_this   = stop_idx - start_idx + 1;

batch = cell(1,numel(input_size));
for i = 1:numel(input_size)
    batch{i}          = zeros(input_size{i}, 'single');
    reshape_vector    = input_size{i};
    reshape_vector(4) = batch_size_this;
    batch{i}(:,:,:,1:batch_size_this) = reshape(input{i}(:,start_idx:stop_idx), reshape_vector);
end
end
