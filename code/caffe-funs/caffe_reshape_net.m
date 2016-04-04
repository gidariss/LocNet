function net = caffe_reshape_net(net, input_new_size)
% 
% This file is part of the code that implements the following ICCV2015 accepted paper:
% title: "Object detection via a multi-region & semantic segmentation-aware CNN model"
% authors: Spyros Gidaris, Nikos Komodakis
% institution: Universite Paris Est, Ecole des Ponts ParisTech
% Technical report: http://arxiv.org/abs/1505.01749
% code: https://github.com/gidariss/mrcnn-object-detection
% 
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2015 Spyros Gidaris
% 
% "Object detection via a multi-region & semantic segmentation-aware CNN model"
% Technical report: http://arxiv.org/abs/1505.01749
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------

input_blob_names = net.inputs;
num_inputs = length(input_blob_names);
assert(numel(input_new_size) == num_inputs);
for i = 1:num_inputs
    net.blobs(input_blob_names{i}).reshape(input_new_size{i}); % reshape input blob
end
net.reshape();
end
