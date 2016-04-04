function caffe_reshape_net_as_input(net, inputs)
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

input_size = cellfun(@(x) size(x), inputs, 'UniformOutput', false);

input_blob_names = net.inputs;
for i = 1:length(input_size)
    size_this = net.blobs(input_blob_names{i}).shape();
    input_size_this = ones(size(size_this));
    input_size_this(1:length(input_size{i})) = input_size{i};
    input_size{i} = input_size_this;
end

caffe_reshape_net(net, input_size);
end
