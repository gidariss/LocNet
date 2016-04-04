function [output, output_blob_names] = caffe_get_output(net)
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

output_blob_names = net.outputs;
% output_size  = caffe_get_blobs_size(net, output_blob_names);
num_outputs  = length(output_blob_names);
output       = cell(num_outputs,1);

for i = 1:num_outputs
    output{i} = net.blobs(output_blob_names{i}).get_data; 
end
end

