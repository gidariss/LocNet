function blobs_size = caffe_get_blobs_size(net, blob_names)
% 
% This file is part of the code that implements the following ICCV2015 accepted paper:
% title: "Object detection via a multi-region & semantic segmentation-aware CNN model"
% authors: Spyros Gidaris, Nikos Komodakis
% institution: Universite Paris Est, Ecole des Ponts ParisTech
% Technical report: http://arxiv.org/abs/1505.01749
% code: https://github.com/gidariss/mrcnn-object-detection
% 
% 
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2015 Spyros Gidaris
% 
% "Object detection via a multi-region & semantic segmentation-aware CNN model"
% Technical report: http://arxiv.org/abs/1505.01749
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------

num_blobs  = length(blob_names);
blobs_size = cell(num_blobs,1);
for i = 1:num_blobs
    size_this     = net.blobs(blob_names{i}).shape;
    blobs_size{i} = ones(1,4);
    blobs_size{i}((end-length(size_this)+1):end) = size_this;
end
end
