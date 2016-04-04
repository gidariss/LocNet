function caffe_set_blobs_data(net, blobs, blob_names)
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

assert(length(blobs) == length(blob_names));
for i = 1:length(blobs)
    shape_this = net.blobs(blob_names{i}).shape;
    blobs{i}   = reshape(blobs{i}, shape_this);
    net.blobs(blob_names{i}).set_data(blobs{i}); 
end
end