function net = caffe_load_model( net_def_file, net_weights_file, phase )
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

if ~exist('phase', 'var'),  phase = 'test'; end
assert(exist(net_def_file,'file')>0);
net = caffe.Net(net_def_file, phase); % create net but not load weights

assert(iscell(net_weights_file))
num_weights_files = length(net_weights_file);
for i = 1:num_weights_files
    assert(exist(net_weights_file{i},'file')>0);
    net.copy_from(net_weights_file{i});
end

end
