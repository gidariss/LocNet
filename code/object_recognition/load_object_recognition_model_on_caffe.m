function model = load_object_recognition_model_on_caffe(model, use_detection_svms, model_dir, model_phase)
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


if (~exist('model_phase','var') > 0), model_phase = 'test'; end

curr_dir = pwd;
cd(model_dir);
model.net = caffe_load_model(model.net_def_file, model.net_weights_file, model_phase);

if use_detection_svms
    assert(exist(model.svm_weights_file, 'file')>0);
    weights = read_svm_detection_weights(model.svm_weights_file);
    model.net = set_net_layer_weights(model.net, model.svm_layer_name, weights);
end

cd(curr_dir);
end