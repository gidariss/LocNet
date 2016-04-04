function net = set_net_layer_weights( net, layer_name, weights )
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

num_params = length(net.layers(layer_name).params);
assert(num_params == length(weights));
assert(iscell(weights))
for p = 1:num_params
    param_shape = net.layers(layer_name).params(p).shape;
    for d = 1:length(param_shape)
        assert(all(param_shape(d) == size(weights{p},d)));
    end
    net.layers(layer_name).params(p).set_data(weights{p});
end
end

