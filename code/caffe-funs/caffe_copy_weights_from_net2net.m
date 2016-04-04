function net_dst = caffe_copy_weights_from_net2net( net_dst, net_src, layers_dst, layers_src, scl_factor, shape_strict )
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

if ~exist('scl_factor','var'),   scl_factor = 1; end
if ~exist('shape_strict','var'), shape_strict = true; end

assert(iscell(layers_dst))
assert(iscell(layers_src))
assert(length(layers_dst) == length(layers_src));

if numel(scl_factor) == 1
    scl_factor = scl_factor * ones(1,length(layers_dst));
end
scl_factor = single(scl_factor);

assert(length(layers_dst) == length(scl_factor));

for i = 1:length(layers_dst)
    try
        fprintf('Copying layer %d/%d  %s to %s:\n', i, length(layers_dst), layers_src{i}, layers_dst{i})

        num_params = min(length(net_dst.layers(layers_dst{i}).params),...
                         length(net_src.layers(layers_src{i}).params));
                     
        for p = 1:num_params
            param_shape_this_dst = net_dst.layers(layers_dst{i}).params(p).shape;
            param_shape_this_src = net_src.layers(layers_src{i}).params(p).shape;
            
            if shape_strict 
                assert(all(param_shape_this_dst == param_shape_this_src));
            else
                assert(prod(param_shape_this_dst) == prod(param_shape_this_src));
            end
            
            param_shape_this_dst4 = ones(1,4);
            param_shape_this_dst4((end-length(param_shape_this_dst)+1):end) = param_shape_this_dst;
            
            param_shape_this_src4 = ones(1,4);
            param_shape_this_src4((end-length(param_shape_this_src)+1):end) = param_shape_this_src;    
            
            scl_factor_this = scl_factor(i);
            
            fprintf('param[%d]:[%d, %d, %d, %d] --> [%d, %d, %d, %d] | scl_factor = %.4f  ', ...
                p, ...
                param_shape_this_src4(1), param_shape_this_src4(2), param_shape_this_src4(3), param_shape_this_src4(4), ...
                param_shape_this_dst4(1), param_shape_this_dst4(2), param_shape_this_dst4(3), param_shape_this_dst4(4), scl_factor_this);
            
            
            data_src = net_src.layers(layers_src{i}).params(p).get_data;
            if (~shape_strict) 
                data_src = reshape(data_src, size(net_dst.layers(layers_dst{i}).params(p).get_data)); 
            end
            if (scl_factor_this ~= 1) 
                data_src = data_src * scl_factor_this; 
            end
            net_dst.layers(layers_dst{i}).params(p).set_data(data_src);
            
            data_dst = net_dst.layers(layers_dst{i}).params(p).get_data;
            assert(all(data_src(:) == data_dst(:)));
            fprintf(' successful\n')
        end          
    catch exception
        fprintf('Exception message %s\n', getReport(exception));
    end
end

end