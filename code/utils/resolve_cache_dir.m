function cache_dir = resolve_cache_dir(model, model_dir, opts)
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

cache_dir_base = fullfile(model_dir, 'cache_dir');
mkdir_if_missing(cache_dir_base);
if ~isempty(opts.cache_dir)
    assert(isempty(opts.cache_dir_name));
    cache_dir = opts.cache_dir;
elseif ~isempty(opts.cache_dir_name)
    assert(isempty(opts.cache_dir));
    cache_dir = fullfile(cache_dir_base, opts.cache_dir_name);
else
    if opts.use_detection_svms
        cache_dir = fullfile(cache_dir_base, 'detection_svms');
    else
        cache_dir = fullfile(cache_dir_base, 'softmax'); 
    end
end
mkdir_if_missing(cache_dir);
end
