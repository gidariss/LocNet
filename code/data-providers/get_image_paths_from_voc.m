function [ image_paths, image_set_name ] = get_image_paths_from_voc( voc_path, image_set, voc_year )
% 
% This file is part of the code that implements the following paper:
% Title      : "LocNet: Improving Localization Accuracy for Object Detection"
% Authors    : Spyros Gidaris, Nikos Komodakis
% Institution: Universite Paris Est, Ecole des Ponts ParisTech
% ArXiv link : http://arxiv.org/abs/1511.07763
% code       : https://github.com/gidariss/LocNet
%
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2016 Spyros Gidaris
% 
% Title     : "LocNet: Improving Localization Accuracy for Object Detection"
% ArXiv link: http://arxiv.org/abs/1511.07763
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------

VOCopts         = initVOCOpts( voc_path, voc_year );
VOCopts.testset = image_set;
image_set_name  = ['voc_', voc_year, '_' image_set];

image_ext       = '.jpg';
image_dir       = fileparts(VOCopts.imgpath);
image_ids       = textread(sprintf(VOCopts.imgsetpath, image_set), '%s');
image_paths     = strcat([image_dir, filesep], image_ids, image_ext);
end


