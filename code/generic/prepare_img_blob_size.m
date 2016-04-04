function im_scaled_size = prepare_img_blob_size(im_size, target_size, max_size)
% 
% This file is part of the code that implements the following paper:
% Title      : "LocNet: Improving Localization Accuracy for Object Detection"
% Authors    : Spyros Gidaris, Nikos Komodakis
% Institution: Universite Paris Est, Ecole des Ponts ParisTech
% ArXiv link : http://arxiv.org/abs/1511.07763
% code       : https://github.com/gidariss/LocNet
%
% Part of the code in this file comes from the Faster R-CNN code and the matlab
% re-implementation of Fast-RCNN (https://github.com/ShaoqingRen/faster_rcnn)
% 
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2016 Spyros Gidaris
% 
% Title     : "LocNet: Improving Localization Accuracy for Object Detection"
% ArXiv link: http://arxiv.org/abs/1511.07763
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------

im_height   = im_size(1);
im_width    = im_size(2);
im_min_size = min([im_height, im_width]);
im_max_size = max([im_height, im_width]);
im_scale    = target_size / im_min_size;

if round(im_max_size * im_scale) > max_size
    im_scale = max_size / im_max_size;
end

im_resized_width  = round(im_scale * im_width);
im_resized_height = round(im_scale * im_height);
im_scaled_size    = [im_resized_height, im_resized_width];
end