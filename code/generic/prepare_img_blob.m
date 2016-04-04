function [img, scale, im_scaled_size] = prepare_img_blob(img, scale, mean_pix, max_size)
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

if numel(mean_pix) == 1, mean_pix = [mean_pix, mean_pix, mean_pix]; end
assert(numel(mean_pix)==3);

im_height = size(img, 1);
im_width = size(img, 2);
% get the image size of the scaled image
im_scaled_size = prepare_img_blob_size([im_height, im_width], scale, max_size);
scale = min(im_scaled_size);

% scale image
if (scale <= 224)
    img = imresize(img, [im_scaled_size(1), im_scaled_size(2)], 'bilinear');
else
    img = imresize(img, [im_scaled_size(1), im_scaled_size(2)], 'bilinear', 'antialiasing', false);
end

img = single(img); % transform to single precision 
img = img(:,:,[3 2 1]); % transform from RGB -> BGR (necessary for Caffe)
% subtruct the mean pixel from the image
img = bsxfun(@minus, img, reshape(mean_pix, [1, 1, 3]));
img = permute(img, [2 1 3]); % permute from [Height x Width x 3] to [Width x Height x 3] (necessary for Caffe)
end