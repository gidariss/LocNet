function blob = img_list_to_blob(imgs)
% it gets as input a list of images (in form of cell array) and it packs 
% them to a single tensor (matrix) with 4 dimensions: [Width x Height x 3 x NumImages]
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

max_shape = max(cell2mat(cellfun(@size, imgs(:), 'UniformOutput', false)), [], 1);
assert(all(cellfun(@(x) size(x, 3), imgs, 'UniformOutput', true) == 3));
num_images = length(imgs);
blob = zeros(max_shape(1), max_shape(2), 3, num_images, 'single');
for i = 1:length(imgs)
    blob(1:size(imgs{i}, 1), 1:size(imgs{i}, 2), :, i) = imgs{i}; 
end
end