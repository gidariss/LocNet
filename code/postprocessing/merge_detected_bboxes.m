function aboxes_out = merge_detected_bboxes(aboxes)
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

num_methods  = length(aboxes);
num_classes  = length(aboxes{1});
num_imgs     = length(aboxes{1}{1});

aboxes_out   = cell(num_classes, 1);

for j = 1:num_classes, aboxes_out{j} = cell(num_imgs,1); end

for i = 1:num_imgs
    for j = 1:num_classes
        aboxes_this_img_this_cls = cell(num_methods, 1);
        for d = 1:num_methods
            aboxes_this_img_this_cls{d} = aboxes{d}{j}{i};
        end
        aboxes_out{j}{i} = cell2mat(aboxes_this_img_this_cls);
    end
end

aboxes_out = {aboxes_out};
end
