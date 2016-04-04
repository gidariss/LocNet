function scores = recognize_bboxes_of_image(model, image, bboxes)
% recognize_bboxes_of_image given a recongiotion model, an image and a set
% of bounding boxes it returns the confidence scores of each bounding box 
% w.r.t. each of the C categories of the recognition model. Those 
% confidence scores represent the likelihood of each bounding box to 
% tightly enclose an object of each of the C cateogies.
%
% INPUTS:
% 1) model:  (type struct) the bounding box recognition model
% 2) image:  a [Height x Width x 3] uint8 matrix with the image 
% 3) bboxes: a N x 4 array with the bounding box coordinates; each row is 
% the oordinates of a bounding box in the form of [x0,y0,x1,y1] where 
% (x0,y0) is tot-left corner and (x1,y1) is the bottom-right corner. N is 
% the number of bounding boxes.
%
% OUTPUT:
% scores: N x C array with the confidence scores of each bounding box
% for each of the C categories. N is the number of bounding boxes.
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


num_classes = length(model.classes);
if isempty(bboxes)
    scores = zeros(0,num_classes,'single');
    return;
end

% apply on the candidate bounding boxes and on the image the region-based 
% CNN network.
[outputs, output_blob_names] = run_region_based_net_on_img(model, image, bboxes);
% get the output blob that corresponds to the confidence scores of the
% bounding boxes
idx = find(strcmp(output_blob_names,model.score_out_blob));
assert(numel(idx) == 1);
scores = outputs{idx}';

% in case that in the scores array there is an exra column than the number 
% of categories, then the first column represents the confidence score of 
% each bounding box to be on background and it is removed before the score 
% array is returned.
if size(scores,2) == (num_classes + 1), scores = scores(:,2:end); end
end 