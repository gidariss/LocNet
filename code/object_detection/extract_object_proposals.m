function bbox_proposals = extract_object_proposals( img, conf )
% extract_object_proposals: given an image it extracts class agnostic
% bounding box proposals from it using one of the following algorithms:
% 1) Edge Box proposals (around 2k proposals) - conf.box_method = ''
% 2) Selective Search proposals (around 2k proposals)
% 3) 10k Dense proposals generated from a sliding window scheme.
%
% INPUT:
% 1) img: a H x W x 3 uint8 matrix that contains the image pixel values
% 2) conf: a struct that must contain the following field(s):
%    a) conf.box_method: string with the box proposals algorithm that
%       will be used in order to generate the set of class agnostic 
%       bounding box proposals. It can take the following values:
%       i)   'edge_boxes'      : Edge Box proposals
%       ii)  'selective_search': Selective Search proposals
%       iii) 'dense_boxes_10k' : 10k Dense proposals
%
% OUTPUT:
% 1) bbox_proposals is a NB X 4 array that contains the candidate boxes; 
% the i-th row contains cordinates [x0, y0, x1, y1] of the i-th candidate 
% box, where the (x0,y0) and (x1,y1) are coorindates of the top-left and 
% bottom-right corners correspondingly. NB is the number of box proposals.
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

switch conf.box_method
    case 'edge_boxes'
        bbox_proposals = extract_edge_boxes_from_image(img);
    case 'selective_search'
        bbox_proposals = extract_selective_search_from_image(img); 
    case 'dense_boxes_10k'
        num_boxes = 10000;
        bbox_proposals = extract_dense_sliding_window_proposals_from_image( img, num_boxes );
    otherwise 
        error('The box proposal type %s is not valid',conf.box_method)
end

% bbox_proposals is a NB X 4 array that contains the candidate boxes; the i-th
% row contains cordinates [x0, y0, x1, y1] of the i-th candidate box, where 
% the (x0,y0) and (x1,y1) are coorindates of the top-left and 
% bottom-right corners correspondingly. NB is the number of box proposals.

end

function bbox_proposals = extract_edge_boxes_from_image(img)
% bbox_proposals is a NB X 4 array contains the candidate boxes; the i-th
% row contains cordinates [x0, y0, x1, y1] of the i-th candidate box, where 
% the (x0,y0) and (x1,y1) are coorindates of the top-left and 
% bottom-right corners correspondingly.

edge_boxes_path = fullfile(pwd, 'external', 'edges');
model=load(fullfile(edge_boxes_path,'models/forest/modelBsds')); 
model=model.model;
model.opts.multiscale=0; model.opts.sharpen=2; model.opts.nThreads=4;

% set up opts for edgeBoxes
opts          = edgeBoxes;
opts.alpha    = .65;     % step size of sliding window search
opts.beta     = .70;     % nms threshold for object proposals
opts.minScore = .01;     % min score of boxes to detect
opts.maxBoxes = 2000;    % max number of boxes to detect
    
boxes = edgeBoxes(img,model,opts);

bbox_proposals = [boxes(:,1:2), boxes(:,1:2) + boxes(:,3:4)-1];
end

function bbox_proposals = extract_selective_search_from_image(img)
% bbox_proposals is a NB X 4 array contains the candidate boxes; the i-th
% row contains cordinates [x0, y0, x1, y1] of the i-th candidate box, where 
% the (x0,y0) and (x1,y1) are coorindates of the top-left and 
% bottom-right corners correspondingly.
fast_mode = true;
boxes = selective_search_boxes(img, fast_mode);
bbox_proposals = single(boxes(:,[2 1 4 3]));
end

function bbox_proposals = extract_dense_sliding_window_proposals_from_image(img, num_boxes)
% bbox_proposals is a NB X 4 array contains the candidate boxes; the i-th
% row contains cordinates [x0, y0, x1, y1] of the i-th candidate box, where 
% the (x0,y0) and (x1,y1) are coorindates of the top-left and 
% bottom-right corners correspondingly.
bbox_proposals = sample_bing_windows( img, num_boxes );
bbox_proposals = single(bbox_proposals);
end