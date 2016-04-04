function all_box_proposals = load_box_proposals( image_db, method )
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

base_directory = fullfile(pwd,'data/');
mkdir_if_missing(base_directory);

selective_search_path       = fullfile(base_directory, 'selective_search_data/');
edge_boxes_path             = fullfile(base_directory, 'edge_boxes_data/');
edge_boxes_mult_scale_path  = fullfile(base_directory, 'edge_boxes_mult_scale_data/');
dense_boxes_10k_path        = fullfile(base_directory, 'dense_boxes_10k_data/');

if ischar(method), method = {method}; end
assert(iscell(method));
num_methods = numel(method);
all_box_proposals_methods = cell(num_methods,1);

image_set_name = image_db.image_set_name;

for m = 1:num_methods
    switch method{m}
        case 'selective_search'
            mkdir_if_missing(selective_search_path);
            proposals_path = sprintf('%s/%s.mat', selective_search_path, image_set_name);
            all_box_proposals = extract_selective_search_boxes_from_dataset(...
                image_db, proposals_path);    
        case 'ground_truth'            
            all_box_proposals = image_db.all_bbox_gt;
            for i = 1:numel(all_box_proposals)
                all_box_proposals{i} = all_box_proposals{i}(:,1:4); 
            end   
        case 'dense_boxes_10k'
            num_boxes = 10000;
            mkdir_if_missing(dense_boxes_10k_path);
            proposals_path = sprintf('%s/%s.mat', dense_boxes_10k_path, image_set_name);
            all_box_proposals = extract_dense_boxes_from_dataset(image_db, proposals_path, num_boxes);                           
        case 'edge_boxes_mult_scales'
            mkdir_if_missing(edge_boxes_mult_scale_path);
            proposals_path = sprintf('%s/%s.mat', edge_boxes_mult_scale_path, image_set_name);
            all_box_proposals = extract_edge_boxes_from_dataset(image_db, proposals_path, true);            
        case 'edge_boxes'    
            mkdir_if_missing(edge_boxes_path);
            proposals_path = sprintf('%s/%s.mat', edge_boxes_path, image_set_name);
            all_box_proposals = extract_edge_boxes_from_dataset(image_db, proposals_path);               
        otherwise
            error('unsupported option')
    end
    all_box_proposals_methods{m} = all_box_proposals;
end
all_box_proposals = merge_bboxes(all_box_proposals_methods);
end

function all_box_proposals = merge_bboxes(all_box_proposals_methods)

num_methods  = length(all_box_proposals_methods);
num_imgs     = length(all_box_proposals_methods{1});


if num_methods == 1
    all_box_proposals = all_box_proposals_methods{1};
    return;
end
all_box_proposals   = cell(num_imgs, 1);


for i = 1:num_imgs
    aboxes_this_img_this = cell(num_methods, 1);
    for d = 1:num_methods
        aboxes_this_img_this{d} = single(all_box_proposals_methods{d}{i});
    end
    all_box_proposals{i} = cell2mat(aboxes_this_img_this);
end

end
