function image_db = load_image_dataset(varargin)
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

% ************************** OPTIONS *************************************
ip = inputParser;
ip.addParamValue('dataset', 'pascal');
ip.addParamValue('image_set', {})
ip.addParamValue('voc_year',  {})
ip.addParamValue('proposals_method', {});
ip.addParamValue('use_flips', false, @islogical);

ip.parse(varargin{:});
opts = ip.Results;

voc_year         = opts.voc_year;
image_set        = opts.image_set;
proposals_method = opts.proposals_method;
use_flips        = opts.use_flips;

voc_path  = [pwd, '/datasets/VOC%s/'];
coco_path = [pwd, '/datasets/MSCOCO/'];


if ~iscell(proposals_method), proposals_method = {proposals_method}; end
if ~iscell(voc_year), voc_year = {voc_year}; end
if ~iscell(image_set), image_set = {image_set}; end


assert(ischar(voc_year{1}));

num_sets = length(image_set);
assert(length(image_set) == length(voc_year));
image_db.all_regions = {};
for i = 1:num_sets
    % add the paths to the images and the grount truth bounding boxes in
    % the image_db structure
    switch opts.dataset
        case 'pascal'
            image_db_this = load_pascal_dataset(...
                voc_year{i}, image_set{i}, voc_path);
        case 'mscoco'
            image_db_this = load_mscoco_dataset(...
                voc_year{i}, image_set{i}, coco_path );            
    end
    % add the bounding box proposals in the image_db structure
    if ~isempty(proposals_method)
        assert(ischar(proposals_method{1}));
        image_db.all_regions = [image_db.all_regions(:); ...
            load_box_proposals(image_db_this, proposals_method)];
    end
    image_db_all(i) = image_db_this;
end
image_db_all = image_db_all(:);
image_db.all_bbox_gt = vertcat(image_db_all.all_bbox_gt);
image_db.image_paths = vertcat(image_db_all.image_paths);
image_db.image_sizes = single(vertcat(image_db_all.image_sizes));

proposals_suffix = '';
if ~isempty(proposals_method)
    for p = 1:length(proposals_method)
        proposals_suffix = [proposals_suffix, proposals_method{p}, '_'];
    end
    proposals_suffix = proposals_suffix(1:end-1);
end

switch opts.dataset
    case 'pascal', image_set_name_all = 'voc_';
    case 'mscoco', image_set_name_all = 'coco_';           
end


for i = 1:num_sets
    image_set_name_all = [image_set_name_all, voc_year{i}, '_' image_set{i},'_'];
end
image_set_name_all = image_set_name_all(1:end-1);

image_db.image_set_name = image_set_name_all;
image_db.proposals_suffix = proposals_suffix;

% add flipped versions of the images
if use_flips, image_db = add_flip_data(image_db); end

switch opts.dataset
    case 'pascal'
        voc_path_year = sprintf(voc_path, '2007');
        VOCopts  = initVOCOpts(voc_path_year,'2007');
        classes = VOCopts.classes;
    case 'mscoco'         
        save_file_categories = fullfile(coco_path,'annotations','categoties.mat');
        ld = load(save_file_categories,'category_names','cat_ids','catIdsToIndex');
        classes = ld.category_names;
end

image_db.classes = classes;

end

function image_db = load_pascal_dataset(voc_year, image_set, voc_path)
image_db = struct;
voc_path_dataset = sprintf(voc_path, voc_year);
assert(exist(voc_path_dataset,'dir')>0);
voc_path_devkit = fullfile(voc_path_dataset,'VOCdevkit');
assert(exist(voc_path_devkit,'dir')>0);

[image_db.image_paths, image_db.image_set_name] = get_image_paths_from_voc( voc_path_devkit, image_set, voc_year );
image_db.all_bbox_gt = get_grount_truth_bboxes_from_voc(voc_path_devkit, image_set, voc_year, true, voc_path_dataset);
image_db.image_sizes = get_img_size(image_db.image_paths);
end

function image_db = load_mscoco_dataset(coco_year, image_set, data_dir )
image_db = struct;

suffix = sprintf('%s%s',image_set,coco_year);
image_db.image_set_name = sprintf('coco_%s_%s',coco_year,image_set);
save_file_bbox_gt = fullfile(data_dir,'annotations',sprintf('all_bbox_gt_%s.mat',suffix));

save_file_image_names = fullfile(data_dir,'images',sprintf('all_image_names_%s.mat',suffix));
ld = load(save_file_image_names, 'image_names', 'image_sizes');
image_names          = ld.image_names;
image_db.image_paths = strcat(sprintf('%s/images/%s/', data_dir, suffix), image_names);
image_db.image_sizes = ld.image_sizes;

if (exist(save_file_bbox_gt,'file') > 0)
    ld = load(save_file_bbox_gt, 'all_bbox_gt');
    image_db.all_bbox_gt = ld.all_bbox_gt;
else
    image_db.all_bbox_gt = cell(length(image_db.image_paths),1);
    for i = 1:length(image_db.all_bbox_gt)
        image_db.all_bbox_gt{i} = zeros(0,6,'single');
    end
end

end

function image_sizes = get_img_size(image_paths)
num_imgs    = numel(image_paths);
image_sizes = zeros(num_imgs,2);

for img_idx = 1:num_imgs
    im_info                = imfinfo(image_paths{img_idx});
    image_sizes(img_idx,1) = im_info.Height;
    image_sizes(img_idx,2) = im_info.Width;
end
end

function image_db = add_flip_data(image_db)
num_imgs          = numel(image_db.image_paths);
image_paths_new   = cell(num_imgs,1);

there_are_regions = isfield(image_db, 'all_regions') && ~isempty(image_db.all_regions);
there_are_bbox_gt  = isfield(image_db, 'all_bbox_gt')  && ~isempty(image_db.all_bbox_gt);
there_are_feature_paths = isfield(image_db, 'feature_paths')  && ~isempty(image_db.feature_paths);

if there_are_regions,   all_regions_new = image_db.all_regions; end
if there_are_bbox_gt,   all_bbox_gt_new = image_db.all_bbox_gt; end
if there_are_feature_paths, feature_paths_new = cell(num_imgs,1); end

for img_idx = 1:num_imgs
    [img_dir, img_name, img_ext]  = fileparts(image_db.image_paths{img_idx});
    image_paths_new{img_idx} = [img_dir, filesep, img_name, '_flip', img_ext];
    
    if there_are_feature_paths
        [feat_dir, feat_name, feat_ext]  = fileparts(image_db.feature_paths{img_idx});
        feature_paths_new{img_idx} = [feat_dir, filesep, feat_name, '_flip', feat_ext];    
    end
    
    img_weight = image_db.image_sizes(img_idx,2);    
    if there_are_regions
        all_regions_new{img_idx}(:,[1,3]) = img_weight + 1 - all_regions_new{img_idx}(:,[3,1]);
    end
    if there_are_bbox_gt
        all_bbox_gt_new{img_idx}(:,[1,3]) = img_weight + 1 - all_bbox_gt_new{img_idx}(:,[3,1]);
    end    
end

image_db.image_paths = cat(1, image_db.image_paths, image_paths_new);
image_db.image_sizes = cat(1, image_db.image_sizes,  image_db.image_sizes);


if there_are_bbox_gt, image_db.all_bbox_gt = cat(1, image_db.all_bbox_gt, all_bbox_gt_new); end
if there_are_regions, image_db.all_regions = cat(1, image_db.all_regions, all_regions_new); end
if there_are_feature_paths, image_db.feature_paths = cat(1,image_db.feature_paths, feature_paths_new); end
end