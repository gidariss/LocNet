function [ all_bboxes_gt ] = get_grount_truth_bboxes_from_voc( voc_path, image_set, voc_year, with_hard_samples, cache_dir )
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

if ~exist('cache_dir','var')
    cache_dir = '.';
end

VOCopts            = initVOCOpts( voc_path, voc_year );
fullimage_set_name = ['voc_', voc_year, '_' image_set];
addpath([voc_path, filesep, 'VOCcode']);
cache_file = [cache_dir, filesep, 'gt_bbox_', fullimage_set_name];
if with_hard_samples, cache_file = [cache_file, '_with_hard_samples']; end
cache_file = [cache_file, '.mat'];

if exist(cache_file,'file')
    all_bboxes_gt = loadGroundTruthBBoxes(cache_file);
else
    class_to_id     = containers.Map(VOCopts.classes, 1:length(VOCopts.classes));
    image_ids       = textread(sprintf(VOCopts.imgsetpath, image_set), '%s');
    num_imgs        = length(image_ids);
    all_bboxes_gt   = cell(num_imgs, 1);
    for img_idx = 1:num_imgs
        if mod(img_idx,500) == 0, fprintf('Load gt bboxes::%s %d/%d\n', fullimage_set_name, img_idx, num_imgs); end
        all_bboxes_gt{img_idx} = getGroundTruthBBoxes(sprintf(VOCopts.annopath, image_ids{img_idx}), class_to_id, with_hard_samples);
    end
    saveGroundTruthBBoxes(cache_file, all_bboxes_gt);
end

end

function gt_bboxes = getGroundTruthBBoxes(filename, class_to_id, with_hard_samples)
try
    voc_rec       = PASreadrecord(filename);
    valid_objects = 1:length(voc_rec.objects(:));
    is_difficult  = cat(1, voc_rec.objects(:).difficult);
   if ~with_hard_samples
       valid_objects = valid_objects(~is_difficult);
       is_difficult  = is_difficult(~is_difficult);
   end
   
   gt_bboxes = single(cat(1, voc_rec.objects(valid_objects).bbox));
   gt_class_idx = class_to_id.values({voc_rec.objects(valid_objects).class});
   gt_class_idx = single(cat(1, gt_class_idx{:}));
   gt_bboxes    = single([gt_bboxes, gt_class_idx, is_difficult(:)]);
catch
    gt_bboxes = zeros(0, 6, 'single');
end
        
end

function all_bboxes_gt = loadGroundTruthBBoxes(filename)
load(filename, 'all_bboxes_gt');
end

function saveGroundTruthBBoxes(filename, all_bboxes_gt)
save(filename, 'all_bboxes_gt', '-v7.3');
end

