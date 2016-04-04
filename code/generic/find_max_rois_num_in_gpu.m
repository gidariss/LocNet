function max_rois_num_in_gpu = find_max_rois_num_in_gpu(model)
% find_max_rois_num_in_gpu: given a region based CNN network (param model) 
% it returns the maximum number of regions/bboxes that can be fed in one go 
% in the caffe network such that everything will fit in the GPU memory.
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


warning('the function find_max_rois_num_in_gpu(model) might be broken');

% create the biggest possible image to be fed in the region based CNN
max_scale_size = max(model.scales); % height
model.scales = max_scale_size;
max_size = model.max_size; % width
image = 255 * ones([max_scale_size, max_size, 3],'uint8'); 

rois_start = 300;
rois_step  = 300;
max_rois_num_in_image = rois_step * 15;
for max_rois_num_in_gpu = rois_start:rois_step:max_rois_num_in_image
    % create random max_rois_num_in_gpu number of regions/bboxes
    bboxes = zeros([max_rois_num_in_gpu, 4],'single'); 
    bboxes(:,1) = randi([1, round(max_size/2)-1],                   [max_rois_num_in_gpu,1]);
    bboxes(:,2) = randi([1, round(max_scale_size/2)-1],             [max_rois_num_in_gpu,1]);
    bboxes(:,3) = randi([round(max_size/2)+1,       max_size],      [max_rois_num_in_gpu,1]);
    bboxes(:,4) = randi([round(max_scale_size/2)+1, max_scale_size],[max_rois_num_in_gpu,1]);
    
    % run the region based network
    model.max_rois_num_in_gpu = max_rois_num_in_gpu;
    run_region_based_net_on_img(model, image, bboxes);
    
    % check if there is enough GPU memory for more regions/bboxes
    gpuInfo = gpuDevice(); 
    if gpuInfo.FreeMemory < 1.5 * 10^9  % 1.5 GBytes for safety
        break;
    end
end

end