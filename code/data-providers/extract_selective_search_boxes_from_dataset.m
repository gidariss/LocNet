function all_bbox_proposals = extract_selective_search_boxes_from_dataset(image_db, ss_boxes_dst_file)
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

try 
    ld = load(ss_boxes_dst_file);
    all_bbox_proposals = ld.all_bbox_proposals;
catch
    chunk_size = 1000;
    num_imgs   = numel(image_db.image_paths);
    num_chunks = ceil(num_imgs/chunk_size);
    
    ss_boxes_dst_file_in_progress1 = regexprep(ss_boxes_dst_file, '.mat', '_in_progress.mat');
    ss_boxes_dst_file_in_progress2 = regexprep(ss_boxes_dst_file, '.mat', '_in_progress_prev.mat');
    
    try
        try
            ld = load(ss_boxes_dst_file_in_progress1);
            all_bbox_proposals = ld.all_bbox_proposals;
            first_chunk = ld.chunk + 1;
        catch
            ld = load(ss_boxes_dst_file_in_progress2);
            all_bbox_proposals = ld.all_bbox_proposals;
            first_chunk = ld.chunk + 1;            
        end
    catch exception
        fprintf('Exception message %s\n', getReport(exception));
        all_bbox_proposals = cell(num_imgs,1);
        first_chunk = 1;
    end
  
    total_num_elems = 0;
    total_time = 0;
    count = 0;
    for chunk = first_chunk:num_chunks
        start_idx = (chunk-1) * chunk_size + 1;
        stop_idx  = min(chunk * chunk_size, num_imgs);
        th = tic;
        all_bbox_proposals(start_idx:stop_idx) = extract_selective_search_prposlas(image_db.image_paths(start_idx:stop_idx));
        for i = start_idx:stop_idx
            count = count + 1;
            total_num_elems = total_num_elems + numel(all_bbox_proposals{i});
        end
        elapsed_time = toc(th);
        total_time = total_time + elapsed_time;
        est_rem_time = (total_time / count) * (num_imgs - stop_idx);
        est_num_bytes = (total_num_elems / count) * num_imgs * 4 / (1024*1024*1024);
        fprintf('Extract Selective Search boxes %s %d/%d: ET %.2fmin | ETA %.2fmin | EST. NUM BYTES %.2f giga\n', ...
            image_db.image_set_name, stop_idx, num_imgs, ...
            total_time/60, est_rem_time/60, est_num_bytes);
        
        if (exist(ss_boxes_dst_file_in_progress1,'file')>0)
            copyfile(ss_boxes_dst_file_in_progress1,ss_boxes_dst_file_in_progress2);
        end

        save(ss_boxes_dst_file_in_progress1, 'all_bbox_proposals', 'chunk', '-v7.3');
    end

    save(ss_boxes_dst_file, 'all_bbox_proposals', '-v7.3');
end
end

function all_box_proposals = extract_selective_search_prposlas(image_paths)
fast_mode = true;
num_imgs = length(image_paths);
all_box_proposals = cell(num_imgs,1);
parfor (i = 1:num_imgs)
%     th = tic;
    img = imread(image_paths{i});
    all_box_proposals{i} = selective_search_boxes(img, fast_mode);
    all_box_proposals{i} = single(all_box_proposals{i}(:,[2 1 4 3]));
%     fprintf(' image %d/%d: elapsed time %.2f\n', i, num_imgs, toc(th))
end
end
