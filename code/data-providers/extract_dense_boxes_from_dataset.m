function all_bbox_proposals = extract_dense_boxes_from_dataset(image_db, dense_boxes_dst_file, num_boxes)


if ~exist('only_small','var')
only_small = false;
end

try 
    ld = load(dense_boxes_dst_file);
    all_bbox_proposals = ld.all_bbox_proposals;
catch

    chunk_size = 1000;
    num_imgs   = numel(image_db.image_paths);
    all_bbox_proposals = cell(num_imgs,1);
    
    for i = 1:num_imgs
        img = imread(image_db.image_paths{i});
        [ all_bbox_proposals{i}, scores ] = sample_bing_windows( img, num_boxes );
        all_bbox_proposals{i} = single(all_bbox_proposals{i});
        tic_toc_print('dense boxes %d / % d (%.2f)\n', i, num_imgs, 100*i/num_imgs);
    end
    
    save(dense_boxes_dst_file, 'all_bbox_proposals', '-v7.3');
end
end
