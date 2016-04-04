function [ image_ids ] = getImageIdsFromImagePaths( image_paths )
num_imgs  = numel(image_paths);
image_ids = cell(num_imgs,1);
for img_idx = 1:num_imgs
    [img_dir, image_id, img_ext] = fileparts(image_paths{img_idx});
    image_ids{img_idx} = image_id;
end
end

