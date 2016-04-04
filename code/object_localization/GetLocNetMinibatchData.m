function [loc_blobs, bbox_targets_loc, bbox_inits_loc, image_paths_all] = ...
        GetLocNetMinibatchData(conf, model, image_roidb, sampled_regions_loc, im_scales, image_sizes)
    
assert(length(image_roidb) == length(sampled_regions_loc));
num_images  = length(image_roidb);

bbox_inits_loc    = zeros([0, 4], 'single'); % coordinates of the training cadidate bounding boxes
rois_blob_loc     = zeros([0, 9], 'single'); % coordinates of the training search regions
bbox_targets_loc  = zeros([0, 5], 'single'); % coordinates of the training target bounding boxes

loc_prob_vectors  = cell([1,1,1,num_images]);

gt_labels_all  = zeros(0,1,'single');

for i = 1:num_images
    bboxes_this = sampled_regions_loc{i}(:,1:4); % candidate bounding boxes
    gt_inds     = sampled_regions_loc{i}(:,5); % index of the target ground truth bounding boxes
    gt_labels   = sampled_regions_loc{i}(:,6); % category id of the target ground truth bounding boxes 
    bbox_targets_this = single([image_roidb(i).bbox_gt(gt_inds,1:4), gt_labels]); % target ground truth bounding boxes
    

    if strcmp(conf.loc_type,  'bboxreg')
        loc_prob_vectors{i} = encode_bbox_targets_to_reg_vals(...
            bboxes_this, bbox_targets_this, conf.num_classes);
    else % inout, borders, or combined
        loc_prob_vectors{i} = encode_bbox_target_to_loc_probs(...
            bboxes_this, bbox_targets_this, conf);
    end
    
    % map the candidate bounding boxes to image regions in the scaled image 
    rois_blob_this_image = map_boxes_to_regions_mult_scales(model.pooler_loc, ...
        bboxes_this, image_sizes(i,:), im_scales(i));
    rois_blob_this_image(:,1) = i;
    
    rois_blob_loc    = [rois_blob_loc;    rois_blob_this_image]; 
    bbox_inits_loc   = [bbox_inits_loc;   bboxes_this];
    bbox_targets_loc = [bbox_targets_loc; bbox_targets_this];
    gt_labels_all    = [gt_labels_all;    single(gt_labels)];
end

image_paths       = {image_roidb.image_path}';  
image_paths_all   = image_paths(rois_blob_loc(:,1));

% permute data into caffe c++ memory, thus [num, channels, height, width]
rois_blob_loc     = rois_blob_loc - 1; % to c's index (start from 0)
rois_blob_loc     = single(permute(rois_blob_loc, [2, 1]));


switch conf.loc_type
    case 'bboxreg'
        loc_prob_vectors = cell2mat(loc_prob_vectors(:));
        
        % set to one only the weights that correspond to the target vectors
        % of the same category id as the candidate bounding boxes
        loc_prob_vector_weights = create_target_weights_BBoxReg(size(loc_prob_vectors),gt_labels_all);
        
        % reshape from [NumBoxes x 4 x num_classes] -> [NumBoxes x (4*num_classes)]
        loc_prob_vectors = reshape(loc_prob_vectors, ...
            [size(loc_prob_vectors,1), 4*conf.num_classes]);
        % permute from [NumBoxes x (4*num_classes)] -> [(4*num_classes) x NumBoxes]
        loc_prob_vectors = single(permute(loc_prob_vectors, [2, 1]));
        
        % reshape from [NumBoxes x 4 x num_classes] -> [NumBoxes x (4*num_classes)]
        loc_prob_vector_weights = reshape(loc_prob_vector_weights, ...
            [size(loc_prob_vector_weights,1), 4*conf.num_classes]);
        % permute from [NumBoxes x (4*num_classes)] -> [(4*num_classes) x NumBoxes]
        loc_prob_vector_weights = single(permute(loc_prob_vector_weights, [2, 1]));
    case {'inout','borders','combined'}
        loc_prob_vectors = cell2mat(loc_prob_vectors);
        
        % set to one only the weights that correspond to the target vectors
        % of the same category id as the candidate bounding boxes
        loc_prob_vector_weights = create_target_weights_LocNet(size(loc_prob_vectors), gt_labels_all);     
        
        if strcmp(conf.loc_type,'borders')
            resolution = size(loc_prob_vectors,1);
            pos_weight = resolution/2;
            neg_weight = 0.5*resolution/(resolution-1);
            % balance the weights that correspond to the target probability
            % vectors for the borders of an object
            loc_prob_vector_weights = loc_prob_vector_weights.*...
                ((loc_prob_vectors==1)*pos_weight+... % increase the weight of the borders rows/columns (positives)
                 (loc_prob_vectors==0)*neg_weight);   % decrease the weight of the non-borders rows/columns (negatives)                
        elseif strcmp(conf.loc_type,'combined')
            resolution = size(loc_prob_vectors,1);
            pos_weight = resolution/2;
            neg_weight = 0.5*resolution/(resolution-1);

            % balance the weights that correspond to the target probability
            % vectors for the borders of an object
            loc_prob_vector_weights(:,[2,3,5,6],:,:) = loc_prob_vector_weights(:,[2,3,5,6],:,:).*...
                ((loc_prob_vectors(:,[2,3,5,6],:,:)==1)*pos_weight+... % increase the weight of the borders rows/columns (positives)
                 (loc_prob_vectors(:,[2,3,5,6],:,:)==0)*neg_weight);   % decrease the weight of the non-borders rows/columns (negatives)             
        end
    otherwise 
        error('Invalid localization type %s',conf.loc_type)
end

assert(~isempty(loc_prob_vector_weights));
assert(~isempty(loc_prob_vectors));

assert(~isempty(rois_blob_loc));
assert(~isempty(bbox_inits_loc));
assert(~isempty(bbox_targets_loc));

bbox_inits_loc = [bbox_inits_loc, gt_labels_all(:)];

% cell array with the caffe blobs needed for training the localization 
% module of the network
loc_blobs = [{rois_blob_loc},{loc_prob_vectors},{loc_prob_vector_weights}];
end

function target_weights = create_target_weights_LocNet(target_size, category_id)
% set to one only the target weights that correspond to the category id of
% each candidate bounding box
target_weights = zeros(target_size, 'single'); 
for i = 1:length(category_id)
    target_weights(:,:,category_id(i),i) = 1; 
end
end

function target_weights = create_target_weights_BBoxReg(target_size, category_id)
% set to one only the target weights that correspond to the category id of
% each candidate bounding box
target_weights = zeros(target_size, 'single'); 
for i = 1:length(category_id)
    target_weights(i,:,category_id(i)) = 1; 
end
end