function bbox_pred = decode_reg_vals_to_bbox_targets(bbox_init, reg_values, class_indices)
%
% The code in this file comes from the RCNN code: 
% https://github.com/rbgirshick/rcnn
%
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Ross Girshick
% 
% This file is part of the R-CNN code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------


num_bboxes  = size(reg_values,1);
num_targets = size(reg_values,2);
num_classes = size(reg_values,3);
assert(num_targets == 4);
assert(numel(class_indices) == num_bboxes);

bbox_pred = zeros(num_bboxes,4, 'single');
for c = 1:num_classes
    this_cls_mask = c == class_indices;
    bbox_pred(this_cls_mask,1:4) = decode_reg_values(...
        bbox_init(this_cls_mask,1:4), reg_values(this_cls_mask,1:4,c));
end
bbox_pred = [bbox_pred, single(class_indices)];
end

function bbox_pred = decode_reg_values(bbox_init, reg_values)

dst_ctr_x = reg_values(:,1);
dst_ctr_y = reg_values(:,2);
dst_scl_x = reg_values(:,3);
dst_scl_y = reg_values(:,4);

src_w     = bbox_init(:,3) - bbox_init(:,1) + 1;
src_h     = bbox_init(:,4) - bbox_init(:,2) + 1;
src_ctr_x = bbox_init(:,1) + 0.5*(src_w-1);
src_ctr_y = bbox_init(:,2) + 0.5*(src_h-1);

pred_ctr_x = (dst_ctr_x .* src_w) + src_ctr_x;
pred_ctr_y = (dst_ctr_y .* src_h) + src_ctr_y;
pred_w     = exp(dst_scl_x) .* src_w;
pred_h     = exp(dst_scl_y) .* src_h;

bbox_pred = [pred_ctr_x - 0.5*(pred_w-1), pred_ctr_y - 0.5*(pred_h-1), ...
             pred_ctr_x + 0.5*(pred_w-1), pred_ctr_y + 0.5*(pred_h-1)];
end