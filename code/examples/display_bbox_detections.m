function display_bbox_detections( img, bbox_detections, score_thresh, category_names )

all_dets = [];
num_categories = length(bbox_detections);
for i = 1:num_categories
    bbox_detections_this_category  = bbox_detections{i};
    detection_scores_this_category = bbox_detections_this_category(:,5);
    is_above_the_thresh = detection_scores_this_category >= score_thresh(i);
    bbox_detections_this_category = bbox_detections_this_category(is_above_the_thresh,:);
    bbox_detections_this_category = [i * ones(size(bbox_detections_this_category, 1), 1), bbox_detections_this_category];
    all_dets = cat(1, all_dets, bbox_detections_this_category);
end

fprintf('Visualize the bounding box detections:\n')
[~, ord] = sort(all_dets(:,end), 'descend');
for i = 1:length(ord)
  score_this = all_dets(ord(i), end);
  category_name_this = category_names{all_dets(ord(i), 1)};
  showboxes(img, all_dets(ord(i), 2:5));
  title(sprintf('det #%d: %s score = %.3f', i, category_name_this, score_this));
  fprintf('det #%d: %s score = %.3f. press any key to continue\n', ...
      i, category_name_this, score_this);
  drawnow;
  pause;
end


fprintf('No more detections\n');

end

