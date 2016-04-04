function model_bbox_loc = load_bbox_loc_model_on_caffe(model_bbox_loc, full_model_loc_dir)
curr_dir = pwd;
cd(full_model_loc_dir);
model_phase = 'test';
model_bbox_loc.net = caffe_load_model( model_bbox_loc.net_def_file, ...
    model_bbox_loc.net_weights_file, model_phase);
cd(curr_dir);
end