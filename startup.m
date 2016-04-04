function startup()
% set to edge_boxes_path the path where the edge boxes code 
% (https://github.com/pdollar/edges) is installed 
edge_boxes_path = fullfile(pwd,'external', 'edges'); 
% set to pdollar_toolbox_path the path where the Piotr's Matlab Toolbox 
% (http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html) is installed   
pdollar_toolbox_path = fullfile(pwd,'external', 'pdollar-toolbox'); 
% set to selective_search_boxes_path the path where the Selective Search code 
% (http://huppelen.nl/publications/SelectiveSearchCodeIJCV.zip) is installed 
selective_search_boxes_path = fullfile(pwd,'external', 'selective_search'); 

curdir = fileparts(mfilename('fullpath'));

addpath(genpath(fullfile(curdir,  'code')));
mkdir_if_missing(fullfile(curdir, 'data'));
mkdir_if_missing(fullfile(curdir, 'bin'));
mkdir_if_missing(fullfile(curdir, 'cached_regions'));
mkdir_if_missing(fullfile(curdir, 'models-exps'));
addpath(fullfile(curdir, 'bin'));

if exist(edge_boxes_path,'dir') > 0
    addpath(edge_boxes_path) 
else
    warning('The Edge Boxes installation directory "%s" is not valid.', edge_boxes_path)
end
edge_boxes_link_path = fullfile(curdir, 'external', 'edges');
if exist(edge_boxes_link_path, 'dir') == 0
    warning('A link to the edge box installation directory is missing from external/edges; See README.md');
end
if exist(pdollar_toolbox_path,'dir') > 0 
    addpath(genpath(pdollar_toolbox_path))
else
    warning('The installation directory "%s" to Piotrs image processing toolbox (http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html) is not valid.', pdollar_toolbox_path)
end
if exist(selective_search_boxes_path,'dir') > 0 
    addpath(genpath(selective_search_boxes_path))
else
    warning('The installation directory "%s" to the Selective Serach code (http://huppelen.nl/publications/SelectiveSearchCodeIJCV.zip) is not valid.', selective_search_boxes_path)
end
caffe_path = fullfile(curdir, 'external', 'caffe_LocNet', 'matlab');
if exist(caffe_path, 'dir') == 0
    error('matcaffe is missing from external/caffe_LocNet/matlab; See README.md');
end
addpath(genpath(caffe_path));

end

