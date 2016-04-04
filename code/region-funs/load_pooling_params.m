function pool_params = load_pooling_params(pool_params_def, varargin)
% load_pooling_params(pool_params_def,...): it initializes the struct that
% contains the adaptive region max pooling parameters and the parameters
% related to the region type
%
% INPUTS:
% 1) pool_params_def: string with path to the configuration file that 
% contains the adaptive region max pooling parameters
% The rest input arguments are given in the form of Name,Value pair
% arguments and are related to the region type:
% 'scale_inner': scalar value with the scaling factor of the inner rectangle 
% of the region. In case this value is 0 then actually no inner rectangle 
% is being used
% 'scale_outer': scalar value with the scaling factor of the outer rectangle 
% of the region. 
% 'half_bbox': intiger value in the range [1,2,3,4]. If this parameter is set
% to 1, 2, 3, or 4 then each bounding box will be reshaped to its left, 
% right, top, or bottom half part correspondingly. This action is performed
% prior to scaling the box according to the scale_inner and scale_outer 
% params. If this parameter is missing or if it is empty then the action of 
% taking the half part of bounding box is NOT performed.
% 
% OUTPUT:
% 1) pool_params: struct that contains the adaptive region max pooling 
% parameters and the parameters related to the region type
% 
% This file is part of the code that implements the following ICCV2015 accepted paper:
% title: "Object detection via a multi-region & semantic segmentation-aware CNN model"
% authors: Spyros Gidaris, Nikos Komodakis
% institution: Universite Paris Est, Ecole des Ponts ParisTech
% Technical report: http://arxiv.org/abs/1505.01749
% code: https://github.com/gidariss/mrcnn-object-detection
% 
% Part of the code in this file comes from the SPP-Net code: 
% https://github.com/ShaoqingRen/SPP_net
% 
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2015 Spyros Gidaris
% 
% "Object detection via a multi-region & semantic segmentation-aware CNN model"
% Technical report: http://arxiv.org/abs/1505.01749
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------
% Copyright (c) 2014, Shaoqing Ren
% 
% This file is part of the SPP code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% --------------------------------------------------------- 

ip = inputParser;
ip.addParamValue('scale_inner',   [], @isnumeric);
ip.addParamValue('scale_outer',   [], @isnumeric);
ip.addParamValue('half_bbox',     [], @isnumeric);

ip.parse(varargin{:});
opts = ip.Results;

%% Read the adaptive region max pooling parameters from the configuration file
[~, ~, ext] = fileparts(pool_params_def);
if isempty(ext), pool_params_def = [pool_params_def, '.m']; end
assert(exist(pool_params_def, 'file') ~= 0);

cur_dir = pwd; % change folder to avoid too long path for eval()
[pool_def_dir, pool_def_file] = fileparts(pool_params_def);

cd(pool_def_dir);
pool_params = eval(pool_def_file);
cd(cur_dir);

%% Set the parameters related to the region type
pool_params.scale_inner = opts.scale_inner;
pool_params.scale_outer = opts.scale_outer;
pool_params.half_bbox   = opts.half_bbox;
end
