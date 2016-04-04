function vgg_region_params = vgg_region_config()
% spp_params = spp_config()
%   spp parameters for special model
%
% AUTORIGHTS
% ---------------------------------------------------------
% Copyright (c) 2014, Shaoqing Ren
% 
% This file is part of the SPP code and is available 
% under the terms of the Simplified BSD License provided in 
% LICENSE. Please retain this notice and LICENSE if you use 
% this file (or any portion of it) in your project.
% ---------------------------------------------------------

vgg_region_params.offset0 = 0; 			% the left-top pixel on last convolution map --- offset0 pixel on input image
vgg_region_params.offset = 6; 			% shrinkage rectangle region on raw image
vgg_region_params.step_standard = 16;		% total stride
vgg_region_params.spm_divs = [7];			% spm parameters
vgg_region_params.sz_conv_standard  = 14;		% last convolution size
vgg_region_params.standard_img_size = 224;		% input image size
end
