function [ candidates, scores ] = sample_bing_windows( im, num_samples)
%SAMPLE_BING_WINDOWS Will generate equaly distributed windows in space,
%following Bing sizes
%   Bing uses 29 specific sizes, this method spread this sizes homogenously
%   inside the image
%
% This file comes from the code that implements the paper: "How good are 
% detection proposals, really?" of Hosang et al.. Link to the code:
% https://github.com/hosang/detection-proposals

scores = [];
im_wh = [size(im, 2), size(im, 1)];

original_bing_window_sizes = [[512 512 ]; [256, 512 ]; [128, 512 ]; [64, 512 ]; ...
    [512, 256 ]; [256, 256 ]; [128, 256 ]; [64, 256 ]; [32, 256 ]; ...
    [512, 128 ]; [256, 128 ]; [128, 128 ]; [64, 128 ]; [32, 128 ]; [16, 128 ]; ...
    [512, 64 ]; [256, 64 ]; [128, 64 ]; [64, 64 ]; [32, 64 ]; [16, 64 ]; ...
    [128, 32 ]; [64, 32 ]; [32, 32 ]; [16, 32 ];  ...
    [64, 16 ]; [32, 16 ]; [16, 16 ]];

%original_bing_window_sizes = [[64, 128]; [32, 128 ]];

original_num_window_sizes = size(original_bing_window_sizes, 1);

bing_window_sizes = [];

% we filter window sizes to fit inside the image
for i=1:size(original_bing_window_sizes, 1),
    window_wh = original_bing_window_sizes(i, :);
    
    if(sum(window_wh < im_wh) == 2),
        bing_window_sizes = [bing_window_sizes; window_wh];
    else
        % the window is disregarded
    end
    
    
end

num_window_sizes = size(bing_window_sizes, 1);

assert(num_samples > num_window_sizes);


candidates = [];
if num_window_sizes ~= original_num_window_sizes, 
    % we add one candidate that covers the whole image size
   candidates =  [0, 0, im_wh];
end
   

use_v0 = false;

if use_v0,
    % will have as many splits in x and y axis (non-square strides)
    
    sqrt_num_samples_size = floor(sqrt(num_samples / num_window_sizes));
    samples_per_size = sqrt_num_samples_size * sqrt_num_samples_size;
    assert(samples_per_size > 3);
    extra_samples = num_samples - (samples_per_size * num_window_sizes) - size(candidates, 1);
    delta_samples = ((sqrt_num_samples_size + 1) * (sqrt_num_samples_size + 1)) - samples_per_size;
    
    divisions_per_size = repmat(sqrt_num_samples_size, 1, num_window_sizes);
    
    start_extra_size_index = 6; % because we like 256x256 (and smaller)
    for i = start_extra_size_index:num_window_sizes,
        if extra_samples > delta_samples,
            divisions_per_size(i) = divisions_per_size(i) + 1;
            extra_samples = extra_samples - delta_samples;
        else
            fprintf('Added %i extra sizes\n', i - 1);
            break;
        end
    end
    
    
    for i = 1:num_window_sizes,
        
        num_divisions = divisions_per_size(i);
        window_wh = bing_window_sizes(i, :);
        x1 = 1;
        y1 = 1;
        x2 = im_wh(1) - window_wh(1);
        y2 = im_wh(2) - window_wh(2);
        
        assert(x2 > 1);
        assert(y2 > 1);
        
        [xx, yy] = meshgrid(linspace(x1, x2, num_divisions), ...
            linspace(y1, y2, num_divisions));
        top_left_xy = [xx(:) yy(:)];
        for j = 1:size(top_left_xy, 1),
            xy = top_left_xy(j, :);
            window = [xy, xy + window_wh];
            candidates = [candidates; window];
        end
        
    end
    
else
    % v1
    % will use square strides
    
    
    num_samples_per_size = floor(num_samples / num_window_sizes);
    
    stride_per_size = zeros(1, num_window_sizes);
    total_placed_samples = size(candidates, 1);
    for i = 1:num_window_sizes,
        
        window_wh = bing_window_sizes(i, :);
        x2 = im_wh(1) - window_wh(1);
        y2 = im_wh(2) - window_wh(2);
        
        assert(x2 > 0);
        assert(y2 > 0);
        
        block_area = (x2 * y2) / num_samples_per_size;
        stride = sqrt(block_area);
        assert(stride > 0);
               
        num_samples_placed = compute_num_samples_placed(im_wh, window_wh, stride);
        
        while(num_samples_placed > num_samples_per_size)
            stride = stride + 1; % larger stride, less placements
            num_samples_placed = compute_num_samples_placed(im_wh, window_wh, stride);
        end
    
        assert(num_samples_placed <= num_samples_per_size);
        
        total_placed_samples = total_placed_samples + num_samples_placed;
        assert(total_placed_samples <= num_samples);
        
        stride_per_size(i) = stride;
    end
    
    assert(total_placed_samples <= num_samples);
    
    sqrt_num_samples_size = floor(sqrt(num_samples / num_window_sizes));
    samples_per_size = sqrt_num_samples_size * sqrt_num_samples_size;
%     assert(samples_per_size > 3);
    assert(samples_per_size > 0);
    extra_samples = num_samples - total_placed_samples;
    assert(extra_samples >= 0);
    
    start_extra_size_index = 6; % because we like 256x256 (and smaller)
    for i = start_extra_size_index:num_window_sizes,
        
        window_wh = bing_window_sizes(i, :);
        stride = stride_per_size(i);
        
        num_samples_placed = compute_num_samples_placed(im_wh, window_wh, stride);
        
        new_stride = stride * 0.75;
        %new_stride = stride - 1;
        new_num_samples_placed = compute_num_samples_placed(im_wh, window_wh, new_stride);
        
        delta_samples = new_num_samples_placed - num_samples_placed;
        
        if extra_samples > delta_samples,
            stride_per_size(i) = new_stride;
            total_placed_samples = total_placed_samples + delta_samples;
            extra_samples = extra_samples - delta_samples;
        else
            fprintf('Added %i extra sizes\n', i - start_extra_size_index);
            break;
        end
    end
    
    if extra_samples > 0,
        fprintf('%i extra_samples remaining\n', extra_samples);
    end
    
    for i = 1:num_window_sizes,
        
        window_wh = bing_window_sizes(i, :);
        stride = stride_per_size(i);
        x1 = 1;
        y1 = 1;
        x2 = im_wh(1) - window_wh(1);
        y2 = im_wh(2) - window_wh(2);
        
        x_dots = x1 + (mod(x2 - x1, stride) / 2): stride : x2;
        y_dots = y1 + (mod(y2 - y1, stride) / 2): stride : y2;
        
        assert(x2 > 0);
        assert(y2 > 0);
        
        [xx, yy] = meshgrid(x_dots, y_dots);
        top_left_xy = [xx(:) yy(:)];
        for j = 1:size(top_left_xy, 1),
            xy = top_left_xy(j, :);
            window = [xy, xy + window_wh];
            candidates = [candidates; window];
        end
        
    end
    
    assert(size(candidates, 1) == total_placed_samples);
end

assert(size(candidates, 1) <= num_samples);
end


function [ret] = compute_num_samples_placed(im_wh, window_wh, stride)
x1 = 1;
y1 = 1;
x2 = im_wh(1) - window_wh(1);
y2 = im_wh(2) - window_wh(2);

x_dots = x1 + (mod(x2 - x1, stride) / 2): stride : x2;
y_dots = y1 + (mod(y2 - y1, stride) / 2): stride : y2;
num_samples_placed = size(x_dots, 2) * size(y_dots, 2);
ret = num_samples_placed;
end