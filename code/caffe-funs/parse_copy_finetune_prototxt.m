function [solver_file, train_net_file, test_net_file, max_iter, snapshot_prefix] = ...
    parse_copy_finetune_prototxt(solver_file_path, dest_dir)
% 
% This file is part of the code that implements the following ICCV2015 accepted paper:
% title: "Object detection via a multi-region & semantic segmentation-aware CNN model"
% authors: Spyros Gidaris, Nikos Komodakis
% institution: Universite Paris Est, Ecole des Ponts ParisTech
% Technical report: http://arxiv.org/abs/1505.01749
% code: https://github.com/gidariss/mrcnn-object-detection
% 
% It is adapted from the SPP-Net code: 
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


% copy solver, train_net and test_net to destination folder
% ------------------------------------------------  

[folder, solver_file, ext] = fileparts(solver_file_path);
solver_file = [solver_file, ext];

solver_prototxt_text = textread(solver_file_path, '%[^\n]');
try  % for old caffe
    train_net_file_pattern = '(?<=train_net[ :]*")[^"]*(?=")';
    test_net_file_pattern = '(?<=test_net[ :]*")[^"]*(?=")';

    train_net_file = cellfun(@(x) regexp(x, train_net_file_pattern, 'match'), solver_prototxt_text, 'UniformOutput', false);
    train_net_file = train_net_file(cellfun(@(x) ~isempty(x), train_net_file, 'UniformOutput', true));
    if isempty(train_net_file)
        error('invalid solver file %s \n', solver_file_path);
    end
    train_net_file = train_net_file{1}{1};

    test_net_file = cellfun(@(x) regexp(x, test_net_file_pattern, 'match'), solver_prototxt_text, 'UniformOutput', false);
    test_net_file = test_net_file(cellfun(@(x) ~isempty(x), test_net_file, 'UniformOutput', true));
    if isempty(test_net_file)
        error('invalid solver file %s \n', solver_file_path);
    end
    test_net_file = cellfun(@(x) x{1}, test_net_file, 'UniformOutput', false);
catch  % for new caffe
    train_test_net_file_pattern = '(?<=net[ :]*")[^"]*(?=")';
    train_test_net_file_pattern = cellfun(@(x) regexp(x, train_test_net_file_pattern, 'match'), solver_prototxt_text, 'UniformOutput', false);
    train_test_net_file_pattern = train_test_net_file_pattern(cellfun(@(x) ~isempty(x), train_test_net_file_pattern, 'UniformOutput', true));
    if isempty(train_test_net_file_pattern)
        error('invalid solver file %s \n', solver_file_path);
    end
    train_net_file = train_test_net_file_pattern{1}{1};
    test_net_file  = {train_net_file};
end
mkdir_if_missing(dest_dir);
copyfile(fullfile(folder, solver_file),    dest_dir);
copyfile(fullfile(folder, train_net_file), dest_dir);

for i = 1:length(test_net_file), copyfile(fullfile(folder, test_net_file{i}),  dest_dir); end

max_iter_pattern = '(?<=max_iter[ :]*)[0-9]*';
max_iter = cellfun(@(x) regexp(x, max_iter_pattern, 'match'), solver_prototxt_text, 'UniformOutput', false);
max_iter = max_iter(cellfun(@(x) ~isempty(x), max_iter, 'UniformOutput', true));
if isempty(max_iter)
    error('invalid solver file %s \n', solver_file_path);
end
max_iter = str2double(max_iter{1}{1});

snapshot_prefix_pattern = '(?<=snapshot_prefix[ :]*")[^"]*(?=")';
snapshot_prefix = cellfun(@(x) regexp(x, snapshot_prefix_pattern, 'match'), solver_prototxt_text, 'UniformOutput', false);
snapshot_prefix = snapshot_prefix(cellfun(@(x) ~isempty(x), snapshot_prefix, 'UniformOutput', true));
if isempty(snapshot_prefix)
    error('invalid solver file %s \n', solver_file_path);
end
snapshot_prefix = snapshot_prefix{1}{1};
end
