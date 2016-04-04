function [ VOCopts ] = initVOCOpts( voc_path, voc_year )

VOCopts.datadir = [voc_path, filesep];
VOCopts.dataset = sprintf('VOC%s', voc_year);
VOCopts.resdir  = [voc_path, filesep, 'results', filesep, VOCopts.dataset, filesep];

% % change this path to a writable local directory for the example code
% VOCopts.localdir = [voc_path, filesep, 'local',  filesep, VOCopts.dataset '/'];

% initialize the test set

%VOCopts.testset='val'; % use validation data for development test set
VOCopts.testset='test'; % use test set for final challenge

% initialize main challenge paths

VOCopts.annopath      = [VOCopts.datadir VOCopts.dataset '/Annotations/%s.xml'];
VOCopts.imgpath       = [VOCopts.datadir VOCopts.dataset '/JPEGImages/%s.jpg'];
VOCopts.imgsetpath    = [VOCopts.datadir VOCopts.dataset '/ImageSets/Main/%s.txt'];
VOCopts.clsimgsetpath = [VOCopts.datadir VOCopts.dataset '/ImageSets/Main/%s_%s.txt'];
VOCopts.clsrespath    = [VOCopts.resdir 'Main/%s_cls_' VOCopts.testset '_%s.txt'];
VOCopts.detrespath    = [VOCopts.resdir 'Main/%s_det_' VOCopts.testset '_%s.txt'];

% initialize segmentation task paths

VOCopts.seg.clsimgpath  = [VOCopts.datadir, VOCopts.dataset, '/SegmentationClass/%s.png'];
VOCopts.seg.instimgpath = [VOCopts.datadir, VOCopts.dataset, '/SegmentationObject/%s.png'];
VOCopts.seg.imgsetpath  = [VOCopts.datadir, VOCopts.dataset, '/ImageSets/Segmentation/%s.txt'];

VOCopts.seg.clsresdir   = [VOCopts.resdir, 'Segmentation/%s_%s_cls'];
VOCopts.seg.instresdir  = [VOCopts.resdir, 'Segmentation/%s_%s_inst'];
VOCopts.seg.clsrespath  = [VOCopts.seg.clsresdir, '/%s.png'];
VOCopts.seg.instrespath = [VOCopts.seg.instresdir, '/%s.png'];

% initialize layout task paths

VOCopts.layout.imgsetpath = [VOCopts.datadir, VOCopts.dataset, '/ImageSets/Layout/%s.txt'];
VOCopts.layout.respath    = [VOCopts.resdir, 'Layout/%s_layout_', VOCopts.testset, '_%s.xml'];

% initialize the VOC challenge options

% VOC2007 classes
VOCopts.classes={...
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'};


VOCopts.nclasses=length(VOCopts.classes);	

VOCopts.poses={...
    'Unspecified'
    'SideFaceLeft'
    'SideFaceRight'
    'Frontal'
    'Rear'};

VOCopts.nposes=length(VOCopts.poses);

VOCopts.parts={...
    'head'
    'hand'
    'foot'};    

VOCopts.maxparts=[1 2 2];   % max of each of above parts

VOCopts.nparts=length(VOCopts.parts);

VOCopts.minoverlap=0.5;

% % initialize example options
% 
% VOCopts.exannocachepath=[VOCopts.localdir '%s_anno.mat'];
% 
% VOCopts.exfdpath=[VOCopts.localdir '%s_fd.mat'];

end

