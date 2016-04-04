% 
% This file is part of the code that implements the following paper:
% Title      : "LocNet: Improving Localization Accuracy for Object Detection"
% Authors    : Spyros Gidaris, Nikos Komodakis
% Institution: Universite Paris Est, Ecole des Ponts ParisTech
% ArXiv link : http://arxiv.org/abs/1511.07763
% code       : https://github.com/gidariss/LocNet
%
% AUTORIGHTS
% --------------------------------------------------------
% Copyright (c) 2016 Spyros Gidaris
% 
% Title     : "LocNet: Improving Localization Accuracy for Object Detection"
% ArXiv link: http://arxiv.org/abs/1511.07763
% Licensed under The MIT License [see LICENSE for details]
% ---------------------------------------------------------

% Commands for training each of the localization models on the union of 
% PASCAL VOC 2007 train+val and VOC2012 train+val datasets using both the 
% selective search and the edge box proposals and flipped versions of the 
% images. Note that the following script should be called from the
% installation directory of the code.


% ************************ LOCNET MODELS **********************************
% train the LocNet InOut localization model 
script_train_LocNet_PASCAL('VGG16_LocNet_InOut','loc_type','inout','gpu_id',1);
% train the LocNet Borders localization model 
script_train_LocNet_PASCAL('VGG16_LocNet_Borders','loc_type','borders','gpu_id',1);
% train the LocNet Combined localization model
script_train_LocNet_PASCAL('VGG16_LocNet_Combined','loc_type','combined','gpu_id',1);
% *************************************************************************

% ********************* BOUNDING BOX REGRESSION MODEL *********************
% train the CNN-based bounding box regression model. 
script_train_LocNet_PASCAL('VGG16_BBoxReg','loc_type','bboxreg','scale_ratio',1.3,'gpu_id',1);
% Note that the bounding box regression model that will be trained with the
% above script will be slightly inferior than this used for reporting 
% results in the paper. The reason is that the model used in the paper was
% sharing its convolutional layers with the Fast-RCNN recognition model 
% during training (see supplementary material).
% *************************************************************************