## *LocNet: Improving Localization Accuracy for Object Detection*

### Introduction

This code implements the following paper:    
**Title:**      "LocNet: Improving Localization Accuracy for Object Detection"    
**Authors:**     Spyros Gidaris, Nikos Komodakis    
**Institution:** Universite Paris Est, Ecole des Ponts ParisTech    
**ArXiv Link:**  http://arxiv.org/abs/1511.07763   
**Code:**        https://github.com/gidariss/LocNet    

**Abstract:**  
"We propose a novel object localization methodology with the purpose of boosting the localization accuracy of state-of-the-art object detection systems. Our model, given a search region, aims at returning the bounding box of an object of interest inside this region. To accomplish its goal, it relies on assigning conditional probabilities to each row and column of this region, where these probabilities provide useful information regarding the location of the boundaries of the object inside the search region and allow the accurate inference of the object bounding box under a simple probabilistic framework.  
For implementing our localization model, we make use of a convolutional neural network architecture that is properly adapted for this task, called LocNet. We show experimentally that LocNet achieves a very significant improvement on the mAP for high IoU thresholds on PASCAL VOC2007 test set and that it can be very easily coupled with recent state-of-the-art object detection systems, helping them to boost their performance. Furthermore, it sets a new state-of-the-art on PASCAL VOC2012 test set achieving mAP of 74.8%. Finally, we demonstrate that our detection approach can achieve high detection accuracy even when it is given as input a set of sliding windows, thus proving that it is independent of bounding box proposal methods."   


### Citing LocNet

If you find LocNet useful in your research, please consider citing:   

> @article{gidaris2015locnet,  
  title={LocNet: Improving Localization Accuracy for Object Detection},  
  author={Gidaris, Spyros and Komodakis, Nikos},  
  journal={arXiv preprint arXiv:1511.07763},  
  year={2015}  
}  

### License
This code is released under the MIT License (refer to the LICENSE file for details).  

### Requirements

**Hardware:**  
For training the LocNet models or testing the LocNet object detection pipeline you will require a  GPU with at least 6 Gbytes of memory.

**Software:**  
1. Modified version of Caffe developed to supprot LocNet and installed with the cuDNN library [[link](https://github.com/gidariss/caffe_LocNet)].  
2. MATLAB (tested with R2014b)  
  
**Optional:**  
The following packages are necessary for using the [EdgeBox](http://research.microsoft.com/apps/pubs/default.aspx?id=220569) or [Selective Search](http://koen.me/research/selectivesearch/) bounding box proposas algorithms:     
1. Edge Boxes code [[link](https://github.com/pdollar/edges)].    
2. The image processing MATLAB toolbox of Piotr Dollar (used for the Edge Boxes) [[link](http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html)].         
3. Selective search code [[link](http://huppelen.nl/publications/SelectiveSearchCodeIJCV.zip)].       
   
**Note:** we provide the bounding box proposals of PASCAL images and hence installing the above packages is not necessary for training or testing models on this dataset. However, they are necessary for running the demo code on images other than the one that is provided by us. 

### Installation (sufficient for the demo)

1. Download and install this modified version of [Caffe](https://github.com/gidariss/caffe_LocNet) developed to supprot LocNet. To clone Caffe on your local machine:
    ```Shell

    # $caffe_LocNet: directory where Caffe will be cloned  
    git clone https://github.com/gidariss/caffe_LocNet $caffe_LocNet
    ``` 
`$caffe_LocNet` is the directory where Caffe is cloned. After cloning Caffe follow the installation instructions [here](http://caffe.berkeleyvision.org/installation.html). **Note** that you have to install Caffe with the cuDNN library.   
2. Clone the ***LocNet*** code in your local machine:
    ```Shell

    # $LocNet: directory where LocNet will be cloned  
    git clone https://github.com/gidariss/LocNet $LocNet
    ``` 
From now on, the directory where ***LocNet*** was cloned will be called `$LocNet`.  
3. Create a symbolic link of [Caffe](https://github.com/gidariss/caffe_LocNet) installatation directory at `$LocNet/external/`:  
    ```Shell

    # $LocNet: directory where LocNet was cloned    
    # $caffe_LoNet: directory where caffe was cloned    
    ln -sf $caffe_LoNet $LocNet/external/caffe_LocNet    
    ```      

4.  open matlab from the `$LocNet/` directory:  
    ```Shell  

    cd $LocNet  
    matlab  
    ```      
5.  Run the `LocNet_build.m` script on matlab command line
    ```Shell

    # matlab command line enviroment
    >> LocNet_build    
    ``` 
  Do not worry about the warning messages. They also appear on my machine.  

### Download and use the pre-trained models

Download the tar.gz files with the pre-trained models:    
**Recognition models:**     
1. [Reduced MR-CNN recognition model](https://drive.google.com/file/d/0BwxkAdGoNzNTNFNKTzV3UnZGLW8/view?usp=sharing).     
2. [Fast RCNN recognition model](https://drive.google.com/file/d/0BwxkAdGoNzNTMDJUMGRhWV9qV2s/view?usp=sharing) (re-implemented by us).   
**Localization models:**  
1. [LocNet In-Out model](https://drive.google.com/file/d/0BwxkAdGoNzNTcFpaYkFVN3FraUU/view?usp=sharing).         
2. [LocNet Borders model](https://drive.google.com/file/d/0BwxkAdGoNzNTTF84MG5Xby1RRzA/view?usp=sharing).    
3. [LocNet Combined model](https://drive.google.com/file/d/0BwxkAdGoNzNTS3FPOVlWSGZUVjQ/view?usp=sharing).   
4. [CNN-based bounding box regression model](https://drive.google.com/file/d/0BwxkAdGoNzNTV0p2Vmh5YS1LRE0/view?usp=sharing).  

Untar and unzip all of the above files on the following locations:

   ```Shell
   
   # Recognition models:
   $LocNet/models-exps/VGG16_Reduced_MRCNN # Reduced MR-CNN recognition model
   $LocNet/models-exps/VGG16_FastRCNN # Fast RCNN recognition model
   # VOC2012 structure:
   $LocNet/models-exps/VGG16_LocNet_InOut # LocNet In-Out model
   $LocNet/models-exps/VGG16_LocNet_Borders # LocNet Borders model
   $LocNet/models-exps/VGG16_LocNet_Combined # LocNet Combined model
   $LocNet/models-exps/VGG16_BBoxReg # CNN-based bounding box regression model
   ```

All of the above models are based on the VGG16-Net and are trained on the union of VOC2007 train+val plus VOC2012 train+val datasets. Note that they are not the same as those used to report result in the paper and hense their performance is slightly different from them (around 0.2 mAP points difference more or less in the percentage scale).

### Demo

After having complete the basic installation, you will be able to run the demo of object detection based on the LocNet localization models. In order to do that, open the matlab from `$LocNet/` directory and then run the script `'demo_LocNet_object_detection_pipeline'` from the matlab command line enviroment:  
    
    
    cd $LocNet
    matlab
    # matlab command line enviroment
    >> demo_LocNet_object_detection_pipeline    
    
  
**Note:** you will require a GPU with at least 6 Gbytes of memory in order to run the demo. 

In order to play with the parameters of the object detection pipeline read and edit the [demo script](https://github.com/gidariss/LocNet/blob/master/code/examples/demo_LocNet_object_detection_pipeline.m):  
    
    # matlab command line enviroment
    >> edit demo_LocNet_object_detection_pipeline.m   
  
### Installing and using the box proposals algorithms (Optional)

First install 1) [Edge Boxes](https://github.com/pdollar/edges), 2) [Dollar's image processing MATLAB toolbox](http://vision.ucsd.edu/~pdollar/toolbox/doc/index.html) (used in Edge Boxes), and 3) [Selective Search](http://huppelen.nl/publications/SelectiveSearchCodeIJCV.zip). Then, create symbolic links of the installation directories at `$LocNet/external/`:       
   
    
    # $edges: installation directory of Edge Boxes     
    ln -sf $edges $LocNet/external/edges   
    # $pdollar-toolbox: installation directory of Dollar's image processing MATLAB toolbox  
    ln -sf $pdollar-toolbox $LocNet/external/pdollar-toolbox  
    # $selective_search: installation directory of selective search code  
    ln -sf $selective_search $LocNet/external/selective_search  
    
The above packages are necessary in case you want to try the demo on an image of your choise.  

### Downloading and preparing the PASCAL VOC2007 and VOC2012 datasets
  
In order to run experiments (e.g. train or test models) on [PASCAL VOC](http://host.robots.ox.ac.uk/pascal/VOC/) datasets you will need to download and prepare the corresponding data. For that purpose you will have to:

1. Download the VOC datasets and VOCdevkit:
   ```Shell
   
   # VOC2007 DATASET
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar # VOC2007 train+val set
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar # VOC2007 test set
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar # VOC2007 devkit
    # VOC2012 DATASET
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar # VOC2012 train+val set
    wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar  # VOC2012 devkit
   ```
2. Untar the VOC2007 tar files in a directory named `$datasets/VOC2007/VOCdevkit` and the VOC2012 tar files in a directory named `$datasets/VOC2012/VOCdevkit`:
   ```Shell
   
   mkdir $dataset
   # VOC2007 data:
    mkdir $datasets/VOC2007
    mkdir $datasets/VOC2007/VOCdevkit
    tar xvf VOCtrainval_06-Nov-2007.tar  -C $datasets/VOC2007/VOCdevkit
    tar xvf VOCtest_06-Nov-2007.tar -C $datasets/VOC2007/VOCdevkit
    tar xvf VOCdevkit_08-Jun-2007.tar -C $datasets/VOC2007/VOCdevkit
   # VOC2012 data:
    mkdir $datasets/VOC2012
    mkdir $datasets/VOC2012/VOCdevkit
    tar xvf VOCtrainval_11-May-2012.tar -C $datasets/VOC2012/VOCdevkit
    tar xvf VOCdevkit_18-May-2011.tar -C $datasets/VOC2012/VOCdevkit
   ```
3. They should have the following structure:
   ```Shell  
   
   # VOC2007 structure:
   $datasets/VOC2007/VOCdevkit/ # VOC2007 development kit
   $datasets/VOC2007/VOCdevkit/VOCcode/ # VOC2007 development kit code
   $datasets/VOC2007/VOCdevkit/VOC2007/ # VOC2007 images, annotations, etc 
   # VOC2012 structure:
   $datasets/VOC2012/VOCdevkit/ # VOC2012 development kit
   $datasets/VOC2012/VOCdevkit/VOCcode/ # VOC2012 development kit code
   $datasets/VOC2012/VOCdevkit/VOC2012/ # VOC2012 images, annotations, etc 
   ```
4. Create symlink of the `$datasets` directory at `$LocNet/datasets`:
   ```Shell
   
   ln -sf $datasets $LocNet/datasets  
   ```
5. Download the pre-computed [Edge Box](https://drive.google.com/file/d/0BwxkAdGoNzNTT194VzhIak9KSk0/view?usp=sharing) and [Selective Search](https://drive.google.com/file/d/0BwxkAdGoNzNTRTdtWTliWC1QQnc/view?usp=sharing) proposals of PASCAL images (click the links) and place them on the following locations:
   ```Shell

    $LocNet/data/edge_boxes_data # Edge Box proposals  
    $LocNet/data/selective_search_data # Selective Search proposals  
   ```

### Testing the (pre-)trained models on VOC2007 test set

To test the object detection pipeline with the (pre-)trained models on VOC2007 test do:
    
    # 1) open matlab from $LocNet directory
    cd $LocNet
    matlab
    # 2) run in the matlab command line enviroment
    >> script_test_object_detection_pipeline_PASCAL('VGG16_Reduced_MRCNN','VGG16_LocNet_InOut','bbox_proposals','edge_boxes','gpu_id',1);    
   

The above command will use the Reduced MR-CNN recognition model and the LocNet In-Out localization model located in the following directories:
   ```Shell

    $LocNet/models-exps/VGG16_Reduced_MRCNN # Reduced MR-CNN recognition model
    $LocNet/models-exps/VGG16_LocNet_InOut # LocNet In-Out localization model
   ```

In general, you can specify the recognition and localization models that will be tested by giving the directory names of the corresponding models as the 1st and 2nd input arguments in the `script_test_object_detection_pipeline_PASCAL` script. For more instructions on how to test the (pre-)trained models see the script:  
   ```Shell
    $LocNet/code/script_test_object_detection_pipeline_commands.m
   ```

### Training your own LocNet models in PASCAL VOC datasets

To train your own LocNet localization models on PASCAL VOC dataset:

1. Downlaod the archive file of [VGG16-Net model](https://drive.google.com/file/d/0BwxkAdGoNzNTLUVnWEg5aWRtaFU/view?usp=sharing) pre-trained on ImageNet classification task and then unzip and untar it in the following location: `$LocNet/data/vgg16_pretrained`.
2. Then   
   ```Shell
   
    # 1) open matlab from $LocNet directory
    cd $LocNet
    matlab
    # 2) run in the matlab command line enviroment
    >> script_train_LocNet_PASCAL('VGG16_LocNet_InOut_model','gpu_id',1,'loc_type','inout');  
   ```
The above command will train a LocNet In-Out localization model in the union of VOC2007 train+val and VOC2012 train+val datasets using both Selecive Search and Edge Box proposals. The trained model will be saved in the destination directory:  
   ```Shell

    $LocNet/models-exps/VGG16_LocNet_InOut_model 
   ```
of which the directory name is the string that is given as 1st input argument in the script `script_train_LocNet_PASCAL`. For more instructions on how to train your own LocNet localization models see the script:
   ```Shell
    $LocNet/code/script_train_localization_models_commands.m
   ```
   
**Note:** for training each of the LocNet models you will require a GPU with at least 6 Gbytes of memory.
