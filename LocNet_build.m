function LocNet_build()
if ~exist('liblinear_train')
    try
      fprintf('Compiling liblinear version 1.94\n');
      fprintf('Source code page:\n');
      fprintf('   http://www.csie.ntu.edu.tw/~cjlin/liblinear/\n');
      mex -outdir bin ...
          CFLAGS="\$CFLAGS -std=c99 -O3 -fPIC" -largeArrayDims ...
          external/liblinear-1.94/matlab/train.c ...
          external/liblinear-1.94/matlab/linear_model_matlab.c ...
          external/liblinear-1.94/linear.cpp ...
          external/liblinear-1.94/tron.cpp ...
          "external/liblinear-1.94/blas/*.c" ...
          -output liblinear_train;
    catch exception
        fprintf('Error message %s\n', getReport(exception));
    end
end

if ~exist('nms_mex')
    try
        fprintf('Compiling nms_mex\n');

        mex -outdir bin ...
          -largeArrayDims ...
          code/postprocessing/nms_mex.cpp ...
          -output nms_mex;
    catch exception
        fprintf('Error message %s\n', getReport(exception));
    end
end

if ~exist('nms_gpu_mex', 'file')
    try
       fprintf('Compiling nms_gpu_mex\n');
       addpath(fullfile(pwd, 'code', 'postprocessing'));
       nvmex('code/postprocessing/nms_gpu_mex.cu', 'bin');
       delete('nms_gpu_mex.o');
    catch exception
        fprintf('Error message %s\n', getReport(exception));
    end
end

end