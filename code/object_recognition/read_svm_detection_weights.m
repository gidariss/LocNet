function model = read_svm_detection_weights( filepath )
load(filepath, 'weights', 'bias');
model = {weights; bias};
end