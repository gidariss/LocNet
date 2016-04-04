function image = get_image(image_path)
[~,~,ext]   = fileparts(image_path);
flip_suffix = ['_flip',ext];
num_chars   = length(flip_suffix);
if strcmp(image_path((end-num_chars+1):end),flip_suffix)
    img_name = regexprep(image_path, flip_suffix, ext);
    image    = imread(img_name);
    image    = fliplr(image);
else
    image = imread(image_path);
end
if size(image,3) == 1, image = repmat(image, [1, 1, 3]); end
end
