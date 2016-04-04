function printAPResults( classes, results )
if ~isnumeric(results)
    aps = [results(:).ap]';
else
    aps = results;
end

for i = 1:numel(classes)
    class_string = classes{i}(1:min(5,length(classes{i})));
    fprintf('& %5s ', class_string)
end
fprintf('& %5s \\\\ \n', 'mean')
for i = 1:numel(classes)
    fprintf('& %2.3f ', aps(i))
end
fprintf('& %2.3f \\\\ \n', mean(aps))

end