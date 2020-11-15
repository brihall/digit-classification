function [testErr] = tree_test_error(trees, X, Y)
%TREE_TEST_ERROR Calculates the test error using the supplied tree(s) and
%                the supplied test set
%   Inputs:
%       trees : A single tree or cell array of trees learned on the
%               training set
%       X : Matrix of testing data
%       Y : Vector of classes of the testing examples
%
%   Outputs: 
%       testErr : The binary classification error of the tree(s) on the
%                 test data
if length(trees) > 1    
    predictions = zeros(length(Y), length(trees));
    for i = 1:length(trees)
        predictions(:, i) = predict(trees{i}, X);
    end   
    testErr = mean(sign(mean(predictions,2)) ~= Y);
   
else  
    predictions = predict(trees, X);
    testErr = mean(sign(predictions) ~= Y);
end

end

