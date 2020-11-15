function [ oobErr, testErr ] = BaggedTrees( X_tr, Y_tr, X_te, Y_te, numBags )
%BAGGEDTREES Returns out-of-bag classification error of an ensemble of
%numBags CART decision trees on the input dataset, and also plots the error
%as a function of the number of bags from 1 to numBags
%   Inputs:
%       X_tr : Matrix of training data
%       Y_tr : Vector of classes of the training examples
%       X_te : Matrix of testing data
%       Y_te : Vector of classes of the testing examples
%       numBags : Number of trees to learn in the ensemble
%      
%
%   Outputs:
%       oobErr : the out-of-bag error as defined in lecture
%       testErr : the test error of the ensemble calculated on zip.test
%   You may use "fitctree" but do not use "TreeBagger" or any other inbuilt
%   bagging function

classes = unique(Y_tr);
Y_tr(Y_tr == classes(1)) = 1;
Y_tr(Y_tr == classes(2)) = -1;

Y_te(Y_te == classes(1)) = 1;
Y_te(Y_te == classes(2)) = -1;

oob_errors = zeros(1, numBags);
oob_idx = zeros(length(Y_tr), numBags);
predictions = NaN(length(Y_tr), numBags);

trees = cell(1, numBags);

for i = 1:numBags
   
    idx = randi(length(Y_tr), 1, length(Y_tr));
    features = X_tr(idx, :);
    labels = Y_tr(idx);
    oob_idx(idx, i) = 1;
  
    tree = fitctree(features, labels);
    trees{i} = tree;
    
    X_out = X_tr(oob_idx(:, i) == 0, :);   
    predictions(oob_idx(:,i) == 0,i) =  predict(tree,X_out);
  
   
    oob_errors(i) = nanmean(sign(nanmean(predictions(:,1:i),2)) ~= Y_tr);
end

oobErr = oob_errors(end);
testErr = tree_test_error(trees, X_te, Y_te);
figure
plot(1:numBags, oob_errors);
title("Number of Bags vs. Out-of-Bag Error (" + num2str(classes(1)) + " vs. " + num2str(classes(2)) + ")");
xlabel("Number of bags");
ylabel("OOB error");
end
