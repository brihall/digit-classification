% Script to load data from zip.train, filter it into datasets with only one
% and three or three and five, and compare the performance of plain
% decision trees (cross-validated) and bagged ensembles (OOB error)

% Added test data set so that the results requested in part c can be
% obtained all in one script
load zip.train;
train = zip;
load zip.test;
test = zip;

fprintf('Working on the one-vs-three problem...\n\n');
subsample = train(find(train(:,1)==1 | train(:,1) == 3),:);
Y_tr = subsample(:,1);
X_tr = subsample(:,2:257);
% Extract appropriate test set
subsample = test(find(test(:,1)==1 | test(:,1) == 3),:);
Y_te = subsample(:,1);
X_te = subsample(:,2:257);
ct = fitctree(X_tr,Y_tr,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);


% Pass test set as paramaters to BaggedTrees
[bee, err_ens] = BaggedTrees(X_tr, Y_tr, X_te, Y_te, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n\n', bee);

% Train a new single decision tree because param 'CrossVal' makes predict
% not work. Also need to change label coding to +/-1 for the single tree.
Y_tr_1 = Y_tr;
Y_te_1 = Y_te;
Y_tr_1(Y_tr_1 == 1) = 1;
Y_tr_1(Y_tr_1 == 3) = -1;
Y_te_1(Y_te_1 == 1) = 1;
Y_te_1(Y_te_1 == 3) = -1;
ct = fitctree(X_tr, Y_tr_1);
err_1 = tree_test_error(ct, X_te, Y_te_1);

fprintf('The test error of a single decision tree is %.4f\n', err_1);
fprintf('The test error of an ensemble of 200 trees is %.4f\n', err_ens);


fprintf('\nNow working on the three-vs-five problem...\n\n');
subsample = train(find(train(:,1)==3 | train(:,1) == 5),:);
Y_tr = subsample(:,1);
X_tr = subsample(:,2:257);
% Extract appropriate test set
subsample = test(find(test(:,1)==3 | test(:,1) == 5),:);
Y_te = subsample(:,1);
X_te = subsample(:,2:257);
ct = fitctree(X_tr,Y_tr,'CrossVal','on');
fprintf('The cross-validation error of decision trees is %.4f\n', ct.kfoldLoss);

% Pass test set as paramaters to BaggedTrees
[bee, err_ens] = BaggedTrees(X_tr, Y_tr, X_te, Y_te, 200);
fprintf('The OOB error of 200 bagged decision trees is %.4f\n\n', bee);

% Train a new single decision tree because param 'CrossVal' makes predict
% not work. Also need to change label coding to +/-1 for the single tree.
Y_tr_1 = Y_tr;
Y_te_1 = Y_te;
Y_tr_1(Y_tr_1 == 3) = 1;
Y_tr_1(Y_tr_1 == 5) = -1;
Y_te_1(Y_te_1 == 3) = 1;
Y_te_1(Y_te_1 == 5) = -1;
ct = fitctree(X_tr, Y_tr_1);
err_1 = tree_test_error(ct, X_te, Y_te_1);
fprintf('The test error of a single decision tree is %.4f\n', err_1);
fprintf('The test error of an ensemble of 200 trees is %.4f\n', err_ens);