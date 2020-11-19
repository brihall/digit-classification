% Script to plot effects of randomly selecting features and increasing 
% the size of the ensemble for the three-vs-five problem.

% Assumes OneThreeFive has already been run so that the data is loaded.

dropOOBErr = zeros(1, 10);
dropTestErr = zeros(1, 10);
props = cat(2, .01:.01:.1, .2:.1:1);
for i=1:length(props)
    [ oobErr, testErr ] = BaggedTrees(X_tr_35, Y_tr_35, X_te_35, Y_te_35, 200, props(i), false);
    dropOOBErr(i) = oobErr;
    dropTestErr(i) = testErr;
end

figure
plot(props, dropOOBErr);
title("Proportion of kept features vs. Out-of-Bag Error (3 vs. 5)");
xlabel("Proportion of kept features");
ylabel("OOB error");

figure
plot(props, dropTestErr);
title("Proportion of kept features vs. Test Error (3 vs. 5)");
xlabel("Proportion of kept features");
ylabel("Test error");

numTrees = 200:25:500;
moreTreesOOBErr = zeros(1, length(numTrees));
moreTreesTestErr = zeros(1, length(numTrees));
for i=1:length(numTrees)
   [oobErr, testErr] =  BaggedTrees(X_tr_35, Y_tr_35, X_te_35, Y_te_35, 200, .1, false);
   moreTreesOOBErr(i) = oobErr;
   moreTreesTestErr(i) = testErr;
end

figure
plot(numTrees, moreTreesOOBErr);
title("Number of Bags vs. Out-of-Bag Error (3 vs. 5 w/ 10% of features kept)");
xlabel("Number of bags");
ylabel("OOB error");

figure
plot(numTrees, moreTreesTestErr);
title("Number of bags vs. Test Error (3 vs. 5 w/ 10% of features kept)");
xlabel("Number of bags");
ylabel("Test error");