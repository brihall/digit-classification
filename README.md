# digit-classification
To run -  
Add all .m files as well as zip.train and zip.test to the same directory. Navigate to the directory in MATLAB. Call OneThreeFive in the MATLAB console. The script calls BaggedTrees.m (where most of the actual work is done) and tree_test_error.m to learn the models and find the test errors. Out-of-bag and test errors will be printed to the console and graphs of number of trees vs. out-of-bag error for one-vs-three and three-vs-five will be displayed.<br><br>
If you don't have access to MATLAB, the results are summarized in the PDF report included in the repository, and you can still look over the included .m files in GitHub. If the class that assigned this project hadn't required us to use MATLAB I probably would have used Python instead. <br><br>
I've included ThreeVsFive.m which calls BaggedTrees.m with random feature selection, but I don't recommend running it. It takes a while, and the only output are two graphs that are included in the PDF report.
