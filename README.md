# Machine-Learning-Applications

IMPORTANT Note:

    These codes may use different versions of a library (e.g. sklearn).


This repository provides code of these applications:

1) Tic Tac Toe play and learn

   method: the Least Mean Square method
   
   coding: MatLab
   
2) Handwritten Digits Recognition on small images

   method: random forest
   
   coding: Python
   
   data:   train and test sets provided
   
3) curve_fitting.cpp is used to generate a curve to fit (x, y) data, by accurately estimating the coefficients of a best-fitting curve

   method: gradient descent
   
   coding: C++
   
   input manner: by a text file (e.g. .dat)
   
                 in input file, there are two columns of numbers (one column is x data, and the other is y data)
   
   recommended platform:  Linux
   
   run the code:  $ g++ curve_fitting.cpp  (will generate a.out)
   
                  $ ./a.out < input.dat

4) classifier.cpp is a code that implements k-NN 

   method: k-dimension tree (for both data structure and algorithm)
   
   coding: C++
   
   input manner: by a text file (e.g. .dat)
   
                 in input file, there are some columns of numbers (each column is a feature) with the last column as label.
   
   recommended platform:  Linux
   
   run the code:  $ g++ classifier.cpp  (will generate a.out)
   
                  $ ./a.out < input.dat

5) Python_MachineLearningLib.py

   This is my own library of machine-learning functions that call APIs from sklearn.

6) k-means_3_clusters_of_iris.rar

   This file include a Python code of k-means for iris and a text file of iris data in csv format.

7) kNN_iris_prediction_Python3.4.3.py

   This code of k-NN for iris classification 
   
       . loads data from sklearn.datasets.load_iris
       
       . creates, trains and uses sklearn.neighbors.KNeighborsClassifier
       
       . uses number of neighbors k = 25 (can be changed in code; suggest using an odd number)
       
       . evaluate accuracy of the model
       
       . prints the model with parameters

8) kNN_via_distance_calculation.py

   The k nearest neighbors are found via distance calculation and sorting the distances.

9) breast_cancer_diagnosis_via_SVM.rar

   It includes a Python code of using sklearn's SVC and a dataset of breast cancer.

10) internet_service_defection.rar

    It includes a Python code, a dataset and the generated reult file on the dataset.
    
    The code classifies/predicts service defection, and it employs the models of SVM, Random Forest, etc.

11) insurance_claim_loss_prediction_LinearRegression.rar

    It includes a Python code, train/test datasets and test result
    
    The code employs Linear Regression.
    
12) insurance_claim_loss_prediction_bayesianRidge.rar

    the code uses Bayesian Ridge model
    
    the zipped file includes a Python code, train/test datasets and test result.
