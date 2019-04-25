# Machine-Learning-Applications

This repository provides code of these application:

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
