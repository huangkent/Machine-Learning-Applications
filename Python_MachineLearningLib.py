# import sys
# import os



def DecisionTree(x_train, y_train, x_test):
    # supervised learning (classification)
    # decision tree classification
    # it works for both categorical and continuous dependent variables.
    # This is done based on most significant attributes/ independent variables to make as distinct groups as possible
    # x_train - input values of the train dataset
    # y_train - target values of the train dataset
    # x_test - input values of the test dataset

    # It uses observations (about certain actions) and
    # identifies an optimal path for arriving at one of the desired outcomes

    from sklearn import tree

    # create the tree object
    model = tree.DecisionTreeClassifier(criterion='gini') # by default, gini; otherwise, entropy

    # train the model
    model.fit(x_train, y_train)

    # return prediction Output
    return model.predict(x_test)


def LinearRegression(x_train, y_train, x_test):
    # supervised learning
    # Linear Regression prediction, return prediction output
    # x_train - input values of the train dataset
    # y_train - target values of the train dataset
    # x_test - input values of the test dataset

    from sklearn import linear_model

    # Train the model using the training set
    linear_model.LinearRegression().fit(x_train, y_train)
    #linear_model.LinearRegression().score(x_train, y_train)    # check score

    # Equation slope and intercept
    print("equation slope: " + str(linear_model.LinearRegression().coef_))
    print("equation intercept: " + str(linear_model.LinearRegression().intercept_))

    # return prediction output
    return linear_model.LinearRegression().predict(x_test)


def LogisticRegression(x_train, y_train, x_test):
    # supervised learning
    # Logistic Regression prediction, return prediction output
    # x_train - input values of the train dataset
    # y_train - target values of the train dataset
    # x_test - input values of the test dataset

    from sklearn.linear_model import LogisticRegression

    # train the model
    LogisticRegression.fit(x_train, y_train)

    # Equation coefficient and Intercept
    print('Coefficient: \n', LogisticRegression.coef_)
    print('Intercept: \n', LogisticRegression.intercept_)

    # return predict Output
    return LogisticRegression.predict(x_test)


def KNN(K, x_train, y_train, x_test):
    # supervised learning (classification)
    # K Nearest Neighbors prediction, return prediction output
    # assumption: the presence of a particular feature in a class is unrelated to the presence of any other feature
    # can be used for both classification and regression problems.
    # K - number of nearest neighbors to be considered
    # x_train - input values of the train dataset
    # y_train - target values of the train dataset
    # x_test - input values of the test dataset

    from sklearn.neighbors import KNeighborsClassifier
    
    # Create KNN classifier object model 
    model = KNeighborsClassifier(n_neighbors = K) # default value for k_neighbors is 5

    # Train the model using the training sets and check score
    model.fit(x_train, y_train)

    # Predict Output
    return model.predict(x_test)


def K_means(K, x_train, x_test):
    # unsupervised learning
    # K-Means clustering
    # K - number of clusters
    # x_train - the train dataset (attributes/features)
    # x_test - the test dataset(attributes/features)

    # It groups a number of data points into a specific number of groups based on like characteristics

    from sklearn.cluster import KMeans

    # Create K-Means classifier object
    k_means = KMeans(n_clusters = K, random_state = 0)

    # Train the model using the training set
    k_means.fit(x_train)

    # Predict Output
    return k_means.predict(x_test)


def NaiveBayes(x_train, y_train, x_test):
    # supervised learning (classification)
    # Naive Bayes prediction, return prediction output
    # x_train - input values of the train dataset
    # y_train - target values of the train dataset
    # x_test - input values of the test dataset

    from sklearn.naive_bayes import GaussianNB

    # Create SVM classification object
    model = GaussianNB() 

    # Train the model using the training sets
    GaussianNB.fit(x_train, y_train)

    #Predict Output
    return GaussianNB.predict(x_test)


def RandomForest(x_train, y_train, x_test):
    # supervised learning
    # Random Forest classification, return prediction output
    # an ensemble of decision trees
    # x_train - input values of the train dataset
    # y_train - target values of the train dataset
    # x_test - input values of the test dataset

    from sklearn.ensemble import RandomForestClassifier

    # Create Random Forest object
    model= RandomForestClassifier()

    # Train the model using the training set
    model.fit(x_train, y_train)

    #Predict Output
    return model.predict(x_test)


def SVM(x_train, y_train, x_test):
    # supervised learning (classification)
    # Support Vector Machine prediction, return prediction output
    # x_train - input values of the train dataset
    # y_train - target values of the train dataset
    # x_test - input values of the test dataset

    from sklearn import svm

    # Create SVM classification object 
    model = svm.SVC()   # by default, classification

    # Train the model using the training sets
    model.fit(x_train, y_train)

    # return predict Output
    return model.predict(x_test)
