#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <cstdlib>
// #include <algorithm>
using namespace std;


class linearModel  {

    public:         // public variables
        vector<vector<double> > X;
        vector<double> X_input;

        vector<double> Y;
        int bestOrder;
        vector<double> bestWeights;


    linearModel()  {

         model();
    }


    void keyboard_input()  {

        double x, y;

        cout << "Enter x y (or input via file): " << endl;

        while ( cin >> x >> y )  {

            X_input.push_back(x);

            Y.push_back(y);
        }


    }


    void model()   {

        keyboard_input();

        int num_iteration = 5000;
        float alpha = 0.05;


        /*  polynomial of order 0: y = w0  */

        // initialize weight vector W = <w0>, where w0 = 1
        vector<double> W;
        W.push_back(1);
        // create the training set X, for the 0th-order polynomial
        X = createTrainingSet(X_input, 0);

        // adapt the weight vector to (x, y), using gradient descent
        W = gradientDescent(W, X, Y, alpha, num_iteration);
        // evaluate loss function
        double L = loss(W, X, Y);

        // minLoss = the so-far smallest loss
        // and the so-far best order polynomial
        double minLoss = L;
        bestOrder = 0;
        bestWeights = W;

        /*  dth-order polynomials y = w0 + w1*x + ... + wd*x^d, d = 1, 2, ...  */

        double L_pre, L_now;	// the previous and current losses
        L_pre = L;
        L_now = L_pre;
        
        int order = 1;
        while (L_pre >= L_now)  {
            L_pre = L_now;

            // initialize weight vector
            W.clear();
            for (int i = 0; i <= order; i++)  {
                W.push_back(1);
            }
            // create vector X = <1, x, x^2, ...>
            X = createTrainingSet(X_input, order);
            // adapt W to (x, y)
            W = gradientDescent(W, X, Y, alpha, num_iteration);
            // evaluate loss function
            L_now = loss(W, X, Y);

            // pick the so-far smallest loss
            // and correspondingly the best order/weights of polynomials
            if (minLoss >= L_now)  {
                minLoss = L_now;
                bestOrder = order;
                bestWeights = W;
            }
            order ++;
        }
    }


    double loss(vector<double> W, vector<vector<double> > X, vector<double> Y)  {
        // loss = summation( (Y[i] - W*X[i])^2 ) over i = 0, 1, 2, ...

        double L;
        double computedYi;
        int m = Y.size();    // the size of data

        L = 0;
        for (int i = 0; i < m; i++)  {
            computedYi = calculateYi(W, X[i]);
            L += pow( (Y[i] - computedYi), 2 );
        }
        L = L / (2.0*m);

        return L;	// the loss
    }


    vector<double> gradientDescent(vector<double> W, vector<vector<double> > X, vector<double> Y, double alpha, int num_iteration)  {

        int m = Y.size();    // the size of data
        double delta;
        vector<double> H;

        for (int i = 0; i < num_iteration; i++)  {
	    // Y = W*X'
            H = calculateYs(W, X);
            // update weight vector W
            for (unsigned int j =0; j < W.size(); j++)  {
                delta = 0;
                for (int k = 0; k < m; k++)  {
                    delta += (H[k] - Y[k]) * X[k][j];
                }
                W[j] -= (alpha / m) * delta;
            }
        }

        return W;
    }


    vector<vector<double> > createTrainingSet(vector<double> X_input, int order)  {
    // X_input -- the input X values
    // order -- the order of the polynomial to fit the data

        vector<double> X_powers;
        vector<vector<double> > X;
        double xPower;
    
        for (unsigned int i = 0; i < X_input.size(); i++)  {
            X_powers.clear();
            X_powers.push_back(1);			    // the 0th power of Xi
            for (int j = 1; j <= order; j++)  {
                xPower = pow(X_input[i], j);	// the jth power of Xi
                X_powers.push_back(xPower);		// save the powers
            }
            X.push_back(X_powers);
        }

        return X;
    }


    double calculateYi(vector<double> W, vector<double> Xi)  {
    // calculate Yi = W * Xi

        double Yi;

        Yi = 0;
        for (unsigned int j = 0; j <= W.size(); j++)  {
            Yi += W[j] * Xi[j];
        }

        return Yi;
    }


    vector<double> calculateYs(vector<double> W, vector<vector<double> > X)  {

        vector<double> Y;
        double y;

        for (unsigned int i = 0; i < X.size(); i++)  {	// X.size() is the # of X values
            y = 0;
            for (unsigned int j = 0; j <= W.size(); j++)  {
                y += W[j] * X[i][j];
            }
            Y.push_back(y);
        }

        return Y;
    }

};


int main() {
  
    linearModel curveFitter;

    cout << endl << "The weights of the curve model:" << endl;
    for (unsigned int i = 0; i < curveFitter.bestWeights.size(); i++)  {
        cout << curveFitter.bestWeights[i] << endl;
    }

    return 0;
}
