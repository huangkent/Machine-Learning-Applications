// This is a k-Nearest Neighbors classifier using k-dimension tree
// the k nearest neighbors will be among the nodes visited during searching for the query node in the tree
// if the number of the visited nodes is odd, then k equals to the number; otherwise, k = the number - 1
// if the number of the visited nodes is zero, then the tree is empty, i.e. no input training data yet
//
// This code also provides functions for in-order traversal of the k-d tree, plus displaying nodes

#include <iostream>
#include <string>
#include <vector>
#include <stack>
#include <cstdlib>

using namespace std;


// k-d tree node
struct kdNode {
    vector<double> features;
    string label;
    kdNode *left, *right;
};
typedef struct kdNode kdNode;

// a stack of k-d tree node pointers
stack <kdNode*> s;


// create a new node of k-d tree
kdNode* new_kdNode(vector<double> features, string label)  {
    // given a feature vector and a label, create a k-d tree node
    
    kdNode *p = new kdNode;
    
    p->features = features;
    p->label = label;
    
    return p;
}


// insert a new node p to a k-d tree root,
// given the number of dimensions and the next dimension
// and return the root of the modified tree
kdNode* insert_kdNode(kdNode* p, kdNode* root, int k, int d)  {
    // k means the tree is a k-dimension tree
    // d means the dth dimension, i.e. 0, 1, 2, ...
    
    if (root == NULL)  {
        return p;
    }
    
    if (p->features[d] < root->features[d])  {
        // insert p to the left subtree of root, 
        // based on the dimension following the dth dimension
        root->left = insert_kdNode(p, root->left, k, (d + 1) % k);
    }
    else  {
        // insert p to the right subtree of root, 
        // based on the dimension following the dth dimension
        root->right = insert_kdNode(p, root->right, k, (d + 1) % k);
    }
    
    return root;
}


void display_kdNode(kdNode* p)  {
    if (p != NULL)  {
        for (unsigned int i = 0; i < p->features.size(); i++)  {
            cout << p->features[i] << ' ';
        }
        cout << p->label << ' ';
    }
}


void inorderTraversal(kdNode* root)  {
    if (root == NULL)  { return; }
    
    inorderTraversal(root->left);
    display_kdNode(root);
    cout << "  ";
        inorderTraversal(root->right);
}


void read_kdTree(kdNode* root)  {
    inorderTraversal(root);
}


// determine if two points are same in k-dimension space 
bool arePointsSame(kdNode* p, kdNode* q, int k)  {
    for (int i = 0; i < k; i++)     // compare individual feature values
        if (p->features[i] != q->features[i]) 
            return false; 

    return true; 
} 


// Search for a point p in the k-d tree root, with total dimensions k
bool searchPoint(kdNode* root, kdNode* p, int k, int depth)  { 
    // The parameter depth is used to determine current axis

    // Base cases 
    if (root == NULL)
        return false;

    if (arePointsSame(root, p, k))  {
        s.push(root);               // save the root pointer in stack s
        return true;
    }
    s.push(root);                   // save the root pointer in stack s

    // Current dimension is computed using current depth and total dimensions (k)
    int cd = depth % k; 

    // Compare p with root with respect to cd
    if (p->features[cd] < root->features[cd]) 
        return searchPoint(root->left, p, depth + 1, k);

    return searchPoint(root->right, p, depth + 1, k);
} 


void display_nodes_in_stack(stack <kdNode*> s)  {
    
    while (!s.empty()) 
    { 
        display_kdNode(s.top());
        cout << '\n'; 
        s.pop(); 
    } 
} 


int main()  {

    kdNode *root = NULL;
    kdNode *p, *q;
    vector<double> features;
    string label;

    string st;
    int m, n, j;
    string v = "";
    vector<string> V;
    
    int vote[2] = {0, 0};   // for yes and no

    cout << "Input data via file ... " << endl;

    while (getline(cin, st))  {
        n = st.size();

        // split the input line at spaces
        // save the substrings in a vector
        for (int i = 0; i < n; i++)  {
            if (st[i] != ' ')  {
                v += st[i];
            }
            else if (v != "")  {
                V.push_back(v);
                v = "";
            }
        }
        if (v != "")  {
            V.push_back(v);
            v = "";
        }
        
        // convert the substrings to numbers, and save them as features
        // if the last substring is a label, save it as label
        int k = V.size() - 1;   // k might be the total dimensions
        for (int i = 0; i < k; i++)  {
            features.push_back(atof(V[i].c_str()));
        }
        if (V[k] == "Yes" || V[k] == "yes" || V[k] == "No" || V[k] == "no")  {
            label = V[k];
        }
        else  {
            features.push_back(atof(V[k].c_str()));
            label = "";
        }
        
        // add labeled node to k-d tree, and classify unlabeled node
        p = new_kdNode(features, label);
        if (label == "")  {
            // classify the unclassified node
            searchPoint(root, p, features.size(), 0);
            m = s.size();           // m nodes were visited in the k-d tree during search
            display_kdNode(p);
            if (s.empty())  {
                cout << "can not be classified since no points input yet" << endl;
            }
            else  {
                j = 1;
                if (m % 2 == 0)  { m --; }
                while (!s.empty()  && j <= m)  {
                    // stack s stores the addresses of the nodes visited in the k-d tree during search
                    q = s.top();
                    if (q->label == "Yes" || q->label == "yes")  {
                        vote[0] ++;
                    }
                    else if (q->label == "No" || q->label == "no")  {
                        vote[1] ++;
                    }
                    j ++;
                }
                if (vote[0] > vote[1])  {
                    cout << "Yes" << endl;
                }
                else  {
                    cout << "No" << endl;
                }
            }
        }
        else  {
            // insert the labeled node to the k-d tree
            root = insert_kdNode(p, root, features.size(), 0);
        }

        V.clear();
        features.clear();
    }

    return 0;
}
