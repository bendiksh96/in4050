import numpy as np
import matplotlib.pyplot as plt
import sklearn #for datasets
from sklearn.datasets import make_blobs
X, t_multi = make_blobs(n_samples=[400,400,400, 400, 400], 
                        centers=[[0,1],[4,2],[8,1],[2,0],[6,0]], 
                        cluster_std=[1.0, 2.0, 1.0, 0.5, 0.5],
                        n_features=2, random_state=2022)
indices = np.arange(X.shape[0])
rng = np.random.RandomState(2022)
rng.shuffle(indices)
indices[:10]    
X_train = X[indices[:1000],:]
X_val = X[indices[1000:1500],:]
X_test = X[indices[1500:],:]
t_multi_train = t_multi[indices[:1000]]
t_multi_val = t_multi[indices[1000:1500]]
t_multi_test = t_multi[indices[1500:]]

t2_train = t_multi_train >= 3
t2_train = t2_train.astype('int')
t2_val = (t_multi_val >= 3).astype('int')
t2_test = (t_multi_test >= 3).astype('int')

def add_bias(X, bias):
    """X is a Nxm matrix: N datapoints, m features
    bias is a bias term, -1 or 1. Use 0 for no bias
    Return a Nx(m+1) matrix with added bias in position zero
    """
    N = X.shape[0]
    biases = np.ones((N, 1))*bias # Make a N*1 matrix of bias-s
    # Concatenate the column of biases in front of the columns of X.
    return np.concatenate((biases, X), axis  = 1) 

class NumpyClassifier():
    """Common methods to all numpy classifiers --- if any"""

class MLPBinaryLinRegClass(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden = 10):
        """Intialize the hyperparameters"""
        self.bias1 = bias
        self.bias2 = bias
        
        self.dim_hidden = dim_hidden
        
        def logistic(x):
            return 1/(1+np.exp(-x))
        self.activ = logistic
        
        def logistic_diff(x):#y):
            return 1/(1+np.exp(-x))*(1-1/(1+np.exp(-x)))
        self.activ_diff = logistic_diff
        
    def fit(self, X_train, t_train, eta=0.000108, epochs = 1000, X_val = 0, t_val = 0):
        """Intialize the weights. Train *epochs* many epochs.
        
        X_train is a Nxm matrix, N data points, m features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        """
        self.eta = eta
        
        T_train = t_train.reshape(-1,1)
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Initilaize the weights
        self.weights1 = (np.random.rand(dim_in +1, self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand(self.dim_hidden , dim_out+1) * 2 - 1)/np.sqrt(self.dim_hidden)
        #print(self.weights1.shape, self.weights2.shape)
        X_train_bias = add_bias(X_train, self.bias1)
        
        for e in range(epochs):
            # One epoch
            z1, a1, z2, a2 = self.forward(X_train_bias)
            # The forward step
            
            err_l2 = self.error_grad(X_train, T_train)  * self.activ_diff(X_train)     
            #Denne er litt shady:
            err_l1 = (err_l2@(self.weights2).T).T @ self.activ_diff(a2)
            
            #Local error
            w2_grad = (z2) * err_l2
            w1_grad = (X_train)@err_l1.T
            #Update gradient
            
            bias1_grad = np.sum(err_l1)
            bias2_grad = np.sum(err_l2)
            #Update gradient of bias
            
            
            print(self.weights1.shape, self.weights2.shape, w1_grad.shape,w2_grad.shape)
            exit()
            self.weights1 -= self.eta * w1_grad
            self.weights2 -= self.eta * w2_grad
            #Update weight
            
            self.bias1 -= self.eta*bias1_grad
            self.bias2 -= self.eta*bias2_grad
            #Update bias
            
            
    def forward(self, X):
        """Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        
        z1 = np.matmul(X,self.weights1) + self.bias1
        a1 = self.activ(z1)

        z2 = np.matmul(a1, self.weights2) + self.bias2
        a2 = self.activ(z2)
        
        return z1, a1, z2, a2
    
    def predict(self, X):
        """Predict the class for the mebers of X"""
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return (score > 0.5)
    
    def bce(self, pred, t2):
        eps = 1e-7   #In order to avoid log(0)
        loss = np.mean((-t2 * np.log(pred+eps) + (1 - t2) * np.log(1 - pred+eps)))
        return loss
    
    def error_grad(self, X, true):
        # print("X: ", X.shape, "t:", true.shape)
        der = np.divide((X-true),(X*(1-X)))
        return der
    

def plot_decision_regions(X, t, clf=[], size=(8,6)):
    """Plot the data set (X,t) together with the decision boundary of the classifier clf"""
    # The region of the plane to consider determined by X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Make a make of the whole region
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # Classify each meshpoint.
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=size) # You may adjust this

    # Put the result into a color plot
    plt.contourf(xx, yy, Z, alpha=0.2, cmap = 'Paired')

    plt.scatter(X[:,0], X[:,1], c=t, s=10.0, cmap='Paired')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("Decision regions")
    plt.xlabel("x0")
    plt.ylabel("x1")
    plt.show()

cl = MLPBinaryLinRegClass()
cl.fit(X_train, t2_train)
# plot_decision_regions(X_train, t2_train, cl)