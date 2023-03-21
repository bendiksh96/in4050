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

#Binary representation
t2_train = t_multi_train >= 3
t2_train = t2_train.astype('int')
#T2 is now 0 or 1 based on whether if it is larger or smaller than 3.
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
    
class NumpyLogisticClass(NumpyClassifier):
    def __init__(self, bias=-1):
        self.bias=bias
    
    #Åpne for muligheten til å legge in validation-set, for å regne på loss og accuracy
    def fit(self, X_train, t_train, eta = 0.0001, epochs=200, X_val = 0, t_val= 0):
        """X_train is a Nxm matrix, N data points, m features
        t_train is a vector of length N,
        the targets values for the training data"""
        if self.bias:
            X_train = add_bias(X_train, self.bias)

        #Konvergens trengs for loss, om det ikke forbedres, må loopen avsluttes. 
        num_epochs_no_update = 5
        tol = 0.001
        conv =  False
        self.w = w = 0.2
                    
        (N, m) = X_train.shape
        
        self.weights = weights = np.zeros(m)
        self.accuracy_list = accuracy_list = []
        self.loss_list= loss_list = []
        e = 0; count = 0; self.no_epochs = no_epochs = 0
        while conv == False:
            weights -= eta / N * X_train.T @ (self.forward(X_train) - t_train)
            pred = self.predict(X_train, False)
            accuracy_list.append(np.mean(pred ==t_train))
            loss_list.append(self.bce(pred, t2_train)) 
            if e>0 and abs(loss_list[e]-loss_list[e-1])<tol:
                count +=1
            elif count != 0 and abs(loss_list[e]-loss_list[e-1])>tol:
                count = 0
            if count == num_epochs_no_update:
                conv = True
                no_epochs = e
            e+=1
        
    def bce(self, pred, t2):
        eps = 1e-7   #In order to avoid log(0)
        loss = -np.mean((t2 * np.log(pred+eps) + (1 - t2) * np.log(1 - pred+eps)))
        # print(loss, 'a')
        return loss
    
    def predict(self, X, boolis, threshold=0.5):
        """X <is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        z = X
        if boolis:
            z = add_bias(X, self.bias)
        return (self.forward(z) > threshold).astype('int')
    
    def predict_prob(self,z,boolis):
        if boolis:
            z = add_bias(X, self.bias)
        else:
            z = X
        return self.forward(z)
    
    def forward(self, X):
        return soft_max(X @ self.weights)

def accuracy(predicted, gold):
    return np.mean(predicted == gold)

def logistic(x):
    return 1/(1+np.exp(-x))

def soft_max(x):
    return np.exp(x) / sum(np.exp(x))


###
# Change the t_multi_train a t_binary_train excluding all values
# except the one we are interested in.
###

##Dette funker ikke helt, hehe
# t_multi_train_bin = np.copy(t_multi_train)
# arg_one_vs_rest =                 
# new_arg = (0 if arg_one_vs_rest==1 else 1)
# a = np.where(t_multi_train==1) 
# b = np.where(t_multi_train!=1)
# t_multi_train_bin[b] = new_arg
# t_multi_train_bin[a] = (0 if new_arg != 0 else 1)


ql = NumpyLogisticClass()
ql.fit(X_train, t2_train)
# print(t_multi_train)

def plot_decision_regions(X, t, clf=[], size=(8,6)):
    """Plot the data set (X,t) together with the decision boundary of the classifier clf"""
    # The region of the plane to consider determined by X
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    
    # Make a make of the whole region
    h = 0.02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()], True)
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

plot_decision_regions(X_train, t2_train, ql)

# plot_decision_regions(X_train, t_multi_train, ql)
plt.show()
