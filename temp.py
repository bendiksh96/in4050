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
    def fit(self, X_train, t_train, eta = 0.070, epochs=200, X_val = 0, t_val= 0):
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
            print(accuracy_list[e])
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
        eps = 1e-7          #In order to avoid log(0)
        loss = ((-t2 * np.log(pred+eps) + (1 - t2) * np.log(1 - pred+eps))).mean()
        # print(loss, 'a')
        return loss
    
    def predict(self, X, boolis, threshold=0.5):
        """X <is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        z = X
        if boolis:
            z = add_bias(X, self.bias)
        return (self.forward(z) > threshold).astype('int')
    
    def predict_prob(self):
        # z = add_bias()
        return 0
    
    def forward(self, X):
        return logistic(X @ self.weights)


class NumpyLinRegClass(NumpyClassifier):
    def __init__(self, bias=-1):
        self.bias=bias
        
    #0.07//200 er bra
    def fit(self, X_train, t_train, eta = 0.070, epochs=200):
        """X_train is a Nxm matrix, N data points, m features
        t_train is a vector of length N,
        the targets values for the training data"""
        self.epochs = epochs
        if self.bias:
            X_train = add_bias(X_train, self.bias)
            
        (N, m) = X_train.shape
        
        self.weights = weights = np.zeros(m)
        self.accuracy_list = accuracy_list = np.zeros(epochs)
        self.loss_list= loss_list = np.zeros(epochs)
        for e in range(epochs):
            weights -= eta / N *  X_train.T @ (X_train @ weights - t_train)              
            var = self.predict(X_train, False)
            accuracy_list[e] = np.mean(var ==t_train)
            loss_list[e] = loss_func(var, t_train) 

    def show_acc_loss(self):
        epochs = self.epochs; accuracy_list = self.accuracy_list; loss_list = self.loss_list
        # Metode for å plotte tap og accuracy som funksjon av epoker
        x = np.linspace(0, epochs-1, epochs, dtype='int')
        plt.plot(x, accuracy_list[x], 'b-', label='Accuracy as a function of epochs')
        plt.plot(x, loss_list[x], 'r-', label='Loss as a funciton of epochs')
        plt.legend(); plt.grid(); plt.ylim(0,1)
        plt.show()

    def predict(self, X, boolis, threshold=0.5):
        """X <is a Kxm matrix for some K>=1
        predict the value for each point in X"""
        if boolis:
            X = add_bias(X, self.bias)
        ys = X @ self.weights
        return ys > threshold
    
def accuracy(predicted, gold):
    return np.mean(predicted == gold)

def logistic(x):
    return 1/(1+np.exp(-x))

def loss_func(x,y):
    return np.mean((x - y)**2)

#Standard Scaler
# mean_ = np.mean(X_train[:][0])
# std_  = np.std(X_train[:][0])
# X_train_scale = np.zeros((len(X_train),1))
# for i in range(len(X_train)):
#     X_train_scale[i] = (X_train[:][i][0]- mean_)/std_
#     X_train[i] = X_train[i]/X_train_scale[i]
#     # print(X_train[i])

#Max-min scaler
# max_ = max(X_train[:][1])
# print(X_train[:][1][0])


#Linear Regression
cl = NumpyLinRegClass()
#Metoden fit
cl.fit(X_train, t2_train)
#print(accuracy(cl.predict(X_val, True), t2_val))
# cl.show_acc_loss()

ql = NumpyLogisticClass()
ql.fit(X_train, t2_train)

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

#plot_decision_regions(X_train, t2_train, cl)

plot_decision_regions(X_train, t2_train, ql)
plt.show()
