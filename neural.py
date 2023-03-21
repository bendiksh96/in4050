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
    #N = X.shape[0]
    #biases = np.ones((N, 1))*bias # Make a N*1 matrix of bias-s
    # Concatenate the column of biases in front of the columns of X.
    return X + bias #np.concatenate((biases, X), axis  = 1) 
np.random.seed(1)

class NumpyClassifier():
    """Common methods to all numpy classifiers --- if any"""
class MLPBinaryLinRegClass(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden = 5):
        """Intialize the hyperparameters"""
        self.bias = bias
        self.dim_hidden = dim_hidden
        
        self.bias1 = np.zeros((5))
        self.bias2 = np.zeros((1))
        
        def logistic(x):
            return 1/(1+np.exp(-x))
        self.activ = logistic
        
        def logistic_diff(y):
            return self.activ(y)*(1-self.activ(y))
        self.activ_diff = logistic_diff
        
    def fit(self, X_train, t_train, eta=0.003, epochs = 100, X_val = 0, T_val = 0):
        """Intialize the weights. Train *epochs* many epochs.
        X_train is a Nxm matrix, N data points, m features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        """
        self.eta = eta
        
        T_train = t_train.reshape(-1,1)
        
        dim_in = X_train.shape[1] 
        dim_out = 1 #T_train.shape[1]
        # print(dim_in, dim_out)b
        
        # Initialize the weights
        self.weights1 = (np.random.rand(dim_in, self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 = (np.random.rand(self.dim_hidden,dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        #X_train_bias = add_bias(X_train, self.bias)
        
        self.acc_list = acc_list = []
        self.loss_list = loss_list = []
        
        self.conv = conv = False
        num_epochs_no_update = 5
        tol = 1e-5
        self.no_epochs = no_epochs = 0
        e = 0; count = 0
        while conv == False:
            # One epoch
            #hidden_outs, outputs = self.forward(X_train_bias)
            z1, z2, a1, a2 = self.forward(X_train)
            outputs = a2
            
            print(outputs)
     
            error = self.bce(outputs, t2_train)
            
            print("Loss: ", error)
            
            # print(outputs.shape)
            # print(error.shape)

            # print(self.bce_derivative(outputs, t2_train).shape)
            # print(self.activ_diff(z2).shape)
            
            # print(self.bce_derivative(outputs, t2_train))
            # print(self.activ_diff(z2).flatten())
           
            
            weights2_error = np.multiply(self.bce_derivative(outputs, t2_train), self.activ_diff(z2).flatten())
            weights2_diff = a1.T@weights2_error
            
            # print(weights2_error)
            # print(weights2_diff)
            # print(np.mean(weights2_error))
            # print(np.mean(a1))
            
            weights1_error = np.expand_dims(weights2_error, 1) @ self.weights2.T * self.activ_diff(z1)
            weights1_diff = X_train.T@weights1_error
            
            # print(a1.shape)
            # print(weights1_error.shape)
            # print(self.weights1.shape)
            # print(weights1_diff.shape)
            
            # input()
            # print(weights1_diff)
            # print(weights2_diff)
            # input()

            
            self.weights2 -= self.eta * np.expand_dims(weights2_diff, 1)
            self.weights1 -= self.eta * weights1_diff
            
            # print(self.weights1)
            # print(self.weights2)
        
            
            self.bias1 -= self.eta * np.sum(weights1_error)
            self.bias2 -= self.eta * np.sum(weights2_error)
            
            
            
            
            # The delta term on the output node
            # Vekter deltaen med en vekt 2
            #hiddenout_diffs = out_deltas @ self.weights2.T
            
            # The delta terms at the output of the hidden layer
            #hiddenact_deltas = (hiddenout_diffs[:, 1:] *  self.activ_diff(hidden_outs[:, 1:]))  
            
            # The deltas at the input to the hidden layer
            #self.weights2 -= self.eta * hidden_outs.T @ out_deltas
            #self.weights1 -= self.eta * X_train_bias.T @ hiddenact_deltas 
            
            # Update the weights
            pred = self.predict(outputs)
            
            #print(t2_train.shape)
            #outputs = outputs.flatten()
            #print(outputs.shape)
            #print(pred.shape)
            #exit()
            
            # print(t2_train)
            # print()
            #print(outputs)
            #exit()
            
            acc_list.append(np.mean(pred == t_train))
            loss_list.append(error)
            
            print("Accuracy: ", np.mean(pred == t_train))
            input()
            
            # print(acc_list[e])
            ##Loss blir negativ. Dette er et problem! 
            if e>0 and abs(loss_list[e]-loss_list[e-1])<tol:
                count +=1
            elif count != 0 and abs(loss_list[e]-loss_list[e-1])>tol:
                count = 0
            if count == num_epochs_no_update:
                conv = True
                no_epochs = e
            e+=1
            # print(e, loss_list[e-1])
            
    def forward(self, X):
        """Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        z1 = X @ self.weights1 
        z1 = add_bias(z1, self.bias1)
        a1 = self.activ(z1)
        
        z2 = a1 @ self.weights2
        z2 = add_bias(z2, self.bias2)
        a2 = self.activ(z2)
        
        
        # hidden_activations = self.activ(X @ self.weights1)
        # hidden_outs = add_bias(hidden_activations, self.bias)
        # outputs = hidden_outs @ self.weights2
        # outputs = self.activ(outputs)   #Lagt til av meg... 
 
        return z1, z2, a1, a2 #hidden_outs, outputs
    
    
    def predict (self, outputs):
        return (np.round(outputs))
        
    # def predict(self, X):
    #     """Predict the class for the mebers of X"""
    #     if np.size(X)!=3000:
    #         Z = add_bias(X, self.bias)
    #     else: 
    #         Z = X
    #     forw = self.forward(Z)[1]
    #     score= forw[:, 0]
    #     return (score > 0.5)
    
    def bce_derivative(self, pred, true):
        diff = (pred.flatten() - true)/((pred.flatten())*(1 - pred.flatten()+1e-7))
        return diff
    
    def bce(self, pred, true):
        eps = 1e-7   #In order to avoid log(0)
        loss = 0
        #print(pred)
        
        for i in range(1000):
            #print(pred[i])                                                                                    
            #print(true[i])
            #print(np.log(pred[i]+eps))
            
            loss -= true[i] * np.log(pred[i]+eps) + (1-true[i])*np.log(1-pred[i])
            #print(loss)
        
        loss /= 1000
        #loss = ((-true * np.log(pred+eps) + (1 - true) * np.log(1 - pred+eps)))
        #print(loss)
        return loss
    
ql = MLPBinaryLinRegClass()
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

#plot_decision_regions(X_train, t2_train, cl)
plot_decision_regions(X_train, t2_train, ql)
plt.show()
