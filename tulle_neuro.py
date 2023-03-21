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
np.random.seed(2)
class MLPBinaryLinRegClass(NumpyClassifier):
    """A multi-layer neural network with one hidden layer"""
    
    def __init__(self, bias=-1, dim_hidden =3):
        """Intialize the hyperparameters"""
        self.bias = bias
        self.dim_hidden = dim_hidden    
        
        def logistic(x):
            return 1/(1+np.exp(-x))
        self.activ = logistic
         
        def logistic_diff(x):#y):
            return 1/(1+np.exp(-x))*(1-1/(1+np.exp(-x)))#y * (1 - y)
        self.activ_diff = logistic_diff
                                    #Ikke under 0.00029
    def fit(self, X_train, t_train, eta=0.0014, epochs = 10000, X_val = 0, t_val = 0):
        """Intialize the weights. Train *epochs* many epochs.
        
        X_train is a Nxm matrix, N data points, m features
        t_train is a vector of length N of targets values for the training data, 
        where the values are 0 or 1.
        """
        self.eta = eta
        
        T_train = t_train.reshape(-1,1)
            
        dim_in = X_train.shape[1] 
        dim_out = T_train.shape[1]
        
        # Itilaize the wights
        self.weights1 =(np.random.rand(dim_in + 1, self.dim_hidden) * 2 - 1)/np.sqrt(dim_in)
        self.weights2 =(np.random.rand(self.dim_hidden+1, dim_out) * 2 - 1)/np.sqrt(self.dim_hidden)
        # print("min1",min(self.weights1))
        # print("max1",max(self.weights1))
        # print("min2",min(self.weights2))
        # print("max2",max(self.weights2))
        # print(self.weights2.shape)
        # exit()
        X_train_bias = add_bias(X_train, self.bias)
        
        #Konvergens trengs for loss, om det ikke forbedres, må loopen avsluttes. 
        num_epochs_no_update = 5
        tol = 1e-5
        conv =  False
                    
        (N, m) = X_train.shape        
        self.accuracy_list = accuracy_list = []
        self.loss_list= loss_list = []
        e = 0; count = 0; 
        while conv == False:
            hidden_outs, outputs = self.forward(X_train_bias)
            # The forward step
            loss = self.bce(self.predict_probability(X_train), t2_train)
            out_deltas = (outputs - T_train) #np.matrix(1); out_deltas[:] = loss
            # The delta term on the output node
            hiddenout_diffs = out_deltas @ self.weights2.T
            # The delta terms at the output of the jidden layer
            hiddenact_deltas = (hiddenout_diffs[:, 1:] * 
                                self.activ_diff(hidden_outs[:, 1:]))  
            # The deltas at the input to the hidden layer
            self.weights2 -= self.eta * hidden_outs.T @ out_deltas
            self.weights1 -= self.eta * X_train_bias.T @ hiddenact_deltas 
            
            accuracy_list.append(np.mean(self.predict(X_train) ==t_train))
            loss_list.append(loss) 
            if e>0 and abs(loss_list[e]-loss_list[e-1])<tol:
                count +=1
            elif count != 0 and abs(loss_list[e]-loss_list[e-1])>tol:
                count = 0
            if count == num_epochs_no_update:
                conv = True
                self.no_epochs = e
            if e>int(epochs/5) and loss_list[e] >= loss_list[e-100]:
                conv = True
                self.no_epochs = e
                print('loopy')
            # print("loss:", loss)
            # print()
            # print()
            e+=1           
            # if e>10:
            #     conv = True
                
            
    def forward(self, X):
        """Perform one forward step. 
        Return a pair consisting of the outputs of the hidden_layer
        and the outputs on the final layer"""
        hidden_activations = self.activ(X @ self.weights1)
        hidden_outs = add_bias(hidden_activations, self.bias)
        outputs = hidden_outs @ self.weights2
        return hidden_outs, outputs
    
    def predict(self, X):
        """Predict the class for the members of X"""
        Z=add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return (score > 0.5)
    
    def bce(self, pred, t2):
        # print("min:", min(pred))
        # print("max:", max(pred))
        # print(-np.mean(t2*np.log(pred) + (1-t2)*np.log(1-pred)))
        
        eps = 1e-4   #In order to avoid log(0)
        #Cheep måte å hindre negativ log
        pred = np.clip(pred, eps, 1-eps)
        
        loss = -np.mean((t2 * np.log(pred+eps) + (1 - t2) * np.log(1 - pred+eps)))
        return loss

    def predict_probability(self, X):
        Z = add_bias(X, self.bias)
        forw = self.forward(Z)[1]
        score= forw[:, 0]
        return score
            
    def plot_loss(self):
        ep_vec = np.linspace(0, self.no_epochs, self.no_epochs+1)
        plt.plot(ep_vec, self.loss_list)
        plt.show()
    
    def plot_acc(self):
        ep_vec = np.linspace(0, self.no_epochs, self.no_epochs+1)
        plt.plot(ep_vec, self.accuracy_list)
        plt.show()        
        

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
# cl.plot_loss()
# cl.plot_acc()
plot_decision_regions(X_train, t2_train, cl)