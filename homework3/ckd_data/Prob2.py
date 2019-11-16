from __future__ import print_function
import numpy as np
from sklearn.metrics import f1_score
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

class RegularizedLogisticRegression:

  def __init__(self,learning_rate=0.001,reg_factor=0.01,rand_seed=None):
    self.learning_rate = learning_rate
    self.reg_factor = reg_factor
    self.rand_seed = None
    self.weights = None
    if rand_seed:
      self.rand_seed = rand_seed

  def fit(self,X,y,stopping_theta = 10e-17,max_iter = 10000,verbose=False):
    if y.shape[0] != 1:
      y = y.T
    self.X_intercept = self.add_intercept(X)
    self.weights = self.init_weights()
    self.verbose = verbose
    self.weights,self.cost_value = self.train(X,y,self.weights,self.learning_rate,stopping_theta,max_iter)
    return self.cost_value

  def init_weights(self,X=None):
    if self.rand_seed:
      np.random.seed(int(self.rand_seed))
    if X is not None:
      X_intercept = self.add_intercept(X)
    else:
      X_intercept = self.X_intercept
    weights = np.random.random(size=(1,X_intercept.shape[1]))
    return weights
      

  def sigmoid(self,x):
    # Activation function used to map any real value between 0 and 1
    return 1.0 / (1 + np.exp(-x))

  def add_intercept(self,X):
    _intercept = np.ones([X.shape[0],1])
    X_intercept = np.concatenate([_intercept,X],axis=1) * 1.0
    return X_intercept

  def predict(self,X,weights=None,return_labels=False):
    if weights is None:
      weights = self.weights
    X_intercept = self.add_intercept(X)
    dot_product = np.dot(X_intercept,weights.T)
    # print('predict dotprod=',dot_product)
    sig = self.sigmoid(dot_product)
    if not return_labels:
      return sig
    results = np.zeros(sig.shape)
    results[sig>0.5] = 1
    return results

  def safe_log(self,value):
    result = np.log(value)
    result[result==-1*np.inf] = -20
    result[result==np.inf] = 20
    return result
    
    

  def cost(self,X,y,weights=None):
    if y.shape[0] != 1:
      y = y.T
    if weights is None:
      weights = self.weights
      
    regularization_factor = self.reg_factor
    h_x = self.predict(X,weights)
    # print('h_x=',h_x)
    small_value = 1e-16
    reg_vals = (np.dot(weights,weights.T))[0,0] * regularization_factor
    h_x[h_x<small_value] = small_value
#    cost_matrix = 1 * y.T * np.log(h_x) + (1 - y.T) * np.log(1 - h_x)
    cost_matrix = 1 * y.T * self.safe_log(h_x) + (1 - y.T) * self.safe_log(1 - h_x)
    
    j = -1 * np.mean(cost_matrix + reg_vals) * 0.5
    # print("cost=",j)
    return j

  def gradient(self,X,y,weights):
    h_x = self.predict(X,weights)
    X_ = self.add_intercept(X)
    num_records = X_.shape[0]
    regularization_factor = self.reg_factor
    reg = regularization_factor * weights
    # print("shape X = {}, y={}, reg={}".format(X_.shape,y.shape,reg.shape))
    gradient_values = 1.0/num_records * (np.dot(X_.T,(h_x - y.T)) + reg.T)
    return gradient_values

  def update(self,X,y,weights,learning_rate=0.01):
    X,y = X*1.0,y*1.0
    last_cost = self.cost(X,y,weights)
    grad = self.gradient(X,y,weights)
    weights = weights - learning_rate * grad.T
    new_cost = self.cost(X,y,weights)
    return weights,last_cost,new_cost

  def train(self,X,y,weights,learning_rate=0.01,stopping_theta = 10e-17,max_iter = 10):
    weights,last_cost,new_cost = self.update(X,y,weights,learning_rate)
    for iter in range(max_iter):
      diff = last_cost - new_cost
      if(new_cost <= stopping_theta):
        if self.verbose:
          print("ends at iter {} loss={}".format(iter,new_cost))
          print("new cost <= {}".format(stopping_theta))
        return weights,new_cost
      weights,last_cost,new_cost = self.update(X,y,weights,learning_rate)
      if self.verbose:
        if(iter % 500 == 0):
          print("loss at iter {} loss = {} diff = {}".format(iter,new_cost,diff))
    if self.verbose:
      print("reach max iter")
    return weights,new_cost
  




if __name__ == "__main__":
    from proprocess_data import get_ckd_dataset
    X_train, X_test, y_train, y_test = get_ckd_dataset(normalized=False)
    print("Unnormalized CDK data loaded with {} train records and {} test records".format(X_train.shape[0],X_test.shape[0]))
    
    reg_param_list = np.array(range(-20,42,2)) / 10.0
    f1_scores_unnormalized = []
    for reg_param in reg_param_list:
        rlr = RegularizedLogisticRegression(rand_seed=0,reg_factor=reg_param,learning_rate=0.1)
        #since the data is not normalized. The sigmoid function goes to 0 or 1 in its 1st or 2nd iteration.
        #which means the gradient go to 0 immediately. 
        rlr.fit(X_train,y_train,verbose=False,max_iter=10000) 
        pred = rlr.predict(X_test,return_labels=True)
        score = f1_score(y_test,pred)
        f1_scores_unnormalized.append(score)
        print("reg param = {} score = {}".format(reg_param,score))  
    
    print('-' * 30)
    X_train, X_test, y_train, y_test = get_ckd_dataset(normalized=True)
    print("Normalized CDK data loaded with {} train records and {} test records".format(X_train.shape[0],X_test.shape[0]))
    f1_scores_normalized = []
    for reg_param in reg_param_list:
        rlr = RegularizedLogisticRegression(rand_seed=0,reg_factor=reg_param,learning_rate=0.1)
        rlr.fit(X_train,y_train,verbose=False,max_iter=10000)
        pred = rlr.predict(X_test,return_labels=True)
        score = f1_score(y_test,pred)
        f1_scores_normalized.append(score)
        print("reg param = {} score = {}".format(reg_param,score))
    
    plt.plot(reg_param_list,f1_scores_unnormalized,'-o',label="without Standardization")
    plt.plot(reg_param_list,f1_scores_normalized,'-o', label="with Standardization")
    plt.title("F1 Score Comparison")
    plt.grid()
    plt.legend()
    plt.xlabel("Regularization Paramter")
    plt.ylabel("F1 value")
    plt.draw()
        
    