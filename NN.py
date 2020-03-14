import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools

def Sigmoid(Z):
    return 1/(1+np.exp(-Z))
def Relu(Z):
    return np.maximum(0,Z)
def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x
def dSigmoid(Z):
    s = 1/(1+np.exp(-Z))
    dZ = s * (1-s)
    return dZ
def tanh(Z):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
def dtanh(Z):
    t = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dZ=1-t**2
    return dZ

class dlnet:
    def __init__(self, x, y, lr=3, d1=68, h=50, d2=1):
        # print(h)
        self.X=x # holds the input layer rows are features, columns are samples
        self.Y=y # desired output to train network
        self.Yh=np.zeros((1,self.Y.shape[1])) # output of the network init at 0
        self.L=2 # number of layers
        self.dims = [d1, h,d2 ] # # of neurons in each layer
        self.param = {} # dict to hold wieght and bias
        self.ch = {} # chace to hold intermediate prameters
        self.grad = {}
        self.loss = [] # loss value of the network
        self.lr= lr # the rate of learning
        # print(self.lr)
        self.sam = self.Y.shape[1] # amount of samples
        self.error_history = []
        self.error_test = []
        self.epoch_list = []
        self.scale = 1
        
    def nInit(self):    
        np.random.seed(1)
        self.param['W1'] = np.random.randn(self.dims[1], self.dims[0]) / np.sqrt(self.dims[0]) # input to hidden layer weight
        self.param['b1'] = np.zeros((self.dims[1], 1))    # bias to first weight    
        self.param['W2'] = np.random.randn(self.dims[2], self.dims[1]) / np.sqrt(self.dims[1]) # hidden layer to output weight
        self.param['b2'] = np.zeros((self.dims[2], 1))  # bias to second weight             
        return
    
    def forward(self):    
            Z1 = self.param['W1'].dot(self.X) + self.param['b1'] 
            A1 = Sigmoid(Z1)
            self.ch['Z1'],self.ch['A1']=Z1,A1

            Z2 = self.param['W2'].dot(A1) + self.param['b2']  
            A2 = Sigmoid(Z2)
            self.ch['Z2'],self.ch['A2']=Z2,A2
            self.Yh=A2
            loss=self.MSE_Loss(A2)
            return self.Yh, loss
    
    def test(self, input, output):
            Z1 = self.param['W1'].dot(input) + self.param['b1'] 
            A1 = Sigmoid(Z1)
            # self.ch['Z1'],self.ch['A1']=Z1,A1

            Z2 = self.param['W2'].dot(A1) + self.param['b2']  
            A2 = Sigmoid(Z2)
            # self.ch['Z2'],self.ch['A2']=Z2,A2
            # self.Yh=A2
            loss = (1./110) * (-np.dot(output,np.log(A2).T) - np.dot(1-output, np.log(1-A2).T))
            return A2, loss
       
    def Final_test(self, input):
            Z1 = self.param['W1'].dot(input) + self.param['b1'] 
            A1 = Sigmoid(Z1)
            # self.ch['Z1'],self.ch['A1']=Z1,A1

            Z2 = self.param['W2'].dot(A1) + self.param['b2']  
            A2 = Sigmoid(Z2)
            # self.ch['Z2'],self.ch['A2']=Z2,A2
            # self.Yh=A2
            # loss = (1./110) * (-np.dot(output,np.log(A2).T) - np.dot(1-output, np.log(1-A2).T))
            return A2
        
    def MSE_Loss(self, Yh):
        squared_errors = (Yh - self.Y) ** 2
        loss = (1./self.sam)*np.sum(squared_errors)
        return loss
        
    def nloss(self,Yh):
        loss = (1./self.sam) * (-np.dot(self.Y,np.log(Yh).T) - np.dot(1-self.Y, np.log(1-Yh).T))    
        return loss
    
    def backward(self):
        dLoss_Yh = - (np.divide(self.Y, self.Yh ) - np.divide(1 - self.Y, 1 - self.Yh))    
        dLoss_Z2 = dLoss_Yh * dSigmoid(self.ch['Z2'])    
        dLoss_A1 = np.dot(self.param["W2"].T,dLoss_Z2)
        dLoss_W2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2,self.ch['A1'].T)
        dLoss_b2 = 1./self.ch['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 
                            
        dLoss_Z1 = dLoss_A1 * dSigmoid(self.ch['Z1'])        
        dLoss_A0 = np.dot(self.param["W1"].T,dLoss_Z1)
        dLoss_W1 = 1./self.X.shape[1] * np.dot(dLoss_Z1,self.X.T)
        dLoss_b1 = 1./self.X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))  
        
        
        self.param["W1"] = self.param["W1"] - self.lr * dLoss_W1
        self.param["b1"] = self.param["b1"] - self.lr * dLoss_b1
        self.param["W2"] = self.param["W2"] - self.lr * dLoss_W2
        self.param["b2"] = self.param["b2"] - self.lr * dLoss_b2

    def gd(self,X, Y, test=[], out=[], iter = 3000):
        np.random.seed(1)                         
    
        self.nInit()
    
        for i in range(0, iter):
            Yh, loss=self.forward()
            self.error_history.append(np.average(loss))
            if len(test) != 0:
                self.error_test.append(np.average(self.test(test, out)[1]))
            self.epoch_list.append(i)
            # print(i)
            # print(loss)
            self.backward()
            if i % 500 == 0:
                print ("Cost after iteration %i: %f" %(i, np.average(loss)))
                # print(loss)
                # print(np.average(loss))
                self.loss.append(loss)
            
    
        return
    

def test_NN(k1,k2,k3,k4, output, lr=3, h=50):
    train1 = np.append(k1, k2, axis =0)
    train1 = np.append(train1, k3, axis =0)
    test1 = k4

    train2 = np.append(k1, k2, axis =0)
    train2 = np.append(train2, k4, axis =0)
    test2 = k3

    train3 = np.append(k1, k3, axis =0)
    train3 = np.append(train3, k4, axis =0)
    test3 = k2

    train4 = np.append(k2, k3, axis =0)
    train4 = np.append(train4, k4, axis =0)
    test4 = k1
    
    k_out = np.tile([1,0],int(len(k1)/2))

    NN1 = dlnet(train1.T, output.T, lr=lr, h=h)
    NN1.gd(train1.T, output.T,test1.T,k_out.reshape(len(k_out),1).T, iter = 3000)
    # plot the error over the entire training duration
    plt.figure(figsize=(15,5))
    plt.plot(NN1.epoch_list, NN1.error_history, color = 'blue')
    plt.plot(NN1.epoch_list, NN1.error_test, color = 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()
    NN2 = dlnet(train2.T, output.T, lr=lr, h=h)
    NN2.gd(train2.T, output.T, test2.T,k_out.reshape(len(k_out),1).T, iter = 3000)
    # plot the error over the entire training duration
    plt.figure(figsize=(15,5))
    plt.plot(NN2.epoch_list, NN2.error_history, color = 'blue')
    plt.plot(NN2.epoch_list, NN2.error_test, color = 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()
    NN3 = dlnet(train3.T, output.T, lr=lr, h=h)
    NN3.gd(train3.T, output.T,test3.T,k_out.reshape(len(k_out),1).T, iter = 3000)
    # plot the error over the entire training duration
    plt.figure(figsize=(15,5))
    plt.plot(NN3.epoch_list, NN3.error_history, color = 'blue')
    plt.plot(NN3.epoch_list, NN3.error_test, color = 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()
    NN4 = dlnet(train4.T, output.T, lr=lr, h=h)
    NN4.gd(train4.T, output.T, test4.T,k_out.reshape(len(k_out),1).T, iter = 3000)
    # plot the error over the entire training duration
    plt.figure(figsize=(15,5))
    plt.plot(NN4.epoch_list, NN4.error_history, color = 'blue')
    plt.plot(NN4.epoch_list, NN4.error_test, color = 'red')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.show()
    

    results = (NN1.test(test1.T, k_out.reshape(len(k_out),1).T)[1]+ NN2.test(test2.T, k_out.reshape(len(k_out),1).T)[1]+NN3.test(test3.T, k_out.reshape(len(k_out),1).T)[1]+NN4.test(test4.T, k_out.reshape(len(k_out),1).T)[1])/4
    return results