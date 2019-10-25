#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 17:29:02 2019

@author: caoqun
"""
import torch
from torch.autograd import Variable as Var
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F 

class GD:
#-------------------variables---------------------------
    def __init__(self, N, power, sigma):
        self.Ein_all = []
        self.Eout_all = []
        self.theta_all = []
        
        self.test_N=1000 
        self.x = []  
        self.y = []
        self.z = []
        
        self.sigma = sigma
        self.N = N
        self.power = power
        
        self.x_hat = []
        self.y_hat = []
        
        self.theta = []
        self.Loss = torch.zeros(0)  #least square error
        self.lr = 0   #learning rate
        self.Ein = 0  #training error
        self.Eout = 0  #test error
        self.N_temp =0
        self.Loss_old = 0
        self.theta_av = []   #final coefficient list
        self.Ein_av = 0     #average training error in 50 trials
        self.Eout_av = 0    #average test error in 50 trials
        self.delta_Loss = 0
        self.Ebias = 0
        self.theta_all_hat = torch.zeros(0)
        self.test = 0
        
#-------------------get dataset---------------------------
    def getData(self):
        self.x = np.random.uniform(0,1,self.N) #get uniform random x vector
#        print(self.x, self.test)
        self.z = np.random.normal(0,self.sigma,self.N) #get normal random z vector
        self.y = np.cos(2*np.pi*self.x) + self.z  #get y vector
        self.x = self.x.reshape(self.N,1)  #convert x into a vertical vector
        self.x = self.x.repeat(self.power+1,axis=1) 
        items = np.array(range(self.power+1))
        self.x = np.power(self.x,items) #convert x to a Nx(power+1) matrix 
        self.x = torch.Tensor(self.x)  #conver x from array to tensor
        self.y = torch.Tensor(self.y)  #same as x
#        print(x,y)
        if self.test == 0:
            self.theta = np.random.randn(self.power+1)
            self.theta = torch.Tensor(self.theta)
            self.theta = Var(self.theta,requires_grad = True)
 #--------------------get dataset---------------------------
    #-------------------Loss function-------------------------------
    def getMSE(self):   
        self.y_hat = self.x * self.theta     #y_hat is result of training model
        self.y_hat = torch.sum(self.y_hat,dim=1)
        self.Loss = torch.mean(torch.pow(self.y-self.y_hat,2)) #calculate least square error 
    #---------------------------GD----------------------------------
    
    def fitData(self):  #GD method for updating model parameters
        self.getMSE()
        self.Loss.backward(retain_graph=True, create_graph=True) #calculate the gradient
#        self.theta_all.append(self.theta.data)kkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk
#        if self.theta.grad is not None:
        self.theta = self.theta.data - self.lr*self.theta.grad.data 
        self.theta = Var(self.theta, requires_grad=True) #convert theta type from tensor to variable
#        print(self.Loss)
    #---------------------------Error-----------------------------------
    def ein(self):          #calculate training loss function
        self.Ein = self.Loss
#        print('Ein=', self.Ein)
     
    def eout(self):          #calculate test loss function
        self.test = 1
        self.N_temp = self.N
        self.N = 1000
        self.getData() #get 1000 test dataset
        self.getMSE()
        self.Eout = self.Loss
#        self.y_hat = self.x * self.theta     
#        self.y_hat = torch.sum(self.y_hat,dim=1)
#        self.Eout = torch.mean(torch.pow(self.y-self.y_hat,2))        
#        self.Eout = self.Loss #calculate test dataset MSE
#        print('Eout=',self.Eout) 
        self.N = self.N_temp
        
    def experiment(self):
        for k in range (50): ##we get all the Ein and Eout in 50 trials
    #-------------------parameters-----------------------------
            self.test = 0
            self.lr=0.01 #learning rate
            self.getData()
#            print(k,"\t")
#            print(len(self.x), len(self.y_hat))
#            self.getMSE()
#            a = 0
    #--------------------------circle--------------------------------- 
            self.Loss_old = 0
            for i in range(2000): #GD circulation
                self.fitData()
                self.delta_Loss = self.Loss - self.Loss_old 
                self.Loss_old = self.Loss
                if abs(self.delta_Loss) < 0.0001:
                    break
            #we get theta and loss after optimization
            self.ein()
            print(self.Loss, "train")
            self.eout()
            print(self.Loss,"test")
#            print(len(self.x), len(self.y_hat))
#            for i in range(len(self.x[i])):
#                if self.y_k[i] == self.y[i]:
#                    a+=1
#            print(a)
            self.Ein_all.append(self.Ein)  #generate Ein vector
            self.Eout_all.append(self.Eout)   #generate Eout vector
    
    def ebias(self): 
        self.N_temp = self.N
        self.N = 1000
        self.getData() #get 1000 test dataset
        self.getMSE()
        self.Ebias = self.Loss
#        print('Ebias=',self.Ebias)
        
    
    def avg(self):
        self.Ein_av = np.sum(self.Ein_all)/len(self.Ein_all)
        self.Eout_av = np.sum(self.Eout_all)/len(self.Eout_all)
#        print('Ein_av=',self.Ein_av)
#        print('Eout_av=',self.Eout_av)
#        self.theta_all_hat = torch.stack(self.theta_all,1)
#        self.theta_av = torch.mean(self.theta_all_hat,1) 
#        print('theta_av=',self.theta_av) 
#        self.ebias()

#N = 200
#power = 4
#sigma = 0.01
#a = GD(N,power,sigma)
#a.experiment()
#a.avg()

#
#Ein_av_all = []
#Eout_av_all = []
x = [2,5,10,20,50,100,200]
#x = range(21)
sigma = 0.01
#N = 200
#power = 18
Ein_av_all = []
Eout_av_all = []
##for sigma in [0.01, 0.1, 1]:
for N in [2,5,10,20,50,100,200]:
#for power in range(21):
    a = GD(N,power,sigma)
    a.getData()
    a.experiment()
    a.avg()
    Ein_av_all.append(a.Ein_av)
    Eout_av_all.append(a.Eout_av)
plt.plot(x, Ein_av_all, 'r--', x, Eout_av_all, 'b--')
plt.xlabel('dimension')
plt.ylabel('Loss')
plt.show()
