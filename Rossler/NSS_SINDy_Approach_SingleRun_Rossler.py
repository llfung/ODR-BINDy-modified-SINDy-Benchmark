# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 14:26:01 2020

@author: kahdi
"""


#%% Import packages
import sys
Param_i = sys.argv[1]
Run_i = sys.argv[2]

DataLength = float((int(Param_i)-1)%11)*2.0+10.0
NoisePercentage = (float((int(Param_i)-1)//11)+1.0)*2.5
import numpy as np
from scipy.integrate import odeint
import tensorflow as tf
# import matplotlib.pyplot as plt
from utils_NSS_SINDy import *
import time
# import tensorflow_probability as tfp
from datetime import datetime
#%% Create a path and folder to save the result
FolderName="Result/"
FilePath=os.getcwd()
SavePath=os.path.join(FilePath,FolderName)
print(SavePath)
# Create the folder
try:
    os.mkdir(SavePath)
    print("The file folder does not exist, will create a new one....\n")
except:
    print("The folder already exist, will store the new result in current folder...\n")

#%% Simulate
# Define the random seed for the noise generation
np.random.seed(nt(Param_i)*1000+int(Run_i))

# Define the parameters
p0=np.array([0.2,0.2,5.7])

# Define the initial conditions
x0=np.array([3.0,5.0,0.0])

# Define the time points
T=DataLength
dt=0.05

t=np.linspace(0.0,T,int(T/dt))

# Now simulate the system
x=odeint(Rossler,x0,t,args=(p0,),rtol = 1e-12, atol = 1e-12)
dx=np.transpose(Rossler(np.transpose(x), 0, p0))

# Get the size info
stateVar,dataLen=np.transpose(x).shape

# Generate the noise
NoiseMag=[np.std(x[:,i])*NoisePercentage*0.01 for i in range(stateVar)]
Noise=np.hstack([NoiseMag[i]*np.random.randn(dataLen,1) for i in range(stateVar)])

# Add the noise and get the noisy data
xn=x+Noise
  
# Test the SINDy
N_SINDy_Iter=15
disp=0
NormalizeLib=0
libOrder=2
lam=0.05

# SoftStart?
softstart=0
#%% Define a neural network
# Check the GPU status
CheckGPU()

# Define the data type
dataType=tf.dtypes.float32

dh=tf.constant(dt)

#%% Define the data
# Define the prediction step
q=1

# Get the middel part of the measurement data (it will be define as constant)
Y=tf.constant(xn,dtype=dataType)
Y0=tf.constant(GetInitialCondition(xn,q,dataLen),dtype=dataType)

# Ge the forward and backward measurement data (it is a constant that wouldn't change)
Ypre_F,Ypre_B=SliceData(xn,q,dataLen)
Ypre_F=tf.constant(Ypre_F,dtype=dataType)
Ypre_B=tf.constant(Ypre_B,dtype=dataType)

# Get the weight for the error
ro=0.9
weights=tf.constant(DecayFactor(ro,stateVar,q),dtype=dataType)

if softstart==1:
    # Soft Start
    NoiseEs,xes=approximate_noise(np.transpose(xn), 20)
    NoiseEs=np.transpose(NoiseEs)
    xes=np.transpose(xes)
else:
    # Hard Start
    NoiseEs=np.zeros((xn.shape[0],xn.shape[1]))
    xes=xn-NoiseEs

dxes=CalDerivative(xes,dt,1)

# Get the initial guess of noise, we make a random guess here. For beteer performance you could first smooth the data and guess the noise. 
NoiseVar=tf.Variable(NoiseEs,dtype=tf.dtypes.float32)

# Get the initial guess of the SINDy parameters
Theta=Lib(xes,libOrder)

Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)

#%%
# Define the initial guess of the selection parameters
Xi=tf.Variable(Xi0,dtype=dataType)

# Set the initial active matrix
Xi_act=tf.constant(np.ones(Xi0.shape),dtype=dataType)

#%% Define the optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001,epsilon=1e-09)

#%% Finally start training!
Nloop=8
N_train=15000

# Set a list to store the noise value
NoiseList=[]
NoiseIDList_SingleRun=[]
TrainTimeList_SingleRun=np.zeros((Nloop,1))
Enoise_error_List_SingleRun=np.zeros((Nloop,1))
Evector_field_error_list_SingleRun=np.zeros((Nloop,1))
Epre_error_list_SingleRun=np.zeros((Nloop,1))
x_sim_list_SingleRun=[]
Xi_List_SingleRun=[]

for k in range(Nloop):
    print("Runing the loop ",str(k+1))
    # Denoise the signal
    NoiseID,totalTime=Train_NSS_SINDy(Y,Y0,Ypre_F,Ypre_B,NoiseVar,Xi,Xi_act,weights,dt,q,stateVar,dataLen,optimizer,N_train)
    
    print("\t Current loop takes ",totalTime)
    # After the first iteration, minus the noise identified from the noisy measurement data
    xes=xn-NoiseID
    xes=xes[q+1:-q-1,:]
    dxes=CalDerivative(xes,dt,1)
         
    print("Current Xi result")
    print(Xi)
    
    # Do SINDy on the denoised data
    Theta=Lib(xes,libOrder)
    Xi0=SINDy(Theta,dxes,lam,N_SINDy_Iter,disp,NormalizeLib)
    
    # Do SINDy on the denoised data
    index_min=abs(Xi.numpy())>lam
    Xi_act_dum=Xi_act.numpy()*index_min.astype(int)
    Xi_num=Xi.numpy()
    Xi_num=Xi_num*Xi_act_dum
    index_min=Xi_act_dum.astype(bool)
    
    # Regress
    for r in range(xes.shape[1]):
        Xi_num[index_min[:,r],r]=solve_minnonzero(Theta[:,index_min[:,r]],dxes[:,r])
    
    # Print the new initial start point
    print("New Xi result")
    print(Xi_num)
    
    # Determine which term should we focus on to optimize next
    Xi_act=tf.constant(Xi_act_dum,dtype=tf.dtypes.float32)
    Xi=tf.Variable(Xi_num,dtype=tf.dtypes.float32)
    
    # Calculate the performance
    Enoise_error,Evector_field_error,Epre_error,x_sim=ID_Accuracy_SINDy(x,dx,Noise,NoiseID,LibGPU,Xi,dataLen,dt)
    
    # Print the performance
    print("\t\t The error between the true noise and estimated noise is:",Enoise_error,"\n")
    print("\t\t The error between the true vector field and estimated vector field is:",Evector_field_error,"\n")
    print("\t\t The error between the true trajector and simulted trajectory is:",Epre_error,"\n")
    
    NoiseIDList_SingleRun.append(NoiseID)
    x_sim_list_SingleRun.append(x_sim)
    TrainTimeList_SingleRun[k]=totalTime
    Enoise_error_List_SingleRun[k]=Enoise_error
    Evector_field_error_list_SingleRun[k]=Evector_field_error
    Epre_error_list_SingleRun[k]=Epre_error
    Xi_List_SingleRun.append(Xi.numpy())
    
#% % Save the result
import scipy.io as iomat
results = {'TrainTimeList':TrainTimeList_SingleRun,
              'Enoise_error_List':Enoise_error_List_SingleRun,'Evector_field_error_list':Evector_field_error_list_SingleRun,
              'Epre_error_list':Epre_error_list_SingleRun,'Xi_List':Xi_List_SingleRun,
              'Xi0_List':Xi0}
iomat.savemat(SavePath+Param_i+"_"+Run_i+"_NSS_SINDy_Rossler.mat",
             results)
