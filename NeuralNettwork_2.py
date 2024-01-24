#models:

    
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import tree, datasets, preprocessing
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
#from skopt import BayesSearchCV
#from skopt.space import Real, Categorical, Integer


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense           
from keras import optimizers             
from keras import regularizers           


import PreprosessingAstroids as pp




#Main code
np.random.seed(0)

#preprosessing data
X_train_scaled, X_test_scaled, y_train, y_test = pp.get_haz_data('select', 'RUS')

#Parameters to tune
print("Set parameters")
lmbda = np.logspace(-5, -1, 5)
learning_rate = np.logspace(-5, -1, 9)
#neurons = np.linspace(2, 100, 20)
#n_layers = np.linspace(1,20,10)
#batch_size = np.linspace(200,1000,10)
#epochs = np.linspace(50, 500,10)

Lfun = 'categorical_crossentropy'
met = 'accuracy'
neurons_out = 1
act_out = 'sigmoid'



activations = ['relu', 'sigmoid', 'softplus', 'softsign', 'tanh', 'selu',
               'elu', 'exponential', 'LeakyReLU','relu']
act = activations[0]
act = 'relu'

neurons = 5
n_layers = 1
batch_size = 100
epochs = 500

def NN_model(inputsize,n_layers,neurons,learning_rate,lmbda):
    model=Sequential() # initialize model  
    #Run loop to add hidden layers to the model
    for i in range(n_layers):
        if (i==0):  
            #First layer                
            model.add(Dense(neurons, activation = act,
                            kernel_regularizer=regularizers.l2(lmbda),
                            input_dim = inputsize))
        else:
            #Hidden layers
            model.add(Dense(neurons, activation = act,
                            kernel_regularizer=regularizers.l2(lmbda)))
    # Output layer
    model.add(Dense(neurons_out,activation=act_out))

    model.compile(loss = Lfun, optimizer = opt, metrics=[met])
    return model


#Define matrices to store accuracy scores as a function
#of learning rate and number of hidden neurons
Train_accuracy=np.zeros((len(lmbda),len(learning_rate)))
Test_accuracy=np.zeros((len(lmbda),len(learning_rate))) 

for i in range(len(lmbda)): 
    for j in range(len(learning_rate)):
        
        optimator = {'Adam':optimizers.Adam(lr=learning_rate[j]), 
                'SGD':optimizers.SGD(lr=learning_rate[j]),
                'RMSprop':optimizers.RMSprop(lr=learning_rate[j]), 
                'Adadelta':optimizers.Adadelta(lr=learning_rate[j]),
                'Adagrad':optimizers.Adagrad(lr=learning_rate[j]), 
                'Adamax':optimizers.Adamax(lr=learning_rate[j]),
                'Nadam':optimizers.Nadam(lr=learning_rate[j]), 
                'Ftrl':optimizers.Ftrl(lr=learning_rate[j])}
        opt = optimator['Adam']
        opt = optimizers.Adam(lr=learning_rate[j])
        print("Lambda: ",lmbda[i], " Eta: ",learning_rate[j])
        print("Create NN")
        DNN_1 = NN_model(X_train_scaled.shape[1], n_layers, neurons, 
                             learning_rate[j], lmbda[i])
        print("Train NN")
        DNN_1.fit(X_train_scaled,y_train, epochs = epochs, 
                      batch_size = batch_size,verbose=1)

        Train_accuracy[i,j] = DNN_1.evaluate(X_train_scaled,y_train)[1]
        Test_accuracy[i,j] = DNN_1.evaluate(X_test_scaled,y_test)[1]

index = np.maxarg(Test_accuracy)


def plotGrid(data, x_ax, y_ax, T1):
    sns.set()
    fig, ax1 = plt.subplots(figsize = (10, 10))
    sns.heatmap(data, annot=True, ax=ax1, cmap="viridis",  fmt=".2%")
    ax1.set_title(T1)
    ax1.set_xlabel(x_ax)
    ax1.set_ylabel(y_ax)
    return

plotGrid(Test_accuracy, learning_rate,  lmbda, "tittel")
    
"""
Train_accuracy=np.zeros((len(epochs),len(batch_size)))
Test_accuracy=np.zeros((len(epochs),len(batch_size))) 

for m in range(len(epochs)):
    for n in range(len(batch_size)):

        DNN_2 = NN_model(X_train_scaled.shape[1], n_layers, neurons, 
                             learning_rate, lmbda)

        DNN_2.fit(X_train_scaled,y_train, epochs = epochs[m], 
                      batch_size = batch_size[n],verbose=1)
                
        Train_accuracy[m,n] = DNN_2.evaluate(X_train_scaled,y_train)[1]
        Test_accuracy[m,n] = DNN_2.evaluate(X_test_scaled,y_test)[1]
        
        
        
"""       
        