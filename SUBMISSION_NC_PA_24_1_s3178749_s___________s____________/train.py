# Task: Time Series Forecasting on a Synthetic Data Set
# Data: please see train.csv available on Brightspace
# Specifications:
# You are required to implement a recurrent neural network in PyTorch, which takes as input,
# a recent history of time step  t , e.g., ... ,  t−3 ,  t−2 ,  t−1 ,  t .
# to predict five time step in the future, i.e.,  t+1 ,  t+2 ,  t+3 ,  t+4 ,  t+5 .
# You can use any recurrent NN models taught from the class.
# You could choose the length of the history fed into the model by yourselves.
# The resulting code structure should contain (1) model.py -> the implementation of your own RNN model; (2) train.py -> the training code, which can be executed from the command line by python train.py; (3) requirements.txt that lists the Python packages your are using, including the version information.
# You need to submit your source code and a dumpy file of the best model you ever trained. When handing in the assigment, please put model.py, train.py, requirements.txt, and the model dump file in the same folder named by your group and student IDs. The name must be SUBMISSION__ (e.g., SUBMISSION_NC_PA_24_1_s3610233_s3610234_s3610235). Please see https://pytorch.org/tutorials/beginner/saving_loading_models.html for a tutorial on how to save/load the model.


# DEADLINE: June 21, 2024, 12:00.
# Please deliver your assignment on Brightspace.
# The practical assignment accounts for 30% of the final grade.
# When training your RNN model locally on train.csv, we suggest to use the [Mean Absolute Percentage Error (MAPE)](Mean Absolute Percentage Error) metric to track the performance since we will use this metric to evaluate your model (see below)
# Evaluation criteria:
# Your train.py should be executable - We will contact you in case a bug is encountered. In this case, you will have one chance to fix it, with a penalty of 1 out of 10.
# We will execute your train.py on the training data set train.csv, checking against bugs.
# We will load your best saved model and evaluate it on a testing data set hidden to you.
# Any bugs occur in the evaluation phase will incur a penalty of 1 out of 10.
# The evaluation performance - MAPE - on the testing data will be ranked and the top-5 groups will get a bonus of 2 of 10.
import sklearn
from sklearn.model_selection import train_test_split # for preparing the training and testing data sets. You should get yourself familiar with it.
from sklearn.preprocessing import MinMaxScaler       # Data preprocessing
from sklearn.metrics import accuracy_score           # performance metrics
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.utils.data as data_util

import pandas as pd
import numpy as np

from model import RecurrentNN, RecurrentNNLSTM, RecurrentNNHopfield, RecurrentNNSeq2Seq

random_seed = 42
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)



def train():
    with open('train.csv', 'r') as f:
        # Date,store,product,number_sold
        # load to df
        df = pd.read_csv(f, sep=',')

    print(df.head())


    data = df['number_sold'].values
    plt.plot(data)
    plt.xlabel('time')
    plt.ylabel('number_sold')
    plt.title('Number of sold products over time')
    plt.savefig('number_sold.png')
    
    # normalize data
    data = data.reshape(-1,1)
    data = data.astype("float32")
    data.shape

    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    
    # split into test and train using train_test_split
    train_size = int(len(data) * 0.67)
    test_size = len(data) - train_size
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    # Data transformation to tensors

    time_step = 20

    xtrain, ytrain = [], []
    for i in range(len(train)-time_step):
        feature = train[i:i+time_step]
        target = train[i+1:i+time_step+1]
        xtrain.append(feature)
        ytrain.append(target)
        trainX = torch.tensor(xtrain)
        trainY = torch.tensor(ytrain)

    xtest, ytest = [], []
    for i in range(len(test)-time_step):
        feature = test[i:i+time_step]
        target = test[i+1:i+time_step+1]
        xtest.append(feature)
        ytest.append(target)
        testX = torch.tensor(xtest)
        testY = torch.tensor(ytest)

    # Please pay attention to the functionality of the following functions:

    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()
    # model.eval()
    # torch.no_grad()

    # train and validate the imported RNN model
    n_epochs = 100
    mod_epochs = 5 # model saving epochfs
    
    learning_rate = 0.005
    
    model = RecurrentNN()
    # model = RecurrentNNLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()
    
    loader = data_util.DataLoader(data_util.TensorDataset(trainX, trainY), shuffle=True, batch_size=8)
    
    for epoch in range(n_epochs + 1):
        model.train() # set the model to training mode
        for X_batch, y_batch in loader:
            y_pred = model(X_batch) # one forward pass
            optimizer.zero_grad() # clear the gradients
            loss = loss_fn(model(X_batch), y_batch) # compute the loss by this forward pass
            loss.backward() # compute the gradients by backpropagation
            optimizer.step() # the actual step of gradient descent
            
        if epoch % mod_epochs != 0:
            continue
        
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            y_pred = model(trainX)
            train_rmse = np.sqrt(loss_fn(y_pred, trainY)) # compute the loss by this forward pass, rmse means root mean squared error
            y_pred = model(testX)
            test_rmse = np.sqrt(loss_fn(y_pred, testY))
        print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
            

if __name__ == "__main__":
    train()