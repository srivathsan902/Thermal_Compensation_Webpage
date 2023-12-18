import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
# import tensorflow as tf
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.callbacks import EarlyStopping
# from keras.initializers import HeNormal
# from keras.regularizers import l2
# from keras.losses import MeanSquaredError
# from keras.metrics import RootMeanSquaredError
# from keras.optimizers import Adam

def linear_regression(X_data, Y_data):
    X_train, X_test = X_data
    Y_train, Y_test = Y_data
    
    LR = LinearRegression()
    LR.fit(X_train, Y_train)
    Y_pred = LR.predict(X_test)
    
    rmse = mean_squared_error(Y_test, Y_pred, squared = False)
    
    return rmse

def random_forest(X_data, Y_data):
    X_train, X_test = X_data
    Y_train, Y_test = Y_data
    
    RF = RandomForestRegressor(n_estimators=100, random_state=7)  
    RF.fit(X_train, Y_train.ravel())
    Y_pred = RF.predict(X_test)
    
    rmse = mean_squared_error(Y_test, Y_pred, squared = False)
    
    return rmse

def support_vector_machine(X_data, Y_data):
    X_train, X_test = X_data
    Y_train, Y_test = Y_data
    
    SVM = SVR(kernel = 'rbf')
    SVM.fit(X_train, Y_train.ravel())
    Y_pred = SVM.predict(X_test)
    
    rmse = mean_squared_error(Y_test, Y_pred, squared = False)   
    
    return rmse

def neural_networks(X_data, Y_data):
    X_train, X_test = X_data
    Y_train, Y_test = Y_data
    
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=7)
    input_shape = X_train.shape[1]

    tf.random.set_seed(7)
    DNN = Sequential()
    DNN.add(Dense(256, input_shape = (input_shape,), activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)))
    # DNN.add(Dropout(0.5))
    DNN.add(Dense(64, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)))
    # DNN.add(Dropout(0.5))
    DNN.add(Dense(32, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)))
    DNN.add(Dense(1))

    DNN.compile(optimizer='adam',loss='mean_squared_error', metrics=['mse'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)

    DNN.fit(X_train, Y_train, epochs=1000, batch_size=32, validation_data=(X_val, Y_val), callbacks=[early_stopping], verbose=0)

    Y_pred = DNN.predict(X_test)
    rmse = mean_squared_error(Y_test, Y_pred, squared = False)   

    return rmse

def NN_online(X_train, Y_train, X_test, Y_test, threshold):
    scaler = MinMaxScaler(feature_range=(0,1))

    input_shape = X_train['Day 1'].shape[1]

    tf.random.set_seed(7)
    DNN = Sequential()
    DNN.add(Dense(256, input_shape = (input_shape,), activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)))
    # DNN.add(Dropout(0.5))
    DNN.add(Dense(64, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)))
    # DNN.add(Dropout(0.5))
    DNN.add(Dense(32, activation='relu', kernel_initializer=HeNormal(), kernel_regularizer=l2(0.01)))
    DNN.add(Dense(1))

    DNN.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate = 0.001), metrics = RootMeanSquaredError())

    X_train_data = X_train['Day 1']
    Y_train_data = Y_train['Day 1']
    X_train_data = scaler.fit_transform(X_train_data)
    
    DNN.fit(X_train_data, Y_train_data, epochs=200, batch_size=32, verbose=0)

    for i in range(2, len(X_train)+1):
        X_test_data = X_train['Day '+str(i)]
        Y_test_data = Y_train['Day '+str(i)]
        X_test_data = scaler.fit_transform(X_test_data)
        Y_pred = DNN.predict(X_test_data, verbose=0)
        rmse = mean_squared_error(Y_test_data, Y_pred, squared = False)  
        if rmse > threshold:
            DNN.compile(loss = MeanSquaredError(), optimizer = Adam(learning_rate = 0.0005), metrics = RootMeanSquaredError())
            DNN.fit(X_test_data, Y_test_data, epochs=200, batch_size=32, verbose=0)

    Y_pred = DNN.predict(X_test, verbose=0)
    rmse = mean_squared_error(Y_test, Y_pred, squared = False)   

    return rmse

def model_training(model,X_data,Y_data, threshold, temp, displacement, test_split):
    scaler = MinMaxScaler(feature_range=(0,1))
    X_train, X_test = X_data
    Y_train, Y_test = Y_data
    
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    X_data_scaled = [X_train, X_test]
    Y_data = [Y_train, Y_test]
    
    model_options = ["Linear Regression", "Random Forest", "Support Vector Machine", "Neural Networks", "Neural Networks - Online Learning"]
    
    if model == model_options[0]:
        results = linear_regression(X_data_scaled, Y_data)
    elif model == model_options[1]:
        results = random_forest(X_data_scaled, Y_data)
    elif model == model_options[2]:
        results = support_vector_machine(X_data_scaled, Y_data)
    elif model == model_options[3]:
        results = neural_networks(X_data_scaled, Y_data)
    elif model == model_options[4]:
        tot_days = len(list(temp.keys()))
        test_days = int(test_split * tot_days)
        train_days = tot_days - test_days
        X_train = dict(list(temp.items())[:train_days])
        Y_train = dict(list(displacement.items())[:train_days])
        results = NN_online(X_train, Y_train, X_test, Y_test, threshold)
        
    return results


# VISUALIZATION OF TEMP PREDICTIONS AND ERROR DAY WISE
# IMPLEMENTATION OF LSTM AND Bi LSTM model