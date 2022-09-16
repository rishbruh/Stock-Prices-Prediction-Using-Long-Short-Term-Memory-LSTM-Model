#This is the code file for training our RNN model on our training dataset for predicting stock prices


#importing required packages
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras import regularizers
from pickle import dump, load
from sklearn.model_selection import train_test_split


#defining the RNN model
def rnn_model():
    model = Sequential()

    # First LSTM layer is added
    model.add(LSTM(units=128, return_sequences=True, input_shape=(3, 3)))
    regularizers.l2(l2=0.01) # using L2 regularizer

    # Second LSTM layer is added
    model.add(LSTM(units=8, return_sequences=False))
    regularizers.l2(l2=0.01) # using L2 regularizer

    # Adding Dense output layer and compiling RNN model
    model.add(Dense(units=1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_absolute_error') # RNN model is compiled
    return model

#running the final RNN model
def rnn_model_run(X_train_data, y_train_data, model):
    #adding early stopping in case of no improvement in val loss curve
    early_stop = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=10, verbose=1)
    #reducing learning rate in case of val_loss curve plateauing
    red_learn_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
    #Model fitted for 300 epochs
    history = model.fit(X_train_data, y_train_data, shuffle=True, epochs=300, callbacks=[early_stop, red_learn_rate], validation_split=0.20, verbose=1, batch_size=32)
    return history

#preparing training dataset
def train_data(data):
    #number of past days
    no_past_days = 3
    #no_past_days = 14
    #number of days to predict in future
    no_future_days = 1

    data_train = data
    # 'Open', is used as y_train (dependent variable) and other 3 features(Volume, High, Low)  make up X_train (independent variables)
    data_train_target = data.loc[:, ' Open']
    data_train.drop(['Date'], axis=1, inplace=True)
    data_train.drop([' Open'], axis=1, inplace=True)

    # converting to np array
    data_train = np.array(data_train)
    data_train_target = np.array(data_train_target).reshape(-1, 1)

    # Creating x_train_data and y_train_data with 3 past days' data and data for 1 day in future
    X_train_data = []
    y_train_data = []
    for i in range(no_past_days, len(data_train) - no_future_days + 1):
        X_train_data.append(data_train[i - no_past_days:i, 1:data_train.shape[1]])
        y_train_data.append(data_train_target[i + no_future_days - 1:i + no_future_days, :])
    X_train_data = np.asarray(X_train_data)
    y_train_data = np.asarray(y_train_data).reshape(-1,1)
    return X_train_data, y_train_data

#defining function to plot train and validation curves
def train_val_plot(history):
    plt.title('Loss vs Epochs')
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.legend()
    plt.show()

def main():
    # data = pd.read_csv('data/q2_dataset.csv') #original dataset loaded into data
    # #calling train_data fucntion to create x_train_data and y_train_data with 3 past days' data and data for 1 day in future
    # X_train_data, y_train_data = train_data(data)
    # #splitting into training and test dataset in a 70/30 ratio randomly
    # X_train_data, X_test, y_train_data, y_test = train_test_split(X_train_data, y_train_data, test_size=0.3, random_state=42, shuffle=True)
    #
    # #reshaping training and test sets and concatenating y_train_data, y_test to X_train_data and X_test respectively
    # X_train_data = X_train_data.reshape(X_train_data.shape[0], -1)
    # X_test = X_test.reshape(X_test.shape[0], -1)
    # y_train_data = y_train_data.reshape(y_train_data.shape[0], -1)
    # y_test = y_test.reshape(y_test.shape[0], -1)
    # X_train_data = np.concatenate((X_train_data, y_train_data), axis=1)
    # X_test = np.concatenate((X_test, y_test), axis=1)
    #
    # #saving final training and test sets in data folder
    # np.savetxt("data/train_data_RNN.csv", X_train_data)
    # np.savetxt("data/test_data_RNN.csv", X_test)

    #Loading training dataset
    X_train_data = np.loadtxt("data/train_data_RNN.csv")

    # taking y_train_data from dataset ( it is the last column when concatenated) and deleting y_train_data column (last column) from X_train_data
    y_train_data = X_train_data[:, -1]
    X_train_data = np.delete(X_train_data, -1, 1)

    # setting scalers for X_train_data and y_train_data for later use by X_test and y_test
    predict_scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = MinMaxScaler(feature_range=(0, 1))

    #y_train_data is reshaped to be fed to the model
    y_train_data = np.reshape(y_train_data, [-1, 1])

    # applying fit_transform on X_train_data and y_train_data
    y_train_data = predict_scaler.fit_transform(y_train_data)
    X_train_data = scaler.fit_transform(X_train_data)
    # X_train_data is reshaped
    X_train_data = X_train_data.reshape(X_train_data.shape[0], 3, 3)

    # scaling weights are saved
    dump(scaler, open('scaler.pkl', 'wb'))
    dump(predict_scaler, open('predict_scaler.pkl', 'wb'))
    model = rnn_model() #creating the model
    history = rnn_model_run(X_train_data, y_train_data, model)  #Fitting the model
    print("Training loss is ", history.history['loss'][-1])
    print("Validation loss is ", history.history['val_loss'][-1])

    model.save('models/Group30_RNN_model.h5') #saving the model to be used in test_RNN.py
    train_val_plot(history) # plotting training and validation loss

if __name__ == "__main__":
    main()
