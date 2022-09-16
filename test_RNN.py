#This is the code file for testing our RNN model on our test dataset for predicting stock prices

#importing required packages
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from tensorflow.python.keras.models import load_model
from pickle import dump, load

#defining the metrics to be used for evaluating our RNN model's performance
def model_metrics(preds, Y):
    mean_sq_er = mean_squared_error(Y, preds)
    mean_abs_er = mean_absolute_error(Y, preds)
    exp_var = explained_variance_score(Y, preds)
    r2 = r2_score(Y, preds)
    print("Mean Squared error is ", mean_sq_er)
    print("Root Mean Squared error is ", mean_sq_er ** 0.5)
    print("Mean Absolute error is ", mean_abs_er)
    print("Explained Variance score is ", exp_var)
    print("R2 value is ", r2)

#plotting the predicted vs actual values after model prediction
def final_plot(preds,y):
    plt.plot(preds, color='red', label='Predictions')
    plt.plot(y, color='green', label='Actual Values')
    plt.title('Predicted vs Actual Values')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

def main():

    X_test = np.loadtxt("data/test_data_RNN.csv") # loading X_test dataset
    sc = load(open('scaler.pkl', 'rb')) #scaling weights are loaded
    sc_predict = load(open('predict_scaler.pkl', 'rb'))
    # extracting y_train and deleting y_test column added earlier from X_test
    y_test = X_test[:, -1]
    X_test = np.delete(X_test, -1, 1)
    # reshaping and transforming y_test and X_test
    y_test = np.reshape(y_test, [-1, 1])
    X_test = sc.transform(X_test)
    X_test = X_test.reshape(X_test.shape[0], 3, 3)

    #loading our saved RNN model
    model = load_model('models/Group30_RNN_model.h5')
    print(model.summary())

    #predicting final values by running the model on the test set)
    final_preds = model.predict(X_test)
    final_preds = sc_predict.inverse_transform(final_preds)

    #printing the final RNN model metrics and plotting predicted vs actual values
    model_metrics(final_preds, y_test)
    final_plot(final_preds, y_test)

if __name__ == "__main__":
    main()