#### Stock-Prices-Prediction-Using-Long-Short-Term-Memory-LSTM-Model

##Preparing dataset and preprocessing:

The stock dataset has 4 features which we use for prediction, namely :- Volume, Open, High, Low. We have to look 3 days into the past for training and 1 day into the future for prediction on our model. We assign the target feature of ‘Open’, as y_train_data (dependent variable) and the other 3 features (Volume, High, Low) make up our X_train_data (independent variables). Next to get all the features in the same range (0,1), we transformed all the features using Min-max scaler. This helps avoid large jumps in uneven data. Scaling dating helps counter the inherent volatility in the data, which is otherwise reflected in our calculations and visualizations. Also, ‘Date’ and ‘Close’ features were dropped out of the dataset.
As for the test data set, we split out original data into training and testing data in a 70:30 ratio.

##Design steps and result discussion:

For selecting the best performing network, several iterations of design were experimented with and we tried varying combinations of LSTM layers ranging from 8 to 128 units. Overfitting was one of the biggest challenges we kept facing and to address this issue we implemented different regularization techniques such as L1 and L2. L2 regularization worked out to give us the best results for our given data, we also had a dense layer with 1 unit at the end for a single regression output. This layer made use of ‘linear’ activation function and we used the Adam optimizer with our output loss function as mean absolute error. We found that a 2 layer LSTM network with the aforementioned settings, to give us the best fit on our data. We set a maximum of 300 epochs for our model to train but enabled it with a patience value of 10, meaning it stopped training in about 30 epochs. We also set a batch size of 32 while training. The model summary for our network is shown below –

![image](https://user-images.githubusercontent.com/62597096/190698885-dff51d4c-f241-47ac-ae72-ce3256978646.png)


Our training algorithm runs for around 30 epochs and we get the following losses:

![image](https://user-images.githubusercontent.com/62597096/190698967-de1a5c98-58c2-4471-96f4-4986c8cb74ce.png)


These were amongst the best loss results we got out of the different network combinations we tried. Following is the plot depicting the Training vs Validation Loss against the Number of Epochs. As you can see, there is an exponential decay in loss up until the 3rd epoch, followed by plateauing of Training and Validation loss metrics thereafter.

![image](https://user-images.githubusercontent.com/62597096/190699474-19c85a61-9319-4ead-ae3a-9e5f5ed84609.png)

We then run our network on the test set and get the following results evaluating our model metrics:

![image](https://user-images.githubusercontent.com/62597096/190699587-e5832793-39ad-45cd-bcfc-9fab32f8c64b.png)


These results obtained with this network were amongst the best ones we got with different network combinations. With R2 value of 0.99 and explained variance of 0.99, we can interpret that 99% variance of our dependent variable (‘Open’) is explained by the variance of our other independent variables.
Next we plot the actual open prices vs predicted open prices. This plot further explains our observations and interpretation given above.

![image](https://user-images.githubusercontent.com/62597096/190700106-4064e27b-d4e0-46e8-a6f8-b496ffb47cfd.png)


For the case of using more days as features, we used 14 days in the past to predict the next day opening prices.
When we use more days in the past to predict the immediate step “Open” prices, then we see the following metrics while training:

![image](https://user-images.githubusercontent.com/62597096/190700414-82173708-e397-4668-a06c-ea1964bb0f3a.png)

There is a marginal improvement in our training and validation loss from what we see in the standard case of looking 3 days in the past.
While testing the same model looking back 14 days in the past to predict our prices, we obtain the following metrics:

![image](https://user-images.githubusercontent.com/62597096/190700507-cb5d2caf-956c-41a3-8bee-3401499a19ed.png)

These metrics are almost similar to our previous case and we don’t see a great improvement or decrease in the model’s performance. A marginal decrease in the MSE can be observed, further showing that an increase in a few days does not necessarily have a significant impact on our model’s performance.
