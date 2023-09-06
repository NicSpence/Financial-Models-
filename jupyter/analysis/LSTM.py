#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasRegressor


# Configurable inputs
total_prediction_days = 365
prediction_window_size = 60
epochs = [10, 20, 30, 40, 50, 60, 70]
batch_size = [50, 100, 200]
prediction_for_days = 1500
# for reproducibility
tf.random.set_seed(1)


def percentageChange(baseValue, currentValue):
    return((float(currentValue)-baseValue) / abs(baseValue)) *100.00

def reversePercentageChange(baseValue, percentage):
    return float(baseValue) + float(baseValue * percentage / 100.00)

def transformToPercentageChange(x):
    baseValue = x[0]
    x[0] = 0
    for i in range(1,len(x)):
        pChange = percentageChange(baseValue,x[i])
        baseValue = x[i]
        x[i] = pChange

def reverseTransformToPercentageChange(baseValue, x):
    x_transform = []
    for i in range(0,len(x)):
        value = reversePercentageChange(baseValue,x[i])
        baseValue = value
        x_transform.append(value)
    return x_transform


df = pd.read_csv('SPXH.csv')
baseValue = df['Close'][0]

data = df.sort_index(ascending=True, axis=0)
new_data = pd.DataFrame(index=range(0,len(df)),columns=['Date', 'Close'])
for i in range(0,len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
transformToPercentageChange(new_data['Close'])

new_data.index = new_data.Date
new_data.drop('Date', axis=1, inplace=True)

dataset = new_data.values
train, valid = train_test_split(dataset, train_size=0.99, test_size=0.01, shuffle=False)


x_train, y_train = [], []
for i in range(prediction_window_size,len(train)):
    x_train.append(dataset[i-prediction_window_size:i,0])
    y_train.append(dataset[i,0])
x_train = np.array(x_train, dtype=float)
y_train = np.array(y_train, dtype=float)
x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


def create_model():
    model = Sequential()
    #first LSTM layer
    model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    #second LSTM layer and some Dropout regularisation
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    #third LSTM layer
    model.add(LSTM(units = 50, return_sequences = True))
    model.add(Dropout(0.2))

    #fourth LSTM layer
    model.add(LSTM(units = 50))
    model.add(Dropout(0.2))

    # Adding the output layer
    model.add(Dense(units = 1))
    # Compiling the RNN
    model.compile(optimizer = 'adam', loss = 'mean_squared_error')
    
    return model

model = KerasRegressor(build_fn = create_model, verbose = 1)

param_grid = dict(batch_size=batch_size, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(x_train, y_train)
# Fitting the RNN to the Training set
#model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, workers=8, use_multiprocessing=True)
print("Best grid search params: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

model = grid_result.best_estimator_.model


[print(i.shape, i.dtype) for i in model.inputs]
[print(o.shape, o.dtype) for o in model.outputs]
[print(l.name, l.input_shape, l.dtype) for l in model.layers]

# serialize the model to disk for later
model.save("model_serialized.h5")


inputs = new_data[-total_prediction_days:].values
inputs = inputs.reshape(-1,1)

X_predict = []
for i in range(prediction_window_size,inputs.shape[0]):
    X_predict.append(inputs[i-prediction_window_size:i,0])
X_predict = np.array(X_predict, dtype=float)

X_predict = np.reshape(X_predict, (X_predict.shape[0],X_predict.shape[1],1))
future_closing_price = model.predict(X_predict)

train, valid = train_test_split(new_data, train_size=0.99, test_size=0.01, shuffle=False)
date_index = pd.to_datetime(train.index)


x_days = (date_index - pd.to_datetime('2018-08-01')).days

future_closing_price = future_closing_price[:prediction_for_days]

x_predict_future_dates = np.asarray(pd.RangeIndex(start=x_days[-1] + 1, stop=x_days[-1] + 1 + (len(future_closing_price))))
future_date_index = pd.to_datetime(x_predict_future_dates, origin='2018-08-01', unit='D')

train_transform = reverseTransformToPercentageChange(baseValue, train['Close'])

baseValue = train_transform[-1]
valid_transform = reverseTransformToPercentageChange(baseValue, valid['Close'])
future_closing_price_transform = reverseTransformToPercentageChange(baseValue, future_closing_price)

# recession peak date is the date on which the index is at the bottom most position.
recessionPeakDate =  future_date_index[future_closing_price_transform.index(min(future_closing_price_transform))]
minCloseInFuture = min(future_closing_price_transform);
print("The stock market will reach to its lowest bottom on", recessionPeakDate)
print("The lowest index the stock market will fall to is ", minCloseInFuture)

# plot the graphs
plt.figure(figsize=(16,8))
df_x = pd.to_datetime(new_data.index)
plt.plot(date_index,train_transform, label='Close Price History')
plt.plot(future_date_index,future_closing_price_transform, label='Predicted Close')

# set the title of the graph
plt.suptitle('Stock Market Predictions', fontsize=16)

# set the title of the graph window
#fig = plt.gcf()
#fig.canvas.set_window_title('Stock Market Predictions')

#display the legends
plt.legend()

#display the graph
#plt.show()
plt.savefig("stock_market_predictions.png")


# # optional pass the predicted prices through a pandas data frame to export results to csv for further analysis.


df = pd.DataFrame(future_closing_price_transform)
df.to_csv('future_closing_price_transform.csv')
# print(df)


