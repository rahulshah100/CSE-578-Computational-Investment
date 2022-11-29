import numpy as np 
import pandas as pd 
import pandas_datareader as data 
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st 
 
model = load_model('keras_model.h5')

# Putting Titles
st.markdown("<h2 style='text-align: center; color: grey;'>CSE 578: Computational Investment 2022</h2>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: grey;'>Stock Market Data Prediction</h3>", unsafe_allow_html=True)

#Start and End Dates
start = '2010-01-01'
end = '2021-12-31'

#Fetching User Input
user_input = st.text_input("Enter Stock Ticker", 'TSLA')

#Making dataframe
df = data.DataReader(user_input, 'yahoo', start,end)
df = df.reset_index()
st.subheader('Data from 2010 - 2021')
st.write(df.describe())

#Visualizing
st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize = (12,6))
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 days MA')
fig = plt.figure(figsize = (12,6))
ma100 = df.Close.rolling(100).mean()
plt.xlabel('Time')
plt.ylabel('Price')
plt.plot(ma100, 'r')
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 days MA & 200 days MA')
fig = plt.figure(figsize = (12,6))
plt.xlabel('Time')
plt.ylabel('Price')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')

plt.plot(ma100, 'r', label = '100 Moving Average')
plt.plot(ma200, 'g', label = '200 Moving Average')

plt.legend()
plt.plot(df.Close)
st.pyplot(fig)

#Splitting into Training and Testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing =  pd.DataFrame(df['Close'][int(len(df)*0.70):int(len(df))])

#Scaling down Data
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
data_training_array = scaler.fit_transform(data_training)

x_train = []
y_train = []

for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

#Model Making LSTM
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

#Testing Part
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index = True)

#scaling down inputs
inputs = scaler.transform(final_df)

x_test = [] 
y_test = []

#x-test and y-test
for i in range(100, (inputs.shape[0])):
    x_test.append(inputs[i-100: i])
    y_test.append(inputs[i,0])

x_test, y_test = np.array(x_test), np.array(y_test)

#Predictions
y_predicted = (model.predict(x_test)) 

#Scaling up values
scale = scaler.scale_[0]

y_predicted = scale * y_predicted
y_test = scale * y_test

st.subheader('Predictions vs Actual')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'r', label = 'Original Price')
plt.plot(y_predicted, 'g', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

st.markdown("<h4 style='text-align: CENTER; color: grey;'><b><hr/>Team Details:</b></h4>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: CENTER; color: grey;'>Rahul Shah</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: CENTER; color: grey;'>Chitravardhini</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: CENTER; color: grey;'>Abhirami</h5>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: CENTER; color: grey;'>Dheeraj</h5><hr/>", unsafe_allow_html=True)