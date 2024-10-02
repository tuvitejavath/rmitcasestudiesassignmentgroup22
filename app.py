import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import datetime
from datetime import date,timedelta
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
#TITLE

app_name = 'Stock Market Prediction'
st.title(app_name)
st.subheader('This Website was designed to predict stock market price for Selected MNC')

#IMAGE
st.image("https://media.istockphoto.com/id/1313570804/vector/bear-bull-with-chart-bar-logo-design-finance-vector-design.jpg?s=612x612&w=0&k=20&c=aCLnilocQ5oegr2q6Rw-rPXukvI4u-aN2wZ4RxPDQDk=")

#Input FROM USERS(Start date and end date)
st.sidebar.header('Filter')

start_date = st.sidebar.date_input('Start date',date(2020,1,1))
end_date = st.sidebar.date_input('End_date',date(2023,12,31))

ticker_list = ['AAPL','MSFT','GOOGL','META','TSLA','NVDA','ADBE','PYPL','INTC','CMCSA','NFLX','PEP']

ticker = st.sidebar.selectbox('Company', ticker_list)

#Getting data from User inputs

data = yf.download(ticker,start = start_date,end = end_date)

data.insert(0,'Date',data.index,True)
data.reset_index(drop =True, inplace= True)
st.write('Data from',start_date, 'to',end_date)

st.write(data)

#PLOT
st.header('Data Visualisation')
st.subheader('Plot of the Data')

fig = px.line(data,x='Date',y = data.columns,title='Stock Price',width=800,height=600)
st.plotly_chart(fig)


column = st.selectbox('Select the Column to be used for Prediction and Visualization', data.columns[1:])
data = data[['Date', column]]
st.write('Selected Data')
st.write(data)

#Checking if the datas are Stationary or not

st.header('Is Data Stationary??')
st.write(adfuller(data[column])[1] < 0.05)

#Model

#p = st.slider('Select the valie of p',0,5,2)
#d = st.slider('Select the value of d',0,5,1)
#q = st.slider('Select the value of q',0,5,2)
#seasonal_order = st.number_input('Select the value of seasonal p',0,24,12)

model = sm.tsa.statespace.SARIMAX(data[column], order = (2,1,2), seasonal_order= (2,1,2,12))
model = model.fit()

#st.header('Model Summary')
#st.write(model.summary())
#st.write('___')

st.write('Predicting the Future Value')
#Predictions
forecast_period = st.number_input('Input the number of days to FORECAST', 1,365,10)
predictions = model.get_prediction(start = len(data), end = len(data)+forecast_period-1)
predictions = predictions.predicted_mean
#st.write(predictions)


#add index to the predictions

predictions.index = pd.date_range(start=end_date, periods = len(predictions),freq ='D')
predictions = pd.DataFrame(predictions)
predictions.insert(0, 'Date', predictions.index, True)
predictions.reset_index(drop=True, inplace =True)
st.write('Predictions',predictions)
st.write('Actual data',data)
st.write('___')

#Lets plot the data
fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data[column], mode='lines',name ='Actual', line = dict(color= 'blue')))

fig.add_trace(go.Scatter(x=predictions['Date'],y = predictions['predicted_mean'], mode= 'lines',name = 'Predicted', line = dict(color = 'red')))

fig.update_layout(title = 'Actual vs Predicted', xaxis_title = 'Date', yaxis_title = 'Price', width= 1200,height = 400)
st.plotly_chart(fig)