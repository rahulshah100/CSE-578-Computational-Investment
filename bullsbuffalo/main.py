from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask import Flask, render_template
import pandas as pd
import json
import plotly
import plotly.express as px
from PIL import Image
import base64
import io
import plotly.graph_objs as go
import yfinance as yf

import warnings

warnings.filterwarnings("ignore")


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/by_stock_home', methods=['POST'])
def by_stock_home():
    return render_template('by_stock_select.html')

@app.route('/by_stock_results', methods=['POST'])
def by_stock_results():
    data = request.get_json()
    print(data)
    tickerstock = request.form.get('sel1')
    model_type = request.form.get('model1')
    time_range = request.form.get('time_range')
    time_range=int(time_range)
    print(tickerstock,model_type,time_range)

    data = yf.download(tickers=tickerstock, period='1d', interval='1m')
    # declare figure
    fig = go.Figure()
    # Candlestick
    trace1 = go.Candlestick(x=data.index,
                            open=data['Open'],
                            high=data['High'],
                            low=data['Low'],
                            close=data['Close'], name='market data')
    data1 = [trace1]
    graphJSON = json.dumps(data1, cls=plotly.utils.PlotlyJSONEncoder)

    if model_type!="ALL":
        error = pd.read_csv("final_results_"+model_type+".csv")
        mse = error.loc[error.Name==tickerstock].MSE.values[0]
        mse=round(mse,4)
        mape = error.loc[error.Name==tickerstock].MAPE.values[0]
        mape = round(mape,4)
        stock_data = pd.read_csv("results_"+model_type+"/"+tickerstock+"_results.csv")
        if model_type=="ARIMA":
            stock_data = stock_data[["date","open","close","outcome"]]
        elif model_type=="LinearRegression":
            stock_data = stock_data[["date","open","Actual","Predicted"]]
        elif model_type=="LSTM":
            stock_data = stock_data[["date","open","close","Predictions"]]
        stock_data.columns = ["date","open","close","outcome"]
        stock_data = stock_data.iloc[-time_range:,]
        fig = px.line(stock_data, x="date", y=['open','close','outcome'],
              hover_data={"date": "|%B %d, %Y"},
              title='Model Results')
        graphJSON1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return render_template('by_stock_results.html',rmse=mse,mape=mape,graphJSON1=graphJSON1,graphJSON=graphJSON, tickerstock=tickerstock, model_type =model_type)
    else:
        error_dict = {}
        graph_array = {}
        for x in ["ARIMA","LSTM","LinearRegression"]:
            print(x)
            error = pd.read_csv("final_results_"+x+".csv")
            mse = error.loc[error.Name==tickerstock].MSE.values[0]
            mse=round(mse,4)
            mape = error.loc[error.Name==tickerstock].MAPE.values[0]
            mape = round(mape,4)
            error_dict[x]=[mse,mape]
            stock_data = pd.read_csv("results_"+x+"/"+tickerstock+"_results.csv")
            if x=="ARIMA":
                stock_data = stock_data.loc[:,["date","open","close","outcome"]]
            elif x=="LinearRegression":
                stock_data = stock_data[["date","open","Actual","Predicted"]]
            elif x=="LSTM":
                stock_data = stock_data[["date","open","close","Predictions"]]
            print(stock_data.head())
            stock_data.columns = ["date","open","close","outcome"]
            stock_data = stock_data.iloc[-time_range:,]
            fig = px.line(stock_data, x="date", y=['open','close','outcome'],
                hover_data={"date": "|%B %d, %Y"},
                title='Model Results for '+x+ " \nRMSE :"+str(mse)+"/MAPE :"+str(mape),width=700, height=400)
            graphJSON1 = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            graph_array[x]=graphJSON1

        min = 10000
        min_rmse = 0
        min_smape = 0
        best_model = ""
        for k,v in error_dict.items():
            if v[1]<min:
                best_model=k
                min_rmse=v[0]
                min_smape=v[1]





        return render_template('by_all_model.html',best_model=best_model,min_rmse=min_rmse,\
            min_smape=min_smape,graph_arima=graph_array['ARIMA'],graph_LSTM=graph_array["LSTM"],\
            graph_Linear=graph_array["LinearRegression"],graph_RF=graph_array["LinearRegression"],graphJSON=graphJSON,tickerstock=tickerstock, model_type =model_type)


    







    
@app.route('/all_stock_results', methods=['POST'])
def all_stock_results():
    data = request.get_json()
    return render_template('by_stock_select.html')



def get_candlestick(tickerstock):
    data = yf.download(tickers='tickerstock', period='90d', interval='1d')
    # declare figure
    fig = go.Figure()
    # Candlestick
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'], name='market data'))
    # Add titles
    fig.update_layout(
        title='Uber live share price evolution',
        yaxis_title='Stock Price (USD per Shares)')
    # X-Axes
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=15, label="15m", step="minute", stepmode="backward"),
                dict(count=45, label="45m", step="minute", stepmode="backward"),
                dict(count=1, label="HTD", step="hour", stepmode="todate"),
                dict(count=3, label="3h", step="hour", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    # Show
    my1_stringIObytes = io.BytesIO()
    fig.savefig(my1_stringIObytes, format='jpg')
    my1_stringIObytes.seek(0)
    my1_base64_jpgData = base64.b64encode(my1_stringIObytes.read())
    return my1_base64_jpgData


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082, debug=True)
