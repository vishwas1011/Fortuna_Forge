from pydantic import BaseModel
from fastapi import FastAPI, HTTPException, Form, Request, Query, status
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pymongo import MongoClient
import yfinance as yf
import pandas as pd
import numpy as np
from arch import arch_model
import os
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from datetime import datetime, timedelta
from pmdarima.arima.utils import ndiffs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, plot_predict
from statsmodels.tsa.arima.model import ARIMA
import datetime as dt
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler 
from keras.layers import Dense, Dropout,LSTM
from keras.models import Sequential 

app = FastAPI()
client = MongoClient('mongodb://localhost:27017/')
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:3000", "http://127.0.0.1:3002"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class User(BaseModel):
    username: str
    email: str
    password: str

@app.post('/signup', response_class=HTMLResponse)
def signup(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    db = client['user_authentication']
    users_collection = db['users']
    existing_user = users_collection.find_one({"username": username})
    if existing_user:
        error_message = "Username already exists"
        return RedirectResponse(url="http://127.0.0.1:3000/templates/signup.html?error=Email not registered", status_code=status.HTTP_301_MOVED_PERMANENTLY)
    users_collection.insert_one({'username': username, 'email': email, 'password': password})
    success_message = "User signed up successfully!"
    return RedirectResponse(url="http://127.0.0.1:3000/templates/login.html?error=Email not registered", status_code=status.HTTP_301_MOVED_PERMANENTLY)


@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    db = client['user_authentication']
    users_collection = db['users']
    
    # Check if user exists with the provided email
    user = users_collection.find_one({"email": username})
    
    # If user not found, redirect to login page with error message
    if user is None:
        return RedirectResponse(url="http://127.0.0.1:3000/templates/login.html?error=Email not registered", status_code=status.HTTP_301_MOVED_PERMANENTLY)
    
    # If user exists, check if the provided password matches
    if user['password'] != password:
        # If password doesn't match, redirect to login page with error message
        return RedirectResponse(url="http://127.0.0.1:3000/templates/login.html?error=Wrong password", status_code=status.HTTP_301_MOVED_PERMANENTLY)

    # If login successful, redirect to home page
    return RedirectResponse(url="http://127.0.0.1:3000/templates/home.html", status_code=status.HTTP_301_MOVED_PERMANENTLY)


@app.get("/templates/{filename}")
async def get_static_file(filename: str):
    return FileResponse(f"templates/{filename}")

@app.get("/trend.html")
async def get_trend_page(request: Request):
    return templates.TemplateResponse("trend.html", {"request": request})

@app.get("/home.html")
async def get_trend_page(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/optimizer.html")
async def get_trend_page(request: Request):
    return templates.TemplateResponse("optimizer.html", {"request": request})

@app.get("/api/stock_price")
async def get_stock_price(symbol: str = Query(...)):
    try:
        stock = yf.Ticker(symbol)
        price = stock.history(period="1d")["Close"].iloc[-1]
        data = stock.history(period="1y")
    
        # Calculate beta
        spy = yf.Ticker("^GSPC")
        spy_data = spy.history(period="1y")
        beta = data['Close'].pct_change().cov(spy_data['Close'].pct_change()) / spy_data['Close'].pct_change().var()
        # beta,high_52week,open_price,close_price,low_price,high_price,pe_ratio,dividend_yield,eps
        # 52-week high
        high_52week = data['High'].max()
        
        # Opening, closing, low, and high values
        open_price = data['Open'].iloc[-1]
        close_price = data['Close'].iloc[-1]
        low_price = data['Low'].min()
        high_price = data['High'].max()
        
        # Price-to-Earnings (P/E) Ratio
        pe_ratio = stock.info['trailingPE']
        
        # Dividend Yield
        dividend_yield = stock.info['dividendYield']
        
        # Earnings Per Share (EPS)
        eps = stock.info['trailingEps']
        
        # Print the results
        print(f"Beta: {beta:.2f}")
        print(f"52-week high: {high_52week:.2f}")
        print(f"Opening price: {open_price:.2f}")
        print(f"Closing price: {close_price:.2f}")
        print(f"Low price: {low_price:.2f}")
        print(f"High price: {high_price:.2f}")
        print(f"P/E Ratio: {pe_ratio:.2f}")
        print(f"Dividend Yield: {dividend_yield:.2%}")
        print(f"EPS: {eps:.2f}")
        
        # Get the NSE Nifty 50 index data
        nse_index = "^NSEI"
        nse_data = yf.download(nse_index, start=data.index[0], end=data.index[-1])

        # Normalize the data for comparison
        stock_normalized = data['Close'] / data['Close'][0]
        nse_normalized = nse_data['Close'] / nse_data['Close'][0]

        # Create the plots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))

        # Plot the normalized stock and NSE Nifty 50 data
        ax1.plot(data.index, stock_normalized, label=symbol)
        ax1.plot(nse_data.index, nse_normalized, label="NSE Nifty 50")
        ax1.set_title(f"{symbol} Stock Price vs. NSE Nifty 50")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Normalized Value")
        ax1.legend()

        # Plot the stock price with moving averages
        ax2.plot(data.index, data['Close'])
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        data['EMA200'] = data['Close'].ewm(span=200, adjust=False).mean()
        ax2.plot(data.index, data['SMA50'])
        ax2.plot(data.index, data['EMA200'])
        ax2.set_title(f"{symbol} Stock Price with Moving Averages")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price (USD)")
        ax2.legend(['Close', '50-day SMA', '200-day EMA'])

        # Plot the daily returns distribution
        ax3.hist(data['Close'].pct_change(), bins=30)
        ax3.set_title(f"{symbol} Daily Returns Distribution")
        ax3.set_xlabel("Daily Return")
        ax3.set_ylabel("Density")
        
        # Convert the graph to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        return {"symbol": symbol, "price": price, "graph": image_base64, 'BETA': beta, 'HIGH_52WEEK': high_52week, 'OPEN_PRICE': open_price, 'CLOSE_PRICE': close_price, 'LOW_PRICE': low_price, 'HIGH_PRICE': high_price, 'PE_RATIO': pe_ratio, 'DIVIDEND_YIELD': dividend_yield, 'EPS': eps}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to generate ACF and PACF graphs and return base64 encoded images
@app.post("/api/arima_acf_pacf_graph")
async def arima_acf_pacf_graph(ticker: str = Form(...)):
    try:
        start = datetime(2015, 1, 1)
        end = datetime(2024, 1, 1)

        df = yf.download(ticker, start=start, end=end)
        df = df[["Adj Close"]].copy()

        d = ndiffs(df["Adj Close"], test='adf')
        for _ in range(d):
            diff = df.diff().dropna()

        # Plot ACF and PACF graphs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 4))
        plot_acf(diff, ax=ax1)
        plot_pacf(diff, ax=ax2)
        plt.suptitle('ACF and PACF Graphs')

        # Save the plot to a buffer and convert it to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        acf_pacf_graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {"acf_pacf_graph": acf_pacf_graph_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Function to perform ARIMA prediction and return actual vs fitted graphs
@app.post("/api/arima_prediction")
async def arima_prediction(ticker: str = Form(...), p: int = Form(...), q: int = Form(...)):
    try:
        start = datetime(2015, 1, 1)
        end = datetime(2024, 1, 1)

        df = yf.download(ticker, start=start, end=end)
        df = df[["Adj Close"]].copy()

        d = ndiffs(df["Adj Close"], test='adf')
        for _ in range(d):
            diff = df.diff().dropna()

        # Perform ARIMA model fitting
        model = ARIMA(df["Adj Close"], order=(p, d, q))
        result = model.fit()

        # Plot Actual vs Fitted Values and Future Predictions with Confidence Interval (7 Days) side by side
        plt.figure(figsize=(16, 6))

        # Plot Actual vs Fitted Values
        plt.subplot(1, 2, 1)
        plt.plot(df["Adj Close"].iloc[-30:], label="Actual", color='blue')
        plt.plot(result.fittedvalues.iloc[-30:], label="Fitted", color='red')
        plt.title('Actual vs Fitted Values (Last 30 Days)')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close')
        plt.legend()

        # Forecast 7 days into the future
        forecast = result.get_forecast(steps=7)
        forecast_values = forecast.predicted_mean
        conf_int = forecast.conf_int()

        # Plot Forecasted Values with Confidence Interval
        forecast_index = pd.date_range(start=end, periods=7)
        plt.subplot(1, 2, 2)
        plt.plot(forecast_index, forecast_values, color='green', label='Forecast')
        plt.fill_between(forecast_index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='lightgray')
        plt.title('Future Predictions with Confidence Interval (7 Days)')
        plt.xlabel('Date')
        plt.ylabel('Adjusted Close')
        plt.legend()

        plt.tight_layout()
        

        # Save the plot to a buffer and convert it to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        actual_fitted_graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {"actual_fitted_graph": actual_fitted_graph_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


    # Function to generate ACF and PACF graphs for GARCH model and return base64 encoded images

# Function to perform GARCH volatility prediction and return the plot
@app.post("/api/garch_volatility_prediction")
async def garch_volatility_prediction(ticker: str = Form(...)):
    try:
        start = datetime(2021, 1, 1)
        end = datetime(2024, 1, 1)

        df = yf.download(ticker, start=start, end=end)
        returns = 100 * df["Adj Close"].pct_change().dropna()

        rolling_predictions = []
        test_size = 365

        for i in range(test_size):
            train = returns[:-(test_size-i)]
            model = arch_model(train, p=2, q=2)
            model_fit = model.fit(disp='off')
            pred = model_fit.forecast(horizon=1)
            rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

        rolling_predictions = pd.Series(rolling_predictions, index=returns.index[-365:])

        # Plot the volatility prediction - rolling forecast
        plt.figure(figsize=(10,4))
        true, = plt.plot(returns[-365:])
        preds, = plt.plot(rolling_predictions)
        plt.title('GARCH Volatility Prediction - Rolling Forecast', fontsize=20)
        plt.legend(['True Returns', 'Predicted Volatility'], fontsize=16)

        # Save the plot to a buffer and convert it to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        volatility_prediction_graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {"actual_fitted_graph_garch": volatility_prediction_graph_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.post("/api/optimize")
async def arima_acf_pacf_graph(request: Request):
    try:
        form = await request.form()
        ticker = form.get('ticker')
        comma_separated_string = ticker

# Split the string by comma and create a list
        stocklist = comma_separated_string.split(',')
        end = dt.datetime.now()
        start = end - dt.timedelta(days=365)

        
        length = len(stocklist)


        def get_data(tickers, start, end):
            stock_data = []
            for ticker in tickers:
                stock_data.append(yf.download(ticker, start=start, end=end)['Adj Close'])
            
            df = pd.concat(stock_data, axis=1)
            df.columns = tickers
            return df

        df = get_data(stocklist,start, end)

        returns = (df-df.shift(1))/df.shift(1)

        # SHARPE RATIO
        noOFPortfolio = 1000
        mean_ret = returns.mean()
        length = len(stocklist)
        sigma = returns.cov()
        weight = np.zeros((noOFPortfolio, length))
        expectedVolatility = np.zeros(noOFPortfolio)
        expected_ret = np.zeros(noOFPortfolio)
        SharpeRatio = np.zeros(noOFPortfolio)
        for k in range(noOFPortfolio):
            # random weughts
            w = np.array(np.random.random(length))
            w = w/np.sum(w)
            weight[k,:] = w
            #exp return
            expected_ret[k]= np.sum(mean_ret*w)
            #exp vol
            expectedVolatility[k] = np.sqrt(np.dot(w.T,np.dot(sigma,w)))
            #sharpe ratio
            SharpeRatio[k] = expected_ret[k]/expectedVolatility[k]

        maxIndex = SharpeRatio.argmax()
        weight[maxIndex,:]



        def negativeSR(w):
            w = np.array(w)
            R = np.sum(mean_ret*w)
            V = np.sqrt(np.dot(w.T, np.dot(sigma,w)))
            SR = R/V
            return -1*SR
        def checkSumToOne(w):
            return np.sum(w)-1
            
        w0 = [1/length for i in range(length)]
        bounds = ((0,1) for i in range(length))
        constraints = ({"type":"eq", "fun":checkSumToOne})
        w_opt = minimize(negativeSR,w0,method="SLSQP",
                        bounds=bounds,constraints=constraints)
        
        
        plt.figure(figsize = (20,6))
        plt.scatter(expectedVolatility,expected_ret,c=SharpeRatio)
        plt.colorbar(label="SR")
        plt.scatter(expectedVolatility[maxIndex],expected_ret[maxIndex],c="red")
        plt.xlabel("Expected Volatility")
        plt.ylabel("Expected Return")


        w_opt.x
        list_from_array = w_opt.x.tolist()
        list_from_array= [round(elem, 2) for elem in list_from_array]
        opt_w = [str(elem*100) for elem in list_from_array]
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        optimized_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {"optimized": optimized_base64, "opt_w": opt_w, "stocklist": stocklist}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/lstm")
async def lstm(ticker: str = Form(...)):
    try:

        start = dt.datetime(2015,1,15)
        end = dt.datetime(2024,3,15)

        ticker = "AAPL"
        df = yf.download(ticker, start=start, end=end)

        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_data = scaler.fit_transform(df["Adj Close"].values.reshape(-1,1))

        prediction_days = 60

        X_train = []
        y_train = []

        for x in range(prediction_days, len(scaled_data)):
            X_train.append(scaled_data[x-prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

        model = Sequential()

        model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(units=50))
        model.add(Dropout(0.2))
        model.add(Dense(units=1)) 

        model.compile (optimizer="adam",
                    loss="mean_squared_error")
        model.fit(X_train,y_train,epochs=3, batch_size= 32)

        test_start = dt.datetime(2024,1,1)
        test_end = dt.datetime.now()

        ticker = "AAPL"
        df_test = yf.download(ticker, start=test_start, end=test_end)


        real_prices= df_test["Adj Close"].values
        df_final = pd.concat((df["Adj Close"], df_test["Adj Close"]), axis=0)

        model_inputs = df_final[len(df_final)-len(df_test)-prediction_days:].values
        model_inputs = model_inputs.reshape(-1,1)
        model_inputs = scaler.transform(model_inputs)

        X_test = []

        for x in range(prediction_days, len(model_inputs)):
            X_test.append(model_inputs[x-prediction_days:x, 0])
            

        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1],1))

        pre = model.predict(X_test)
        pre = scaler.inverse_transform(pre)

        plt.plot(real_prices, "r", label="real price")
        plt.plot(pre, label="predicted price")
        plt.legend()

        # Save the plot to a buffer and convert it to base64 string
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        lstm_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return {"lstm_graph": lstm_base64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
