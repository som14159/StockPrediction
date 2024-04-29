from flask import Flask, render_template, request
from alpha_vantage.timeseries import TimeSeries
import plotly
import plotly.express as px
import json
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from datetime import datetime
from newsapi import NewsApiClient
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime



app = Flask(__name__)

api_key = '4R2PZVTZEK2IBW58'

def get_news(stock):
    newsapi = NewsApiClient(api_key='d7f4f06cb623498587f38f5107c19bf5')
    all_articles = newsapi.get_everything(q=stock, language='en', sort_by = 'publishedAt')
    articles_data = []
    for article in all_articles['articles']:
        article_data = {
            'title': article['title'],
            'date': article['publishedAt'],
            'description': article['description'] if article['description'] is not None else 'N/A',
            'content': article['content'] if article['content'] is not None else 'N/A'
        }
        articles_data.append(article_data)
    file_path = "articles.json"
    with open(file_path, "w") as json_file:
        json.dump(articles_data, json_file, indent=4)

def predict_score():
    with open('articles.json', 'r') as file:
        data = json.load(file)
    current_date = datetime.today()
    total_score = 0
    relevance_factor = 0.85
    freq_sum = 0
    for obj in data:
        title = obj['title']
        publishedAt = obj['date']
        description = obj['description']
        content = obj['content']
        date = datetime.strptime(publishedAt[:10], "%Y-%m-%d")
        score = analyze_sentiment(title+description+content) * (relevance_factor ** (current_date - date).days)
        freq = (relevance_factor ** (current_date - date).days)
        freq_sum += freq
        print("Score:",score)
        total_score += score
    total_score = (1+total_score)/(1+freq_sum)
    print("Total Score:",total_score)
    s = {'Score:': total_score}
    with open("score.json",'w') as json_file:
        json.dump(s,json_file,indent = 4)
    return total_score

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment_scores = analyzer.polarity_scores(text)
    score = (sentiment_scores['compound'] + 1) * 2.5
    return score

def plot_stock_data(symbol):
    # ts = TimeSeries(key=api_key, output_format='pandas')
    # data, meta_data = ts.get_daily(symbol=symbol)
    # data.to_csv(symbol + 'DailyStockPrices.csv')
    data = pd.read_csv(symbol+'DailyStockPrices.csv')
    data['Close Price'] = data['4. close']
    data['Date'] = data.index
    fig = px.line(data, x='Date', y='Close Price', title=f'Daily Close Prices for {symbol}')
    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

def predict_price(symbol):
    df = pd.read_csv(symbol+'DailyStockPrices.csv')
    df['date'] = pd.to_datetime(df['date'])

    df.sort_values(by='date', inplace=True)

    # Extract the target variables
    y_open = df['1. open'].values.reshape(-1, 1)
    y_high = df['2. high'].values.reshape(-1, 1)
    y_close = df['4. close'].values.reshape(-1, 1)
    y_volume = df['5. volume'].values.reshape(-1, 1)

    # Normalize the target variables
    scaler_open = MinMaxScaler(feature_range=(0, 1))
    scaler_high = MinMaxScaler(feature_range=(0, 1))
    scaler_close = MinMaxScaler(feature_range=(0, 1))
    scaler_volume = MinMaxScaler(feature_range=(0, 1))
    y_open_scaled = scaler_open.fit_transform(y_open)
    y_high_scaled = scaler_high.fit_transform(y_high)
    y_close_scaled = scaler_close.fit_transform(y_close)
    y_volume_scaled = scaler_volume.fit_transform(y_volume)

    # Define a function to create the LSTM model
    def create_lstm_model(input_shape):
        model = Sequential()
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model


    # Function to prepare the data for training the LSTM model
    def prepare_data_for_lstm(df, window_size=60):
        X = []
        y = []
        for i in range(window_size, len(df)):
            X.append(df['date'].iloc[i-window_size:i].astype(int))  # Convert datetime to integer
            y.append(df['4. close'].iloc[i])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y


    # Prepare the data for training
    X_open, y_open = prepare_data_for_lstm(df)
    X_high, y_high = prepare_data_for_lstm(df)
    X_close, y_close = prepare_data_for_lstm(df)
    X_volume, y_volume = prepare_data_for_lstm(df)


    model_open = create_lstm_model(input_shape=(X_open.shape[1], 1))
    model_open.fit(X_open, y_open, epochs=10, batch_size=32)

    model_high = create_lstm_model(input_shape=(X_high.shape[1], 1))
    model_high.fit(X_high, y_high, epochs=10, batch_size=32)

    model_close = create_lstm_model(input_shape=(X_close.shape[1], 1))
    model_close.fit(X_close, y_close, epochs=10, batch_size=32)

    model_volume = create_lstm_model(input_shape=(X_volume.shape[1], 1))
    model_volume.fit(X_volume, y_volume, epochs=10, batch_size=32)


    def predict_stock_prices(model, scaler):
        current_date = datetime.now().strftime('%Y-%m-%d')
        X_input = np.array([pd.to_datetime(current_date).timestamp()]).reshape(1, 1, 1)
        prediction_scaled = model.predict(X_input)
        prediction = scaler.inverse_transform(prediction_scaled)
        return prediction[0][0]

    predicted_open = int(predict_stock_prices(model_open, scaler_open))
    predicted_high = int(predict_stock_prices(model_high, scaler_high))
    predicted_close = int(predict_stock_prices(model_close, scaler_close))
    predicted_volume = int(predict_stock_prices(model_volume, scaler_volume))

    prediction = [{
        'open':predicted_open,  
        'high':predicted_high,  
        'close':predicted_close,  
        'volume':predicted_volume,  
    }]

    with open("prediction.json",'w') as json_file:
        json.dump(prediction,json_file,indent = 4)
    
    return predicted_close,datetime.now().strftime('%Y-%m-%d')

def final_layer(sentiment_score,close,date,symbol):
        fl = {}
        try:
            print("fILE FOUND")
            with open(symbol+'.json', 'r') as file:
                fl = json.load(file)
            data = pd.read_csv(symbol+'DailyStockPrices.csv')
            print("fILE FOUND")
            # Then you can use loc to filter based on the datetime value
            data['date'] = pd.to_datetime(df['date'])
            data.set_index('date', inplace=True)
            def get_close_price(date_string):
                date = pd.to_datetime(date_string)
                try:
                    close_price = data.loc[date]['4. close']
                    return close_price
                except KeyError:
                    return "Date not found"

            date_string = "2024-02-15"
            close_price = get_close_price(date_string)
            print(f"The close price for {date_string} is: {close_price}")
            print("Close",true_close)
            w1 = fl['w1'] + ((true_close - fl['final_close']) * fl['score'] * 0.05)
            w2 = fl['w2'] + ((true_close - fl['final_close']) * fl['close'] * 0.05)
            final_close = w1 * sentiment_score + w2 * close
            print(w1+"hi"+w2)
            fl = {  
                'score':sentiment_score,
                'close':close,
                'final_close':final_close,
                'date':date,
                'w1':w1,
                'w2':w2,
            }
            with open(symbol+'.json','w') as json_file:
                json.dump(fl,json_file,indent = 4)
        except Exception as e :
            print(e)
            final_close = close + sentiment_score
            fl = {  
                'score':sentiment_score,
                'close':close,
                'final_close':final_close,
                'date':date,
                'w1':1,
                'w2':1,
            }
            with open(symbol+'.json','w') as json_file:
                json.dump(fl,json_file,indent = 4)

@app.route('/', methods=['GET', 'POST'])
def index():
    plot_data = None
    articles_data = []
    ts = 0
    if request.method == 'POST':
        symbol = request.form['symbols']
        plot_data = plot_stock_data(symbol)
        # get_news(symbol)
        ts = predict_score()
        lstm_close,date_now = predict_price(symbol)
        final_layer(ts,lstm_close,date_now,symbol)
        with open('articles.json', 'r') as file:
            articles_data = json.load(file)
    return render_template('index.html', plot_data=plot_data,articles_data=articles_data,score=ts,ss="Sentiment Score :")

if __name__ == '__main__':
    app.run(debug=True)
