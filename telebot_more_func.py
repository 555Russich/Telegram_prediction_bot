import datetime
import time
import telebot
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

# API telegram
my_token = '1831630337:AAE8ePa4y7S6wS9W3tVEWdaLhBIuqYDE0ZU'
bot = telebot.TeleBot(my_token)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, '''
Write a company name like in finance.yahoo.com and wait a little bit
For example: 'AAPL', 'FB', 'SBER.ME' or 'TATN.ME'
    ''')


def save_graph(actual_prices, predicted_prices, company):
    plt.plot(actual_prices, color='black', label=f'actual {company} price')
    plt.plot(predicted_prices, color='red', label=f'predicted {company} price')
    plt.title(f'{company} Share Price')
    plt.xlabel('Time')
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches((25, 15), forward=False)  # Масштабируем график для сохранения
    fig.savefig('graph.jpg', dpi=250)
    fig.clf()


@bot.message_handler(content_types='text')
def predict_stock_close_price(message):
    company = message.text
    start = dt.datetime.now() - dt.timedelta(weeks=156)
    end = dt.datetime.now()
    start_time_of_script = time.time()
    try:
        data = web.DataReader(company, 'yahoo', start, end)
        # Prepare Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

        prediction_days = 60

        x_train = []
        y_train = []

        for x in range(prediction_days, len(scaled_data)):
            x_train.append(scaled_data[x - prediction_days:x, 0])
            y_train.append(scaled_data[x, 0])

        x_train, y_train = np.array(x_train), np.array(y_train)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        number_of_prediction = 3
        prediction_list = []
        for n in range(0, number_of_prediction):

            # Build The Model
            model = Sequential()

            model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=50))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))  # Prediction of the next closing value

            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=25, batch_size=32)

            ''' Test the model Accuracy on Existing Data '''

            # Load test data

            test_start = dt.datetime.now() - dt.timedelta(weeks=156)
            test_end = dt.datetime.now()

            test_data = web.DataReader(company, 'yahoo', test_start, test_end)
            actual_prices = test_data['Close'].values

            total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

            model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
            model_inputs = model_inputs.reshape(-1, 1)
            model_inputs = scaler.transform(model_inputs)

            # Make predictions on test data

            x_test = []

            for x in range(prediction_days, len(model_inputs)):
                x_test.append(model_inputs[x - prediction_days:x, 0])

            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            predicted_prices = model.predict(x_test)
            predicted_prices = scaler.inverse_transform(predicted_prices)

            # Predict next day

            real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs + 1), 0]]
            real_data = np.array(real_data)
            real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

            prediction = model.predict(real_data)
            prediction = scaler.inverse_transform(prediction)
            prediction_list.append(prediction)
            bot.reply_to(message, f'{n+1} prediction of {number_of_prediction}: {prediction}')

        prediction_average = sum(prediction_list) / len(prediction_list)
        bot.reply_to(message, f'Average price of predictions {prediction_average}')
        delta_time = time.time() - start_time_of_script
        bot.reply_to(message, delta_time)

        try:
            save_graph(actual_prices, predicted_prices, company)
            photo_graph = open('graph.jpg', 'rb')
            bot.send_photo(message.chat.id, photo_graph)
            photo_graph.close()
        except Exception:
            bot.reply_to(message, 'Can not send graph :(')
    except Exception:
        bot.reply_to(message, 'Wrong company name (Probably not like in Yahoo)')


bot.polling()
