from flask import Flask, render_template, request, redirect, session, url_for, flash, send_file, jsonify
from flask_mysqldb import MySQL
from datetime import datetime
from werkzeug.security import check_password_hash, generate_password_hash
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.optimizers import Adam
from datetime import timedelta
from openpyxl import load_workbook
import matplotlib.pyplot as plt
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from flask import send_file
from dateutil.relativedelta import relativedelta



app = Flask(__name__)


# Koneksi
app.secret_key = 'bebasapasaja'
app.config['MYSQL_HOST'] ='localhost'
app.config['MYSQL_USER'] ='root'
app.config['MYSQL_PASSWORD'] =''
app.config['MYSQL_DB'] ='dataakun'
mysql = MySQL(app)

# Index
@app.route('/')
def index():
    if 'loggedin' in session:
        return render_template('index.html')
    flash('Harap Login dulu', 'danger')
    return redirect(url_for('login'))


@app.route('/data', methods=['GET', 'POST'])
def data():
    # Membaca data dari file excel
    data = pd.read_excel('hi.xlsx')

    if request.method == 'POST':
        date = request.form['date']
        price = request.form['price']
        data.loc[len(data)] = [date, price]
        save_to_excel(data)

    # Mengambil 10 baris terakhir
    last_10_days = data.tail(10)

    # Mengubah data menjadi format chart.js
    chart_data = {
        'labels': last_10_days['date'].tolist(),
        'datasets': [{
            'label': 'Harga Ayam',
            'data': last_10_days['price'].tolist(),
            'backgroundColor': '#0079FF',
            'borderColor': '#0079FF',
            'borderWidth': 1
        }]
    }

    if 'loggedin' in session:
        return render_template('data.html', chart_data=chart_data, data=data.to_dict('records'))
    flash('Harap Login dulu', 'bahaya')
    return redirect(url_for('login'))


def save_to_excel(data):
    df = pd.DataFrame(data)
    df.to_excel('hi.xlsx', index=False)


@app.route('/hi')
def hi():
    filename = 'hi.xlsx'
    return send_file(filename, as_attachment=True)


@app.route('/history')
def history():
    filename = 'history.xlsx'
    return send_file(filename, as_attachment=True)


# Registrasi
@app.route('/registrasi', methods=('GET', 'POST'))
def registrasi():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        level = request.form['level']

        # Cek username atau email
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM tb_users WHERE username=%s OR email=%s', (username, email, ))
        akun = cursor.fetchone()
        if akun is None:
            cursor.execute('INSERT INTO tb_users VALUES (NULL, %s, %s, %s, %s)', (username, email, generate_password_hash(password), level))
            mysql.connection.commit()
            flash('Registrasi Berhasil', 'success')
        else:
            flash('Username atau email sudah ada', 'danger')
    return render_template('registrasi.html')


# Login
@app.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        # Cek data username
        cursor = mysql.connection.cursor()
        cursor.execute('SELECT * FROM tb_users WHERE email=%s', (email, ))
        akun = cursor.fetchone()
        if akun is None:
            flash('Login Gagal, Cek Username Anda', 'danger')
        elif not check_password_hash(akun[3], password):
            flash('Login gagal, Cek Password Anda', 'danger')
        else:
            session['loggedin'] = True
            session['username'] = akun[1]
            session['level'] = akun[4]
            return redirect(url_for('index'))
    return render_template('login.html')


# Logout
@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('username', None)
    session.pop('level', None)
    return redirect(url_for('login'))


@app.route('/prediksi')
def prediksi():
    if 'loggedin' in session:
        return render_template('prediksi.html')
    flash('Harap Login dulu', 'bahaya')
    return redirect(url_for('login'))


# Route untuk prediksi
@app.route('/predict', methods=['POST'])
def predict():
    prediction_days = int(request.form['predictionDays'])

    dataset = pd.read_excel('hi.xlsx')
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset.set_index('date', inplace=True)


    data = dataset['price'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))

    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]

    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    look_back = 7
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(units=50, input_shape=(1, look_back)))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer=Adam())
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    last_sequence = testX[-1]
    predicted_data = []
    for _ in range(prediction_days):
        prediction = model.predict(np.reshape(last_sequence, (1, 1, look_back)))
        predicted_data.append(scaler.inverse_transform(prediction)[0][0])
        last_sequence = np.append(np.reshape(last_sequence, (1, 1, look_back))[0][0][1:], prediction)

    
    

    predicted_data = [float(x) for x in predicted_data]

    dates = [dataset.index[-1] + timedelta(days=i) for i in range(prediction_days + 1)]
    prices = np.append(dataset['price'].values[-1], predicted_data)

    actual_dates = dataset.index[-prediction_days:]
    actual_prices = dataset['price'].values[-prediction_days:]


    mse = mean_squared_error(actual_prices, predicted_data)
    mape = np.mean(np.abs((actual_prices - predicted_data) / actual_prices)) * 100


    

    # Melakukan prediksi harga menggunakan data uji
    predicted = model.predict(testX)
    predicted = scaler.inverse_transform(predicted)

    # Visualisasi hasil prediksi
    plt.plot(scaler.inverse_transform(testY.reshape(-1, 1)), label='Data Asli')
    plt.plot(predicted, label='Prediksi')
    plt.legend()
    plt.show()


    # Persiapkan respons dengan informasi terbaru
    response = {
        'predictedData': [{'tanggal': str(date.date()), 'harga': price} for date, price in zip(dates, prices)],
        'actualData': [{'tanggal': str(date.date()), 'harga': float(price)} for date, price in zip(actual_dates, actual_prices)],
        'mse': mse,
        'mape': mape,
    }

    
    return jsonify(response)

# Route untuk prediksi
@app.route('/he', methods=['POST'])
def he():
    prediction_days = int(request.form['predictionDays'])
    start_date = pd.to_datetime(request.form['startDate'])
    end_date = pd.to_datetime(request.form['endDate'])

    

    dataset = pd.read_excel('hi.xlsx')
    dataset['date'] = pd.to_datetime(dataset['date'])
    dataset.set_index('date', inplace=True)

    dataset = dataset.loc[start_date:end_date]

    data = dataset['price'].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data.reshape(-1, 1))

    train_size = int(len(data) * 0.8)
    test_size = len(data) - train_size
    train_data, test_data = data[0:train_size, :], data[train_size:len(data), :]

    def create_dataset(dataset, look_back=1):
        dataX, dataY = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            dataX.append(a)
            dataY.append(dataset[i + look_back, 0])
        return np.array(dataX), np.array(dataY)

    look_back = 7
    trainX, trainY = create_dataset(train_data, look_back)
    testX, testY = create_dataset(test_data, look_back)

    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

    model = Sequential()
    model.add(LSTM(units=50, input_shape=(1, look_back)))
    model.add(Dense(units=1))
    model.compile(loss='mean_squared_error', optimizer=Adam())
    model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

    last_sequence = testX[-1]
    predicted_data = []
    for _ in range(prediction_days):
        prediction = model.predict(np.reshape(last_sequence, (1, 1, look_back)))
        predicted_data.append(scaler.inverse_transform(prediction)[0][0])
        last_sequence = np.append(np.reshape(last_sequence, (1, 1, look_back))[0][0][1:], prediction)

    
    

    predicted_data = [float(x) for x in predicted_data]

    dates = [dataset.index[-1] + timedelta(days=i) for i in range(prediction_days + 1)]
    prices = np.append(dataset['price'].values[-1], predicted_data)

    actual_dates = dataset.index[-prediction_days:]
    actual_prices = dataset['price'].values[-prediction_days:]


    mse = mean_squared_error(actual_prices, predicted_data)
    mape = np.mean(np.abs((actual_prices - predicted_data) / actual_prices)) * 100


    

    # Melakukan prediksi harga menggunakan data uji
    predicted = model.predict(testX)
    predicted = scaler.inverse_transform(predicted)


# Visualisasi hasil prediksi
    plt.plot(scaler.inverse_transform(testY.reshape(-1, 1)), label='Data Asli')
    plt.plot(predicted, label='Prediksi')
    plt.legend()
    plt.show()

    # Persiapkan respons dengan informasi terbaru
    response = {
        'predictedData': [{'tanggal': str(date.date()), 'harga': price} for date, price in zip(dates, prices)],
        'actualData': [{'tanggal': str(date.date()), 'harga': float(price)} for date, price in zip(actual_dates, actual_prices)],
        'mse': mse,
        'mape': mape,
    }

    
    return jsonify(response)

# Download history data
@app.route('/download')
def download():
    data = pd.read_excel('hi.xlsx')
    workbook = load_workbook('history.xlsx')
    writer = pd.ExcelWriter('history.xlsx', engine='openpyxl')
    writer.book = workbook

    # Append data to the existing sheet
    writer.sheets = dict((ws.title, ws) for ws in workbook.worksheets)
    data.to_excel(writer, index=False, header=False, startrow=len(data) + 1)

    # Save the Excel file
    writer.save()
    writer.close()

    return send_file('history.xlsx', as_attachment=True)


@app.route('/about')
def about():
    if 'loggedin' in session:
        return render_template('about.html')
    flash('Harap Login dulu', 'bahaya')
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True)
