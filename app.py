import os
from flask import Flask, render_template, send_file, request
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error
import csv
import codecs

app = Flask(__name__)
df = pd.read_csv("bit_coin_price_data_set.csv")
df.drop(['Date', 'Open', 'High', 'Low', 'Vol.', 'Change %'], axis=1, inplace=True)
df = df.dropna()
df['Price'] = df['Price'].str.replace(',', '').astype(float)

# Display the updated DataFrame
print(df.head())

def testing(input_data, prediction_days, df):
    df['Prediction'] = df['Price'].shift(-int(prediction_days))
    X = np.array(df.drop(['Prediction']))
    y = np.array(df['Price'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
    k_neighbors = 9
    model = KNeighborsClassifier(n_neighbors=k_neighbors)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(pred)
    error = np.sqrt(mean_squared_error(y_test, pred))
    print("error",error)
    return [pred[[input_data]], error]

@app.route('/home', methods=["GET", "POST"])
def home():
    days = request.form.get("days")
    if request.method == "POST":
        radioRes = request.form.get("priceRadio")
        if radioRes == 'day_price':
            p = request.form['dPrice']
            print("day One",p)
            data = testing(p, days, df)
            print(data)
            error = round(data[1], 2)
            data = round(data[0][0], 2)
            return render_template('result.html', data=data, error=error)
        elif radioRes == 'month_Price':
            print("dealing csv")
            monthData = []
            file = request.files['mPrice']
            if not os.path.isdir('static'):
                os.mkdir('static')
            filepath = os.path.join('static', file.filename)
            file.save(filepath)
            df['PredictedPrice'] = 0
            j = 0
            print("going into for loop")
            csvreader = csv.reader(codecs.iterdecode(file, 'utf-8'))
            error = 0
            for i in csvreader:
                if j == 0:
                    j += 1
                    continue
                print("testing", i)
                v = testing(i[0], days, df)
                print("hellop")
                print(v)
                error = v[1]
                v = round(v[0][0], 2)
                df.loc[j - 1, 'PredictedPrice'] = v
                j += 1
                monthData.append(v)
            error = round(error, 2)
            print(df.head())
            return render_template("result.html", data=error)
        df.to_csv("AllDetails.csv", index=False)
        print("last=", error)
    return render_template("index.html")

@app.route('/download', methods=["GET", "POST"])
def download():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Download File':
            print("downloading......")
            path = os.path.join("static", "AllDetails.csv")
            return send_file(path, as_attachment=True)
        else:
            print("Generating Graph...")
            data = pd.read_csv('AllDetails.csv')
            plt.plot(data['PredictedPrice'], label="predicted")
            plt.plot(data['Price'], label="current")
            plt.xlabel('Days')
            plt.ylabel('Price')
            plt.title('Price Prediction for the entered number of days')
            print("done")
            plt.legend(['predicted', 'current'])
            imagePath = os.path.join('static', 'image' + '.png')
            plt.savefig(imagePath)
            return render_template('graph.html', image=imagePath)
    return render_template('index.html')
    
if __name__ == "__main__":
    app.run(debug=True)
