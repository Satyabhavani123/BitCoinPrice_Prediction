from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
import os
import matplotlib.pyplot as plt


# Initialize the Flask application
app = Flask(__name__)

# Function to preprocess data (remove commas and convert to float)
def preprocess_data(df):
    df['Price'] = df['Price'].str.replace(',', '').astype(float)
    return df

# Function to predict the daily price and calculate error rate
def predict_daily_price(df, current_price, days):
    current_price = float(current_price)
    days = int(days)
    
    df['Prediction'] = df['Price'].shift(-days)
    df.dropna(subset=['Prediction'], inplace=True)
    
    X = df[['Price']].values
    y = df['Prediction'].values
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(x_train, y_train)
    
    predicted_price = knn_model.predict([[current_price]])
    
    y_pred = knn_model.predict(x_test)
    error = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return round(predicted_price[0], 2), round(error, 2), df

# Function to predict the monthly price and calculate error rate
def predict_monthly_price(df, year):
    df['Date'] = pd.to_datetime(df['Date'])
    df_year = df[df['Date'].dt.year == year]
    
    if len(df_year) < 2:
        return None, None, df
    
    df_year['Prediction'] = df_year['Price'].shift(-30)
    df_year.dropna(subset=['Prediction'], inplace=True)
    
    X = df_year[['Price']].values
    y = df_year['Prediction'].values
    
    if len(X) < 5:
        return None, None, df_year
    
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model.fit(x_train, y_train)
    
    predicted_prices = knn_model.predict(x_test)
    mean_predicted_price = round(np.mean(predicted_prices), 2)
    
    error = np.sqrt(mean_squared_error(y_test, predicted_prices))
    
    return mean_predicted_price, round(error, 2), df_year

# Route for the home page (index page)
@app.route('/')
def index():
    return render_template('index.html')

# Route for handling daily and monthly predictions
@app.route('/home', methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Handle daily prediction
        if 'dPrice' in request.form and 'days' in request.form:
            df = pd.read_csv('bit_coin_price_data_set.csv')
            df = preprocess_data(df)
            
            current_price = request.form.get("dPrice")
            days = request.form.get("days")
            
            predicted_price, error_rate, df = predict_daily_price(df, current_price, days)
            return render_template('result.html', data=predicted_price, error=error_rate, df=df.to_json())
        
        # Handle monthly prediction
        elif 'file' in request.files and 'year' in request.form:
            file = request.files['file']
            year = int(request.form.get('year'))
            
            df = pd.read_csv(file)
            df = preprocess_data(df)
            
            predicted_price, error_rate, df_year = predict_monthly_price(df, year)
            
            if predicted_price is None:
                return render_template('result.html', data="Insufficient data for prediction", error="N/A", df=None)
            
            return render_template('result.html', data=predicted_price, error=error_rate, df=df_year.to_json())
    
    return render_template('home.html')

# Route for the graph display
@app.route('/graph', methods=["POST"])
def graph():
    df_json = request.form.get('df')
    df = pd.read_json(df_json)
    
    plt.figure()
    plt.plot(df['Price'], label='Current Price')
    plt.plot(df['Prediction'], label='Predicted Price')
    plt.xlabel('Days')
    plt.ylabel('Price')
    plt.title('Price Prediction Graph')
    plt.legend()
    
    image_path = os.path.join('static', 'price_prediction.png')
    plt.savefig(image_path)
    
    return render_template('graph.html', image=image_path)
# Route for the about page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the contact page
@app.route('/contact', methods=["GET", "POST"])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Handle the received message
        # (Here, you can store the message or send an email as required)
        
        return redirect(url_for('home'))
    
    return render_template('contactus.html')

# Route for user signup
@app.route('/signup', methods=["GET", "POST"])
def signup():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Handle user registration
        # (Here, you can store the user's information in a database)
        
        return redirect(url_for('home'))
    
    return render_template('signup.html')

# Route for user login
@app.route('/login', methods=["GET", "POST"])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Handle user login
        # (Here, you can verify the user's credentials)
        
        return redirect(url_for('home'))
    
    return render_template('login.html')

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
