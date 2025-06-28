# app.py
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('churn_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    credit_score = int(request.form['credit_score'])
    gender = request.form['gender']
    age = int(request.form['age'])
    tenure = int(request.form['tenure'])
    balance = float(request.form['balance'])
    num_of_products = int(request.form['num_of_products'])
    has_cr_card = int(request.form['has_cr_card'])
    is_active_member = int(request.form['is_active_member'])
    estimated_salary = float(request.form['estimated_salary'])
    geography = request.form['geography']

    # Encode gender
    gender_encoded = 1 if gender.lower() == 'male' else 0

    # Encode geography
    geo_germany = 0
    geo_spain = 0
    if geography.lower() == 'germany':
        geo_germany = 1
    elif geography.lower() == 'spain':
        geo_spain = 1

    # Create input array
    user_input = np.array([[credit_score, gender_encoded, age, tenure, balance, num_of_products,
                            has_cr_card, is_active_member, estimated_salary, geo_germany, geo_spain]])

    # Scale input
    user_input_scaled = scaler.transform(user_input)

    # Predict
    prediction = model.predict(user_input_scaled)
    prediction_text = 'Churn' if prediction == 1 else 'No Churn'

    return render_template('index.html', prediction_text=f'Prediction: {prediction_text}')

if __name__ == "__main__":
    app.run(debug=True)
