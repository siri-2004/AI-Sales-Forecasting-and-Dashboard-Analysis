from flask import Flask, render_template, request, redirect, url_for, session
import pickle
import numpy as np
import plotly.express as px

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load models
with open('models/sales_model.pkl', 'rb') as f:
    sales_model = pickle.load(f)

with open('models/profit_model.pkl', 'rb') as f:
    profit_model = pickle.load(f)

with open('models/label_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

# Home Page (Login)
@app.route('/', methods=['GET', 'POST'])
def home():
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == 'admin' and password == 'admin':
            session['user'] = username
            return redirect(url_for('predict'))
        else:
            error = 'Invalid Credentials'
    return render_template('home.html', error=error)

# Prediction Page
@app.route('/predict', methods=['GET', 'POST'])
def predict():

    predicted_sales = None
    profit_prediction = None

    if request.method == 'POST':
        try:
            # Get user input
            category_name = request.form['category_name']
            customer_region = request.form['customer_region']
            shipping_type = request.form['shipping_type']
            order_quantity = int(request.form['order_quantity'])
            days_scheduled = int(request.form['days_for_shipment_scheduled'])

            # Encode inputs
            cat = encoders['category_name'].transform([category_name])[0]
            region = encoders['customer_region'].transform([customer_region])[0]
            shipping = encoders['shipping_type'].transform([shipping_type])[0]

            input_data = np.array([[cat, region, shipping, order_quantity, days_scheduled]])

            # Predict sales
            predicted_sales = round(sales_model.predict(input_data)[0], 2)

            # Predict profit class
            profit_class = profit_model.predict(input_data)[0]
            profit_prediction = "Profitable ✅" if profit_class == 1 else "Not Profitable ❌"
        except Exception as e:
            predicted_sales = None
            profit_prediction = f"Error: {str(e)}"

    return render_template('predict.html',
                           predicted_sales=predicted_sales,
                           profit_prediction=profit_prediction)

# Logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == "__main__":
    app.run(debug=True)
