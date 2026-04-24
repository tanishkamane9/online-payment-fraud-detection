from flask import Flask, render_template, request
import numpy as np
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load the pre-trained DecisionTreeClassifier model
model_path = os.path.join(os.path.dirname(__file__), 'fraud_detection_model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
@app.route("/home")
def about():
    """Renders the landing page."""
    return render_template('home.html')

@app.route("/predict")
def home1():
    """Renders the prediction form page."""
    return render_template('predict.html')

@app.route("/pred", methods=['POST', 'GET'])
def predict():
    """Handles the prediction logic."""
    if request.method == 'POST':
        try:
            # Extracting values from the form
            step = float(request.form.get('step', 0))
            # 'type_num' is 0–4 entered directly in the form (0=CASH_IN,1=CASH_OUT,2=DEBIT,3=PAYMENT,4=TRANSFER)
            type_val = int(request.form.get('type_num', 0))
            amount = float(request.form.get('amount', 0))
            oldbalanceOrg = float(request.form.get('oldbalanceOrg', 0))
            newbalanceOrig = float(request.form.get('newbalanceOrig', 0))
            oldbalanceDest = float(request.form.get('oldbalanceDest', 0))
            newbalanceDest = float(request.form.get('newbalanceDest', 0))
            
            # Preprocessing: Double Log Transformation for 'amount'
            amount_transformed = np.log1p(np.log1p(amount))
            
            # Constructing a DataFrame with proper column names to match training data
            feature_names = ['step', 'type', 'amount', 'oldbalanceOrg', 
                           'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
            features = pd.DataFrame([[
                step, 
                type_val, 
                amount_transformed, 
                oldbalanceOrg, 
                newbalanceOrig, 
                oldbalanceDest, 
                newbalanceDest
            ]], columns=feature_names)
            
            # Performing prediction
            if model:
                prediction = model.predict(features)
                pred_val = prediction[0]
                # Handle both string labels ('is Fraud'/'is not Fraud') and numeric (1/0)
                is_fraud = (pred_val == 'is Fraud') if isinstance(pred_val, str) else (pred_val == 1)
                result = "is Fraud" if is_fraud else "is not Fraud"
            else:
                result = "Model not loaded correctly."
            
            return render_template('submit.html', prediction_text=result)
            
        except Exception as e:
            return render_template('submit.html', prediction_text=f"An error occurred: {str(e)}")
            
    return render_template('predict.html')

if __name__ == "__main__":
    # Switching to False for production-like state as per screenshots, 
    # but using True for local development/debugging purposes.
    app.run(debug=True)
