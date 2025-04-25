from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np
app = Flask(__name__)
model_data = joblib.load('credit_model.pkl')
model = model_data['model']
preprocessor = model_data['preprocessor']
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        input_df = pd.DataFrame([[
            int(data['age']),
            int(data['job']),
            float(data['credit_amount']),
            int(data['duration']),
            float(data['credit_amount']) / max(1, int(data['duration'])),
            data['sex'],
            data['housing'],
            data['saving_accounts'],
            data['checking_account'],
            data['purpose'],
            pd.cut([int(data['age'])], bins=[18,30,45,60,100], labels=['18-30','30-45','45-60','60+'])[0]
        ]], columns=[
            'Age', 'Job', 'Credit amount', 'Duration', 'Monthly_Payment',
            'Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose', 'Age_Group'
        ])
        input_processed = preprocessor.transform(input_df)
        prediction = model.predict(input_processed)[0]
        probability = model.predict_proba(input_processed)[0][1]
        return jsonify({
            'prediction': 'Low Risk' if prediction == 1 else 'High Risk',
            'probability': f"{probability:.2%}",
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True)