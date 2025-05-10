from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np  # <-- Added this
from datetime import datetime
import json

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and template data
model = joblib.load('restaurant_model.pkl')

# Define special dates
veg_special = pd.to_datetime([
    '2024-01-14', '2024-01-26', '2024-02-14', '2024-03-08', '2024-03-25',
    '2024-04-14', '2024-05-01', '2024-06-21', '2024-07-10', '2024-08-15',
    '2024-08-28', '2024-09-02', '2024-09-17', '2024-10-02', '2024-10-24',
    '2024-11-01', '2024-11-12', '2024-11-15', '2024-12-25', '2024-12-31'
])

nonveg_special = pd.to_datetime([
    '2024-01-10', '2024-02-20', '2024-03-11', '2024-04-10', '2024-05-18',
    '2024-06-16', '2024-07-14', '2024-08-12', '2024-09-20', '2024-11-20'
])

# Load and preprocess data once at startup
df = pd.read_excel("Transformed_Restaurant_Data.xlsx")
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek
df['veg_special'] = df['date'].isin(veg_special)
df['nonveg_special'] = df['date'].isin(nonveg_special)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=['item_name', 'day'], drop_first=True)

# Define X globally
X = df.drop(columns=['quantity_made', 'quantity_sold', 'quantity_surplus', 'date', 'establishmentId'])

# Helper function to predict quantities
def predict_all_quantities(date_str, df_template, model):
    date = pd.to_datetime(date_str)
    day_of_week = date.dayofweek
    is_veg_special = date in veg_special
    is_nonveg_special = date in nonveg_special

    input_data = {
        'day_of_week': day_of_week,
        'veg_special': int(is_veg_special),
        'nonveg_special': int(is_nonveg_special)
    }

    item_columns = [col for col in df_template.columns if col.startswith('item_name_')]
    dishes = [col.replace('item_name_', '') for col in item_columns]

    for col in df_template.columns:
        if col.startswith('item_name_'):
            input_data[col] = 1 if col == f'item_name_{dishes[0]}' else 0
        elif col.startswith('day_'):
            input_data[col] = 1 if col == f'day_{date.day_name()}' else 0
        elif col not in input_data:
            input_data[col] = 0

    predictions = {}

    for dish in dishes:
        input_data_copy = input_data.copy()
        for col in df_template.columns:
            if col.startswith('item_name_'):
                input_data_copy[col] = 1 if col == f'item_name_{dish}' else 0

        input_df = pd.DataFrame([input_data_copy])
        prediction = model.predict(input_df)[0]
        if is_veg_special or is_nonveg_special:
            prediction *= np.random.uniform(1.4, 1.5)  # <-- now works because np is imported
        predictions[dish] = int(round(prediction))

    return predictions


# API endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if 'date' not in data:
            return jsonify({'error': 'Missing date in request'}), 400
        
        date_str = data['date']
        predictions = predict_all_quantities(date_str, X, model)
        return jsonify(predictions)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)