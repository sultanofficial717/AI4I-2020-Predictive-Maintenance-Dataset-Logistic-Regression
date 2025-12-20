"""
Machine Failure Prediction Web Application
Flask backend for predictive maintenance
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Train and save model if not exists
def train_model():
    print("Training model...")
    df = pd.read_csv('ai4i2020.csv')
    df_dropped = df.drop(['UDI', 'Product ID', 'Type'], axis=1)
    
    # Prepare features
    x = df_dropped.drop(['Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], axis=1)
    y = df_dropped['Machine failure']
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)
    
    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("Model trained and saved!")
    return model, scaler

# Load or train model
if os.path.exists('model.pkl') and os.path.exists('scaler.pkl'):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    print("Model loaded from file")
else:
    model, scaler = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        data = request.json
        
        air_temp = float(data['air_temp'])
        process_temp = float(data['process_temp'])
        rotational_speed = float(data['rotational_speed'])
        torque = float(data['torque'])
        tool_wear = float(data['tool_wear'])
        
        # Create dataframe
        input_df = pd.DataFrame({
            'Air temperature [K]': [air_temp],
            'Process temperature [K]': [process_temp],
            'Rotational speed [rpm]': [rotational_speed],
            'Torque [Nm]': [torque],
            'Tool wear [min]': [tool_wear]
        })
        
        # Scale input
        input_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Determine risk level
        failure_prob = probability[1] * 100
        
        if failure_prob < 10:
            risk_level = "LOW"
            risk_color = "success"
            message = "Machine is operating within normal parameters. Low risk of failure."
        elif failure_prob < 50:
            risk_level = "MODERATE"
            risk_color = "warning"
            message = "Some parameters are approaching critical levels. Monitor closely."
        else:
            risk_level = "HIGH"
            risk_color = "danger"
            message = "Machine failure is likely! Immediate maintenance recommended."
        
        return jsonify({
            'success': True,
            'prediction': int(prediction),
            'failure_probability': round(failure_prob, 2),
            'normal_probability': round(probability[0] * 100, 2),
            'risk_level': risk_level,
            'risk_color': risk_color,
            'message': message
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/statistics')
def statistics():
    """Get dataset statistics for reference"""
    df = pd.read_csv('ai4i2020.csv')
    
    stats = {
        'air_temp': {
            'min': float(df['Air temperature [K]'].min()),
            'max': float(df['Air temperature [K]'].max()),
            'mean': float(df['Air temperature [K]'].mean())
        },
        'process_temp': {
            'min': float(df['Process temperature [K]'].min()),
            'max': float(df['Process temperature [K]'].max()),
            'mean': float(df['Process temperature [K]'].mean())
        },
        'rotational_speed': {
            'min': float(df['Rotational speed [rpm]'].min()),
            'max': float(df['Rotational speed [rpm]'].max()),
            'mean': float(df['Rotational speed [rpm]'].mean())
        },
        'torque': {
            'min': float(df['Torque [Nm]'].min()),
            'max': float(df['Torque [Nm]'].max()),
            'mean': float(df['Torque [Nm]'].mean())
        },
        'tool_wear': {
            'min': float(df['Tool wear [min]'].min()),
            'max': float(df['Tool wear [min]'].max()),
            'mean': float(df['Tool wear [min]'].mean())
        }
    }
    
    return jsonify(stats)

@app.route('/visualizations/<path:filename>')
def serve_visualization(filename):
    """Serve visualization images"""
    return send_from_directory('visualizations', filename)

@app.route('/api/graphs')
def get_graphs():
    """Get list of available visualization graphs"""
    graphs = [
        {'id': '01_air_temperature_boxplot.png', 'name': 'Air Temperature Box Plot', 'description': 'Distribution of air temperature readings with outlier detection'},
        {'id': '02_process_temperature_boxplot.png', 'name': 'Process Temperature Box Plot', 'description': 'Distribution of process temperature during operations'},
        {'id': '03_rotational_speed_boxplot.png', 'name': 'Rotational Speed Box Plot', 'description': 'Machine rotational speed distribution and outliers'},
        {'id': '04_torque_boxplot.png', 'name': 'Torque Box Plot', 'description': 'Torque distribution and mechanical stress analysis'},
        {'id': '05_tool_wear_boxplot.png', 'name': 'Tool Wear Box Plot', 'description': 'Tool wear patterns and usage time analysis'},
        {'id': '06_correlation_outliers.png', 'name': 'Correlation Matrix (Outliers)', 'description': 'Feature correlations highlighting outlier patterns'},
        {'id': '07_correlation_all_features.png', 'name': 'Correlation Matrix (All Features)', 'description': 'Complete correlation analysis of all features'},
        {'id': '08_distributions_before_scaling.png', 'name': 'Distributions (Before Scaling)', 'description': 'Feature distributions before standardization'},
        {'id': '09_distributions_after_scaling.png', 'name': 'Distributions (After Scaling)', 'description': 'Feature distributions after standardization'},
        {'id': '10_roc_curve.png', 'name': 'ROC Curve', 'description': 'Model performance with AUC score and random classifier baseline'}
    ]
    return jsonify(graphs)

if __name__ == '__main__':
    print("Starting Machine Failure Prediction Web App...")
    print("Navigate to: http://localhost:7000")
    # Running on port 6000
    app.run(debug=True, port=7000)
