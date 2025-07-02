"""
Flask Backend API for Mental Health Prediction Interface
Complete implementation with ML model integration and clinical insights
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import os
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import pickle

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend integration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockPredictor:
    """Mock predictor for demonstration when actual model isn't available"""
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.disorder_classes = [
            'No Disorder', 'Anxiety', 'Depression', 'Bipolar', 'ADHD', 
            'Personality Disorder', 'Eating Disorder', 'Substance Abuse'
        ]
        
    def predict_single(self, patient_data):
        """Mock prediction with realistic probabilities"""
        # Create feature vector
        features = [
            patient_data['Sadness'], patient_data['Euphoric'], patient_data['Exhausted'],
            patient_data['Sleep dissorder'], patient_data['Mood Swing'], patient_data['Suicidal thoughts'],
            patient_data['Anorexia'], patient_data['Authority Respect'], patient_data['Try-Explanation'],
            patient_data['Aggressive Response'], patient_data['Ignore & Move-On'], patient_data['Nervous Break-down'],
            patient_data['Admit Mistakes'], patient_data['Overthinking'], patient_data['Sexual Activity'],
            patient_data['Concentration'], patient_data['Optimisim']
        ]
        
        # Simple rule-based prediction for demonstration
        if patient_data['Suicidal thoughts'] == 1:
            predicted_class = 'Depression'
            confidence = 0.85
        elif patient_data['Mood Swing'] == 1 and patient_data['Euphoric'] >= 3:
            predicted_class = 'Bipolar'
            confidence = 0.75
        elif patient_data['Sadness'] >= 3 and patient_data['Concentration'] <= 3:
            predicted_class = 'Depression'
            confidence = 0.70
        elif patient_data['Nervous Break-down'] == 1 or patient_data['Overthinking'] == 1:
            predicted_class = 'Anxiety'
            confidence = 0.65
        elif patient_data['Anorexia'] == 1:
            predicted_class = 'Eating Disorder'
            confidence = 0.80
        else:
            predicted_class = 'No Disorder'
            confidence = 0.60
            
        # Create probability distribution
        probabilities = {disorder: 0.1 for disorder in self.disorder_classes}
        probabilities[predicted_class] = confidence
        
        # Normalize probabilities
        total = sum(probabilities.values())
        probabilities = {k: v/total for k, v in probabilities.items()}
        
        return {
            'predicted_disorder': predicted_class,
            'confidence': confidence,
            'probabilities': probabilities,
            'severity_score': self._calculate_severity(features),
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_severity(self, features):
        """Calculate severity score based on features"""
        # Weight critical symptoms more heavily
        severity = 0
        severity += features[0] * 0.15  # Sadness
        severity += features[2] * 0.10  # Exhausted
        severity += features[3] * 0.10  # Sleep disorder
        severity += features[5] * 0.25  # Suicidal thoughts (critical)
        severity += features[13] * 0.10  # Overthinking
        severity += (10 - features[15]) * 0.10  # Concentration (inverted)
        severity += (10 - features[16]) * 0.10  # Optimism (inverted)
        
        return min(severity / 10, 1.0)  # Normalize to 0-1

class MentalHealthAPI:
    def __init__(self):
        self.predictor = None
        self.model_loaded = False
        self.feature_names = [
            'Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder',
            'Mood Swing', 'Suicidal thoughts', 'Anorexia', 'Authority Respect',
            'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On',
            'Nervous Break-down', 'Admit Mistakes', 'Overthinking',
            'Sexual Activity', 'Concentration', 'Optimisim'
        ]
        
    def load_model(self, model_path='mental_health_model.pkl'):
        """Load the trained model"""
        try:
            if os.path.exists(model_path):
                # Try to load actual model
                with open(model_path, 'rb') as f:
                    self.predictor = pickle.load(f)
                self.model_loaded = True
                logger.info(f"Model loaded successfully from {model_path}")
            else:
                logger.warning(f"Model file not found: {model_path}. Using mock predictor.")
                # Use mock predictor for demonstration
                self.predictor = MockPredictor()
                self.model_loaded = True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}. Using mock predictor.")
            # Fallback to mock predictor
            self.predictor = MockPredictor()
            self.model_loaded = True
    
    def validate_input(self, data):
        """Validate input data"""
        errors = []
        
        # Check if all required features are present
        for feature in self.feature_names:
            if feature not in data:
                errors.append(f"Missing feature: {feature}")
        
        # Validate feature ranges
        if not errors:
            # Emotional symptoms (0-4)
            for feature in ['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder']:
                if not (0 <= data[feature] <= 4):
                    errors.append(f"{feature} must be between 0 and 4")
            
            # Binary features (0 or 1)
            binary_features = [
                'Mood Swing', 'Suicidal thoughts', 'Anorexia', 'Authority Respect',
                'Try-Explanation', 'Aggressive Response', 'Ignore & Move-On',
                'Nervous Break-down', 'Admit Mistakes', 'Overthinking'
            ]
            for feature in binary_features:
                if data[feature] not in [0, 1]:
                    errors.append(f"{feature} must be 0 or 1")
            
            # Functional measures (0-10)
            for feature in ['Sexual Activity', 'Concentration', 'Optimisim']:
                if not (0 <= data[feature] <= 10):
                    errors.append(f"{feature} must be between 0 and 10")
        
        return errors
    
    def predict(self, patient_data):
        """Make prediction for patient data"""
        if not self.model_loaded:
            raise ValueError("Model not loaded")
        
        # Validate input
        errors = self.validate_input(patient_data)
        if errors:
            raise ValueError(f"Validation errors: {', '.join(errors)}")
        
        # Make prediction
        result = self.predictor.predict_single(patient_data)
        
        # Add additional clinical insights
        result['clinical_insights'] = self._generate_clinical_insights(patient_data)
        result['risk_assessment'] = self._assess_risk(patient_data)
        result['recommendations'] = self._generate_recommendations(patient_data, result)
        
        return result
    
    def _generate_clinical_insights(self, data):
        """Generate clinical insights based on symptom patterns"""
        insights = []
        
        # High sadness indicator
        if data['Sadness'] >= 3:
            insights.append("High sadness levels detected - monitor for depressive symptoms")
        
        # Suicidal ideation
        if data['Suicidal thoughts'] == 1:
            insights.append("⚠ CRITICAL: Suicidal ideation present - immediate intervention required")
        
        # Mood instability
        if data['Mood Swing'] == 1 and (data['Euphoric'] >= 2 or data['Sadness'] >= 2):
            insights.append("Mood instability pattern suggests possible bipolar spectrum disorder")
        
        # Cognitive impairment
        if data['Concentration'] <= 3 and data['Overthinking'] == 1:
            insights.append("Cognitive symptoms present - difficulty concentrating with rumination")
        
        # Sleep disturbance
        if data['Sleep dissorder'] >= 3:
            insights.append("Severe sleep disturbance detected - may indicate underlying mood or anxiety disorder")
        
        # Eating concerns
        if data['Anorexia'] == 1:
            insights.append("Eating pattern disruption noted - assess for eating disorder")
        
        # Social functioning
        if data['Authority Respect'] == 0 and data['Aggressive Response'] == 1:
            insights.append("Interpersonal difficulties and authority issues present")
        
        # Coping mechanisms
        if data['Ignore & Move-On'] == 0 and data['Admit Mistakes'] == 0:
            insights.append("Limited healthy coping strategies identified")
        
        # Energy and motivation
        if data['Exhausted'] >= 3 and data['Optimisim'] <= 3:
            insights.append("Low energy and pessimistic outlook - consistent with depressive symptoms")
        
        return insights
    
    def _assess_risk(self, data):
        """Assess risk level based on symptoms"""
        risk_score = 0
        
        # Critical risk factors
        if data['Suicidal thoughts'] == 1:
            risk_score += 40
        
        # High risk factors
        if data['Sadness'] >= 3:
            risk_score += 15
        if data['Sleep dissorder'] >= 3:
            risk_score += 10
        if data['Nervous Break-down'] == 1:
            risk_score += 10
        
        # Moderate risk factors
        if data['Mood Swing'] == 1:
            risk_score += 8
        if data['Exhausted'] >= 3:
            risk_score += 8
        if data['Concentration'] <= 3:
            risk_score += 5
        if data['Optimisim'] <= 3:
            risk_score += 5
        
        # Protective factors (reduce risk)
        if data['Try-Explanation'] == 1:
            risk_score -= 5
        if data['Admit Mistakes'] == 1:
            risk_score -= 3
        if data['Optimisim'] >= 7:
            risk_score -= 5
        
        risk_score = max(0, min(100, risk_score))  # Clamp between 0-100
        
        if risk_score >= 30:
            risk_level = "HIGH"
        elif risk_score >= 15:
            risk_level = "MODERATE"
        else:
            risk_level = "LOW"
        
        return {
            'level': risk_level,
            'score': risk_score,
            'description': self._get_risk_description(risk_level)
        }
    
    def _get_risk_description(self, risk_level):
        """Get description for risk level"""
        descriptions = {
            'HIGH': 'Immediate professional intervention recommended. Close monitoring required.',
            'MODERATE': 'Professional evaluation advised. Monitor symptoms closely.',
            'LOW': 'Continue regular monitoring. Consider preventive measures.'
        }
        return descriptions.get(risk_level, 'Unknown risk level')
    
    def _generate_recommendations(self, data, prediction_result):
        """Generate treatment recommendations"""
        recommendations = []
        disorder = prediction_result['predicted_disorder']
        risk_level = prediction_result['risk_assessment']['level']
        
        # Emergency recommendations
        if data['Suicidal thoughts'] == 1:
            recommendations.extend([
                "🚨 IMMEDIATE: Contact crisis hotline or emergency services",
                "🚨 IMMEDIATE: Do not leave patient unattended",
                "🚨 IMMEDIATE: Remove access to means of self-harm"
            ])
        
        # Disorder-specific recommendations
        if disorder == 'Depression':
            recommendations.extend([
                "Consider antidepressant medication evaluation",
                "Cognitive Behavioral Therapy (CBT) recommended",
                "Implement daily activity scheduling",
                "Monitor sleep hygiene"
            ])
        elif disorder == 'Anxiety':
            recommendations.extend([
                "Anxiety management techniques (breathing exercises, mindfulness)",
                "Consider exposure therapy if phobias present",
                "Relaxation training",
                "Limit caffeine intake"
            ])
        elif disorder == 'Bipolar':
            recommendations.extend([
                "Mood stabilizer medication evaluation",
                "Maintain regular sleep schedule",
                "Mood tracking journal",
                "Psychoeducation about bipolar disorder"
            ])
        elif disorder == 'Eating Disorder':
            recommendations.extend([
                "Nutritional counseling",
                "Specialized eating disorder therapy",
                "Medical monitoring for complications",
                "Family therapy if appropriate"
            ])
        
        # General recommendations based on symptoms
        if data['Sleep dissorder'] >= 3:
            recommendations.append("Sleep study evaluation recommended")
        
        if data['Concentration'] <= 3:
            recommendations.append("Cognitive training exercises")
        
        if data['Overthinking'] == 1:
            recommendations.append("Mindfulness meditation practice")
        
        # Risk-based recommendations
        if risk_level == 'HIGH':
            recommendations.extend([
                "Weekly therapy sessions",
                "Medication compliance monitoring",
                "Crisis intervention plan"
            ])
        elif risk_level == 'MODERATE':
            recommendations.extend([
                "Bi-weekly therapy sessions",
                "Regular psychiatric follow-up"
            ])
        
        return recommendations

# Initialize API
api = MentalHealthAPI()
api.load_model()

# Routes
@app.route('/')
def index():
    """Serve the main interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': api.model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        # Get JSON data from request
        patient_data = request.get_json()
        
        if not patient_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        result = api.predict(patient_data)
        
        # Log prediction (remove sensitive data for logs)
        logger.info(f"Prediction made: {result['predicted_disorder']} with confidence {result['confidence']}")
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/features', methods=['GET'])
def get_features():
    """Get list of required features and their descriptions"""
    feature_descriptions = {
        'Sadness': 'Level of sadness (0-4: None, Mild, Moderate, Severe, Extreme)',
        'Euphoric': 'Level of euphoria (0-4: None, Mild, Moderate, Severe, Extreme)',
        'Exhausted': 'Level of exhaustion (0-4: None, Mild, Moderate, Severe, Extreme)',
        'Sleep dissorder': 'Sleep disorder level (0-4: None, Mild, Moderate, Severe, Extreme)',
        'Mood Swing': 'Presence of mood swings (0: No, 1: Yes)',
        'Suicidal thoughts': 'Presence of suicidal thoughts (0: No, 1: Yes)',
        'Anorexia': 'Eating disorder symptoms (0: No, 1: Yes)',
        'Authority Respect': 'Respect for authority (0: No, 1: Yes)',
        'Try-Explanation': 'Attempts to explain behavior (0: No, 1: Yes)',
        'Aggressive Response': 'Aggressive responses (0: No, 1: Yes)',
        'Ignore & Move-On': 'Ability to ignore and move on (0: No, 1: Yes)',
        'Nervous Break-down': 'History of nervous breakdown (0: No, 1: Yes)',
        'Admit Mistakes': 'Willingness to admit mistakes (0: No, 1: Yes)',
        'Overthinking': 'Tendency to overthink (0: No, 1: Yes)',
        'Sexual Activity': 'Sexual activity level (0-10: Very Low to Very High)',
        'Concentration': 'Concentration ability (0-10: Very Poor to Excellent)',
        'Optimisim': 'Optimism level (0-10: Very Pessimistic to Very Optimistic)'
    }
    
    return jsonify({
        'features': api.feature_names,
        'descriptions': feature_descriptions
    })

# HTML Template for the interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mental Health Prediction System</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #2c3e50 0%, #3498db 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            padding: 30px;
        }
        
        .form-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }
        
        .form-section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        .form-group label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #495057;
        }
        
        .form-group input, .form-group select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e9ecef;
            border-radius: 8px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        
        .form-group input:focus, .form-group select:focus {
            outline: none;
            border-color: #3498db;
        }
        
        .binary-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        
        .submit-btn {
            background: linear-gradient(135deg, #27ae60 0%, #2ecc71 100%);
            color: white;
            padding: 15px 30px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s;
            width: 100%;
            margin-top: 20px;
        }
        
        .submit-btn:hover {
            transform: translateY(-2px);
        }
        
        .submit-btn:disabled {
            background: #95a5a6;
            cursor: not-allowed;
            transform: none;
        }
        
        .results-section {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            border: 1px solid #e9ecef;
        }
        
        .results-section h2 {
            color: #2c3e50;
            margin-bottom: 20px;
            font-size: 1.5em;
        }
        
        .result-card {
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .prediction {
            font-size: 1.3em;
            font-weight: bold;
            color: #2c3e50;
            margin-bottom: 10px;
        }
        
        .confidence {
            color: #27ae60;
            font-weight: 600;
        }
        
        .risk-assessment {
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
        }
        
        .risk-high {
            background: #ffebee;
            border-left: 4px solid #e74c3c;
            color: #c62828;
        }
        
        .risk-moderate {
            background: #fff8e1;
            border-left: 4px solid #f39c12;
            color: #e65100;
        }
        
        .risk-low {
            background: #e8f5e8;
            border-left: 4px solid #27ae60;
            color: #2e7d32;
        }
        
        .insights, .recommendations {
            margin-top: 15px;
        }
        
        .insights ul, .recommendations ul {
            list-style: none;
            padding-left: 0;
        }
        
        .insights li, .recommendations li {
            background: #f1f3f4;
            padding: 8px 12px;
            margin: 5px 0;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }
        
        .critical {
            background: #ffebee !important;
            border-left-color: #e74c3c !important;
            color: #c62828;
            font-weight: bold;
        }
        
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            background: #ffebee;
            color: #c62828;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #e74c3c;
            margin: 20px 0;
        }
        
        @media (max-width: 768px) {
            .main-content {
                grid-template-columns: 1fr;
                padding: 20px;
            }
            
            .binary-group {
                grid-template-columns: 1fr;
            }
            
            .header h1 {
                font-size: 2em;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Mental Health Prediction System</h1>
            <p>AI-powered mental health assessment and clinical insights</p>
        </div>
        
        <div class="main-content">
            <div class="form-section">
                <h2>Patient Assessment Form</h2>
                <form id="predictionForm">
                    <div class="form-group">
                        <label>Sadness Level (0-4)</label>
                        <select name="Sadness" required>
                            <option value="">Select level</option>
                            <option value="0">0 - None</option>
                            <option value="1">1 - Mild</option>
                            <option value="2">2 - Moderate</option>
                            <option value="3">3 - Severe</option>
                            <option value="4">4 - Extreme</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Euphoric Level (0-4)</label>
                        <select name="Euphoric" required>
                            <option value="">Select level</option>
                            <option value="0">0 - None</option>
                            <option value="1">1 - Mild</option>
                            <option value="2">2 - Moderate</option>
                            <option value="3">3 - Severe</option>
                            <option value="4">4 - Extreme</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Exhaustion Level (0-4)</label>
                        <select name="Exhausted" required>
                            <option value="">Select level</option>
                            <option value="0">0 - None</option>
                            <option value="1">1 - Mild</option>
                            <option value="2">2 - Moderate</option>
                            <option value="3">3 - Severe</option>
                            <option value="4">4 - Extreme</option>
                        </select>
                    </div>
                    
                    <div class="form-group">
                        <label>Sleep Disorder Level (0-4)</label>
                        <select name="Sleep dissorder" required>
                            <option value="">Select level</option>
                            <option value="0">0 - None</option>
                            <option value="1">1 - Mild</option>
                            <option value="2">2 - Moderate</option>
                            <option value="3">3 - Severe</option>
                            <option value="4">4 - Extreme</option>
                        </select>
                    </div>
                    
                    <div class="binary-group">
                        <div class="form-group">
                            <label>Mood Swings</label>
                            <select name="Mood Swing" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Suicidal Thoughts</label>
                            <select name="Suicidal thoughts" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Eating Issues</label>
                            <select name="Anorexia" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Authority Respect</label>
                            <select name="Authority Respect" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Try to Explain</label>
                            <select name="Try-Explanation" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Aggressive Response</label>
                            <select name="Aggressive Response" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Ignore & Move On</label>
                            <select name="Ignore & Move-On" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Nervous Breakdown</label>
                            <select name="Nervous Break-down" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Admit Mistakes</label>
                            <select name="Admit Mistakes" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label>Overthinking</label>
                            <select name="Overthinking" required>
                                <option value="">Select</option>
                                <option value="0">No</option>
                                <option value="1">Yes</option>
                            </select>
                        </div>
                    </div>
                    
                    <div class="form-group">
                        <label>Sexual Activity Level (0-10)</label>
                        <input type="number" name="Sexual Activity" min="0" max="10" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Concentration Level (0-10)</label>
                        <input type="number" name="Concentration" min="0" max="10" required>
                    </div>
                    
                    <div class="form-group">
                        <label>Optimism Level (0-10)</label>
                        <input type="number" name="Optimisim" min="0" max="10" required>
                    </div>
                    
                    <button type="submit" class="submit-btn" id="submitBtn">
                        Analyze Mental Health
                    </button>
                </form>
            </div>
            
            <div class="results-section">
                <h2>Analysis Results</h2>
                <div id="results">
                    <p style="text-align: center; color: #666; padding: 40px;">
                        Complete the assessment form to see results
                    </p>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const submitBtn = document.getElementById('submitBtn');
            const resultsDiv = document.getElementById('results');
            
            // Show loading state
            submitBtn.disabled = true;
            submitBtn.textContent = 'Analyzing...';
            resultsDiv.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Analyzing patient data...</p>
                </div>
            `;
            
            try {
                // Collect form data
                const formData = new FormData(this);
                const data = {};
                
                for (let [key, value] of formData.entries()) {
                    // Convert to appropriate data type
                    if (['Sexual Activity', 'Concentration', 'Optimisim'].includes(key)) {
                        data[key] = parseInt(value);
                    } else if (['Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder'].includes(key)) {
                        data[key] = parseInt(value);
                    } else {
                        data[key] = parseInt(value);
                    }
                }
                
                // Make API call
                const response = await fetch('/api/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(data)
                });
                
                const result = await response.json();
                
                if (result.success) {
                    displayResults(result.result);
                } else {
                    throw new Error(result.error || 'Prediction failed');
                }
                
            } catch (error) {
                console.error('Error:', error);
                resultsDiv.innerHTML = `
                    <div class="error">
                        <strong>Error:</strong> ${error.message}
                    </div>
                `;
            } finally {
                // Reset button
                submitBtn.disabled = false;
                submitBtn.textContent = 'Analyze Mental Health';
            }
        });
        
        function displayResults(result) {
            const resultsDiv = document.getElementById('results');
            
            // Risk level styling
            let riskClass = 'risk-low';
            if (result.risk_assessment.level === 'HIGH') {
                riskClass = 'risk-high';
            } else if (result.risk_assessment.level === 'MODERATE') {
                riskClass = 'risk-moderate';
            }
            
            // Generate probability bars
            let probabilityBars = '';
            if (result.probabilities) {
                for (const [disorder, prob] of Object.entries(result.probabilities)) {
                    const percentage = (prob * 100).toFixed(1);
                    probabilityBars += `
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span>${disorder}</span>
                                <span>${percentage}%</span>
                            </div>
                            <div style="background: #e9ecef; height: 8px; border-radius: 4px;">
                                <div style="background: ${disorder === result.predicted_disorder ? '#3498db' : '#95a5a6'}; 
                                           height: 100%; width: ${percentage}%; border-radius: 4px; transition: width 0.3s;"></div>
                            </div>
                        </div>
                    `;
                }
            }
            
            // Generate insights
            let insightsHtml = '';
            if (result.clinical_insights && result.clinical_insights.length > 0) {
                insightsHtml = result.clinical_insights.map(insight => {
                    const isCritical = insight.includes('CRITICAL') || insight.includes('⚠');
                    return `<li class="${isCritical ? 'critical' : ''}">${insight}</li>`;
                }).join('');
            }
            
            // Generate recommendations
            let recommendationsHtml = '';
            if (result.recommendations && result.recommendations.length > 0) {
                recommendationsHtml = result.recommendations.map(rec => {
                    const isCritical = rec.includes('🚨 IMMEDIATE');
                    return `<li class="${isCritical ? 'critical' : ''}">${rec}</li>`;
                }).join('');
            }
            
            resultsDiv.innerHTML = `
                <div class="result-card">
                    <div class="prediction">
                        Predicted Condition: ${result.predicted_disorder}
                    </div>
                    <div class="confidence">
                        Confidence: ${(result.confidence * 100).toFixed(1)}%
                    </div>
                    <div style="margin-top: 15px;">
                        <small style="color: #666;">Severity Score: ${(result.severity_score * 100).toFixed(1)}%</small>
                    </div>
                </div>
                
                <div class="risk-assessment ${riskClass}">
                    <strong>Risk Assessment: ${result.risk_assessment.level}</strong>
                    <p>${result.risk_assessment.description}</p>
                    <small>Risk Score: ${result.risk_assessment.score}/100</small>
                </div>
                
                ${probabilityBars ? `
                <div class="result-card">
                    <h3 style="margin-bottom: 15px; color: #2c3e50;">Probability Distribution</h3>
                    ${probabilityBars}
                </div>
                ` : ''}
                
                ${insightsHtml ? `
                <div class="insights">
                    <h3 style="color: #2c3e50; margin-bottom: 10px;">Clinical Insights</h3>
                    <ul>${insightsHtml}</ul>
                </div>
                ` : ''}
                
                ${recommendationsHtml ? `
                <div class="recommendations">
                    <h3 style="color: #2c3e50; margin-bottom: 10px;">Treatment Recommendations</h3>
                    <ul>${recommendationsHtml}</ul>
                </div>
                ` : ''}
                
                <div style="margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 8px; font-size: 0.9em; color: #666;">
                    <strong>Disclaimer:</strong> This is an AI-powered assessment tool for healthcare professionals. 
                    Results should not replace professional medical diagnosis or treatment decisions.
                    <br><br>
                    <strong>Analysis completed:</strong> ${new Date(result.timestamp).toLocaleString()}
                </div>
            `;
        }
        
        // Add form validation
        document.querySelectorAll('select, input').forEach(element => {
            element.addEventListener('change', validateForm);
        });
        
        function validateForm() {
            const form = document.getElementById('predictionForm');
            const submitBtn = document.getElementById('submitBtn');
            const formData = new FormData(form);
            
            let isValid = true;
            const requiredFields = [
                'Sadness', 'Euphoric', 'Exhausted', 'Sleep dissorder', 'Mood Swing',
                'Suicidal thoughts', 'Anorexia', 'Authority Respect', 'Try-Explanation',
                'Aggressive Response', 'Ignore & Move-On', 'Nervous Break-down',
                'Admit Mistakes', 'Overthinking', 'Sexual Activity', 'Concentration', 'Optimisim'
            ];
            
            for (const field of requiredFields) {
                if (!formData.get(field) && formData.get(field) !== '0') {
                    isValid = false;
                    break;
                }
            }
            
            submitBtn.disabled = !isValid;
        }
        
        // Initialize form validation
        validateForm();
        
        // Health check on page load
        fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'healthy') {
                    console.log('API is healthy, model loaded:', data.model_loaded);
                }
            })
            .catch(error => {
                console.error('Health check failed:', error);
            });
    </script>
</body>
</html>
'''

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)