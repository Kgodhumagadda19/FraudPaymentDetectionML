#!/usr/bin/env python3
"""
Convert enhanced fraud model to API-compatible format
"""

import joblib
import pandas as pd
from datetime import datetime
from src.models.fraud_detection_model import FraudDetectionModel

def convert_enhanced_model():
    """Convert enhanced model to API-compatible format"""
    print("Converting enhanced model to API-compatible format...")
    
    # Load the enhanced model
    enhanced_model_data = joblib.load('enhanced_fraud_model.pkl')
    
    # Create a new FraudDetectionModel instance
    fraud_model = FraudDetectionModel(model_type='lightgbm')
    
    # Extract the model and metadata
    model = enhanced_model_data['model']
    scaler = enhanced_model_data['scaler']
    label_encoders = enhanced_model_data['label_encoders']
    performance = enhanced_model_data['performance']
    feature_columns = enhanced_model_data['feature_columns']
    
    # Store the model in the expected format
    fraud_model.models['credit_card'] = model
    fraud_model.model_metadata['credit_card'] = {
        'training_date': enhanced_model_data['training_date'],
        'model_type': enhanced_model_data['model_type'],
        'features_used': feature_columns,
        'training_samples': 80000,  # Approximate
        'test_samples': 20000,      # Approximate
        'accuracy': performance['accuracy'],
        'roc_auc': performance['roc_auc'],
        'fraud_rate': 0.15  # Approximate
    }
    
    # Store additional components
    fraud_model.scalers['credit_card'] = scaler
    fraud_model.label_encoders = label_encoders
    fraud_model.feature_columns = feature_columns
    
    # Save in the expected format
    model_data = {
        'models': fraud_model.models,
        'metadata': fraud_model.model_metadata,
        'feature_importance': fraud_model.feature_importance,
        'fraud_types': fraud_model.fraud_types,
        'model_type': fraud_model.model_type,
        'scalers': fraud_model.scalers,
        'label_encoders': fraud_model.label_encoders,
        'feature_columns': fraud_model.feature_columns
    }
    
    joblib.dump(model_data, 'api_compatible_model.pkl')
    print("✅ Enhanced model converted and saved as 'api_compatible_model.pkl'")
    
    return fraud_model

if __name__ == "__main__":
    convert_enhanced_model() 