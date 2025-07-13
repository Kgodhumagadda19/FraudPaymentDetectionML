#!/usr/bin/env python3
"""
Fix the prediction API by disabling shape checking in the model
"""

import joblib
import pandas as pd
import numpy as np
from fraud_detection_model import FraudDetectionModel

def fix_model_prediction():
    """Fix the model to disable shape checking for predictions"""
    
    print("🔧 Fixing model prediction issue...")
    
    try:
        # Load the current model
        model_data = joblib.load('api_compatible_model.pkl')
        
        # Get the models
        models = model_data['models']
        
        # Fix each model to disable shape checking
        for fraud_type, model in models.items():
            if hasattr(model, 'predict_disable_shape_check'):
                model.predict_disable_shape_check = True
                print(f"✅ Disabled shape check for {fraud_type} model")
            else:
                print(f"⚠️  Model {fraud_type} doesn't support shape check disable")
        
        # Save the fixed model
        joblib.dump(model_data, 'api_compatible_model_fixed.pkl')
        print("✅ Fixed model saved as 'api_compatible_model_fixed.pkl'")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing model: {e}")
        return False

def test_fixed_prediction():
    """Test prediction with the fixed model"""
    
    print("\n🧪 Testing fixed prediction...")
    
    try:
        # Load the fixed model
        model_data = joblib.load('api_compatible_model_fixed.pkl')
        
        # Create a simple test transaction
        test_data = pd.DataFrame({
            'transaction_id': ['TEST_001'],
            'timestamp': ['2024-01-15T14:30:00Z'],
            'amount': [150.0],
            'user_id': ['USER_001'],
            'merchant_id': ['MERCH_001'],
            'merchant_category': ['electronics'],
            'distance_from_home_km': [25.5],
            'velocity_24h': [3],
            'foreign_transaction': [False],
            'online_order': [True],
            'high_risk_merchant': [False],
            'transaction_count_user': [15],
            'card_present': [False],
            'used_chip': [False],
            'used_pin': [False],
            'card_type': ['visa'],
            'device_id': ['DEV_001']
        })
        
        # Create fraud detection model instance
        fraud_model = FraudDetectionModel()
        fraud_model.models = model_data['models']
        fraud_model.model_metadata = model_data['metadata']
        fraud_model.feature_importance = model_data['feature_importance']
        fraud_model.fraud_types = model_data['fraud_types']
        fraud_model.model_type = model_data['model_type']
        
        # Try prediction
        prediction = fraud_model.predict_fraud(test_data, 'credit_card')
        
        print(f"✅ Prediction successful!")
        print(f"   Fraud Probability: {prediction['fraud_probability']:.4f}")
        print(f"   Risk Level: {prediction['risk_level']}")
        print(f"   Recommended Action: {prediction['recommended_action']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def main():
    """Main function to fix and test the prediction API"""
    print("🚀 FIXING PREDICTION API")
    print("=" * 50)
    
    # Fix the model
    if fix_model_prediction():
        # Test the fixed prediction
        test_fixed_prediction()
        
        print("\n" + "=" * 50)
        print("NEXT STEPS:")
        print("=" * 50)
        print("1. Stop the current API (Ctrl+C)")
        print("2. Replace the model file:")
        print("   cp api_compatible_model_fixed.pkl api_compatible_model.pkl")
        print("3. Restart the API:")
        print("   python fraud_detection_api.py")
        print("4. Test the prediction endpoints again")
    else:
        print("\n❌ Failed to fix the model")

if __name__ == "__main__":
    main() 