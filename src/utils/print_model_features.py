import joblib

model_data = joblib.load('api_compatible_model.pkl')
if hasattr(model_data, 'model_metadata') and 'credit_card' in model_data.model_metadata:
    features = model_data.model_metadata['credit_card'].get('features_used')
    if features:
        print("Features used for training (credit_card):")
        for i, feat in enumerate(features):
            print(f"{i+1:2d}. {feat}")
    else:
        print("No feature list found in model metadata.")
elif isinstance(model_data, dict) and 'model_metadata' in model_data and 'credit_card' in model_data['model_metadata']:
    features = model_data['model_metadata']['credit_card'].get('features_used')
    if features:
        print("Features used for training (credit_card):")
        for i, feat in enumerate(features):
            print(f"{i+1:2d}. {feat}")
    else:
        print("No feature list found in model metadata.")
else:
    print("Model metadata or credit_card entry not found.") 