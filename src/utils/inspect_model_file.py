import joblib

# Try both model files if you have them
for fname in ['enhanced_fraud_model.pkl', 'fraud_detection_model.pkl']:
    try:
        print(f"\nInspecting {fname} ...")
        model_data = joblib.load(fname)
        print("Type:", type(model_data))
        if isinstance(model_data, dict):
            print("Top-level keys:", list(model_data.keys()))
        else:
            print("Model is not a dict. It is:", type(model_data))
    except Exception as e:
        print(f"Could not load {fname}: {e}") 