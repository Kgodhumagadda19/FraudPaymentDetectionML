import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
from sklearn.ensemble import RandomForestClassifier
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def load_credit_card_data():
    """
    Load the credit card fraud dataset
    """
    print("LOADING CREDIT CARD FRAUD DATASET")
    print("=" * 50)
    
    try:
        # Load the CSV file we saved earlier
        df = pd.read_csv('fraud_dataset.csv')
        print(f"Dataset loaded: {df.shape}")
        
        # Separate features and target
        X = df.drop('Class', axis=1)  # All columns except 'Class'
        y = df['Class']  # Target variable
        
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Fraud rate: {y.mean():.4f} ({y.sum()} fraud cases)")
        
        return X, y
        
    except FileNotFoundError:
        print("Dataset file not found. Please run find_fraud_datasets.py first.")
        return None, None

def preprocess_data(X, y):
    """
    Preprocess the credit card data
    """
    print("\nPREPROCESSING DATA")
    print("=" * 30)
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Scale features (important for credit card data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Data preprocessed successfully")
    print(f"Features scaled: {X_scaled.shape[1]}")
    
    return X_scaled, y, scaler

def train_models(X, y):
    """
    Train multiple fraud detection models
    """
    print("\nTRAINING FRAUD DETECTION MODELS")
    print("=" * 40)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Fraud rate in training: {y_train.mean():.4f}")  # type: ignore
    print(f"Fraud rate in test: {y_test.mean():.4f}")  # type: ignore
    
    # Initialize models
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=-1
        ) if LGBMClassifier else None,
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        if model is None:
            print(f"\nSkipping {name} (not available)")
            continue
            
        print(f"\nTraining {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
    
    return results, X_test, y_test

def evaluate_models(results, X_test, y_test):
    """
    Evaluate and compare models
    """
    print("\nMODEL EVALUATION")
    print("=" * 30)
    
    best_model = None
    best_auc = 0
    
    for name, result in results.items():
        print(f"\n{name.upper()} RESULTS:")
        print("-" * 20)
        
        # Classification report
        print("Classification Report:")
        print(classification_report(y_test, result['predictions']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, result['predictions'])
        print(f"Confusion Matrix:")
        print(f"True Negatives: {cm[0,0]}")
        print(f"False Positives: {cm[0,1]}")
        print(f"False Negatives: {cm[1,0]}")
        print(f"True Positives: {cm[1,1]}")
        
        # Precision and recall for fraud detection
        precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
        recall = cm[1,1] / (cm[1,1] + cm[0,0]) if (cm[1,1] + cm[0,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {result['roc_auc']:.4f}")
        
        # Track best model
        if result['roc_auc'] > best_auc:
            best_auc = result['roc_auc']
            best_model = name
    
    print(f"\nBEST MODEL: {best_model} (ROC AUC: {best_auc:.4f})")
    return best_model

def save_model_and_results(results, best_model, scaler):
    """
    Save the best model and results
    """
    print(f"\nSAVING MODEL AND RESULTS")
    print("=" * 30)
    
    # Save best model
    best_model_obj = results[best_model]['model']
    model_data = {
        'model': best_model_obj,
        'scaler': scaler,
        'model_type': best_model,
        'training_date': datetime.now().isoformat(),
        'performance': {
            'accuracy': results[best_model]['accuracy'],
            'roc_auc': results[best_model]['roc_auc']
        }
    }
    
    joblib.dump(model_data, 'fraud_detection_model.pkl')
    print(f"Best model ({best_model}) saved to 'fraud_detection_model.pkl'")
    
    # Save results summary
    results_summary = {}
    for name, result in results.items():
        results_summary[name] = {
            'accuracy': result['accuracy'],
            'roc_auc': result['roc_auc']
        }
    
    with open('model_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("Results summary saved to 'model_results.json'")
    
    return best_model_obj

def test_model_prediction(model, scaler, X_test, y_test):
    """
    Test the model with sample predictions
    """
    print(f"\nTESTING MODEL PREDICTIONS")
    print("=" * 30)
    
    # Get a few test samples
    sample_indices = np.random.choice(len(X_test), 5, replace=False)
    sample_X = X_test.iloc[sample_indices]
    sample_y = y_test.iloc[sample_indices]
    
    # Scale the sample data
    sample_X_scaled = scaler.transform(sample_X)
    
    # Make predictions
    predictions = model.predict(sample_X_scaled)
    probabilities = model.predict_proba(sample_X_scaled)[:, 1]
    
    print("Sample Predictions:")
    for i, (idx, true_label, pred, prob) in enumerate(zip(sample_indices, sample_y, predictions, probabilities)):
        status = "✅ CORRECT" if pred == true_label else "❌ WRONG"
        fraud_status = "FRAUD" if true_label == 1 else "LEGITIMATE"
        print(f"Sample {i+1}: True={fraud_status}, Pred={pred}, Prob={prob:.4f} {status}")

def main():
    """
    Main function to train fraud detection model
    """
    print("CREDIT CARD FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Load data
    X, y = load_credit_card_data()
    if X is None:
        return
    
    # Preprocess data
    X_scaled, y, scaler = preprocess_data(X, y)
    
    # Train models
    results, X_test, y_test = train_models(X_scaled, y)
    
    # Evaluate models
    best_model_name = evaluate_models(results, X_test, y_test)
    
    # Save model and results
    best_model = save_model_and_results(results, best_model_name, scaler)
    
    # Test predictions
    test_model_prediction(best_model, scaler, X_test, y_test)
    
    print(f"\n🎉 FRAUD DETECTION MODEL TRAINING COMPLETE!")
    print(f"Best model: {best_model_name}")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"Model saved and ready for API deployment!")

if __name__ == "__main__":
    main() 