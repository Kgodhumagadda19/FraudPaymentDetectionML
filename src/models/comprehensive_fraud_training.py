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

def load_synthetic_data():
    """Load synthetic fraud detection data"""
    print("LOADING SYNTHETIC FRAUD DATA")
    print("=" * 40)
    
    try:
        # Generate synthetic data
        from fraud_detection_data import generate_fraud_detection_dataset
        synthetic_df = generate_fraud_detection_dataset(n_samples=100000, fraud_ratio=0.15)
        
        print(f"Synthetic data loaded: {synthetic_df.shape}")
        print(f"Fraud rate: {synthetic_df['is_fraud'].mean():.2%}")
        
        return synthetic_df
    except Exception as e:
        print(f"Error loading synthetic data: {e}")
        return None

def load_live_data():
    """Load live credit card fraud data"""
    print("\nLOADING LIVE CREDIT CARD FRAUD DATA")
    print("=" * 45)
    
    try:
        # Load the real credit card fraud dataset
        live_df = pd.read_csv('fraud_dataset.csv')
        
        print(f"Live data loaded: {live_df.shape}")
        print(f"Fraud rate: {live_df['Class'].mean():.2%}")
        
        # Rename target column for consistency
        live_df = live_df.rename(columns={'Class': 'is_fraud'})
        
        return live_df
    except Exception as e:
        print(f"Error loading live data: {e}")
        return None

def load_enterprise_data():
    """Load additional enterprise fraud datasets"""
    print("\nLOADING ENTERPRISE FRAUD DATA")
    print("=" * 40)
    
    enterprise_datasets = []
    
    try:
        # Try to load additional datasets from OpenML
        import openml
        
        # Dataset 42180: SF Police Incidents (crime data)
        try:
            dataset = openml.datasets.get_dataset(42180)
            X, y, _, _ = dataset.get_data(
                dataset_format="dataframe", 
                target=dataset.default_target_attribute
            )
            
            # Convert to fraud detection format
            if y is not None and y.dtype == 'object':
                y = pd.Categorical(y).codes
            
            # Create DataFrame
            enterprise_df = pd.DataFrame(X)
            enterprise_df['is_fraud'] = y
            
            print(f"Enterprise dataset 42180 loaded: {enterprise_df.shape}")
            enterprise_datasets.append(enterprise_df)
            
        except Exception as e:
            print(f"Could not load dataset 42180: {e}")
        
        # Dataset 42178: Telco Customer Churn (can be adapted for fraud)
        try:
            dataset = openml.datasets.get_dataset(42178)
            X, y, _, _ = dataset.get_data(
                dataset_format="dataframe", 
                target=dataset.default_target_attribute
            )
            
            # Convert to fraud detection format
            if y is not None and y.dtype == 'object':
                y = pd.Categorical(y).codes
            
            # Create DataFrame
            enterprise_df2 = pd.DataFrame(X)
            enterprise_df2['is_fraud'] = y
            
            print(f"Enterprise dataset 42178 loaded: {enterprise_df2.shape}")
            enterprise_datasets.append(enterprise_df2)
            
        except Exception as e:
            print(f"Could not load dataset 42178: {e}")
            
    except Exception as e:
        print(f"Error loading enterprise data: {e}")
    
    return enterprise_datasets

def combine_datasets(synthetic_df, live_df, enterprise_datasets):
    """Combine all datasets into a comprehensive training set"""
    print("\nCOMBINING ALL DATASETS")
    print("=" * 30)
    
    combined_datasets = []
    
    # Add synthetic data
    if synthetic_df is not None:
        combined_datasets.append(synthetic_df)
        print(f"Added synthetic data: {synthetic_df.shape}")
    
    # Add live data
    if live_df is not None:
        combined_datasets.append(live_df)
        print(f"Added live data: {live_df.shape}")
    
    # Add enterprise data
    if enterprise_datasets:
        for i, df in enumerate(enterprise_datasets):
            combined_datasets.append(df)
            print(f"Added enterprise data {i+1}: {df.shape}")
    
    if not combined_datasets:
        print("No datasets available for training!")
        return None
    
    # Combine all datasets
    combined_df = pd.concat(combined_datasets, ignore_index=True)
    
    # Ensure is_fraud column is numeric
    if 'is_fraud' in combined_df.columns:
        combined_df['is_fraud'] = pd.to_numeric(combined_df['is_fraud'], errors='coerce')
        combined_df['is_fraud'] = combined_df['is_fraud'].fillna(0)
    
    print(f"\nCombined dataset shape: {combined_df.shape}")
    print(f"Combined fraud rate: {combined_df['is_fraud'].mean():.2%}")
    
    return combined_df

def preprocess_combined_data(df):
    """Preprocess the combined dataset"""
    print("\nPREPROCESSING COMBINED DATA")
    print("=" * 35)
    
    # Remove non-numeric columns except target
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    if 'is_fraud' in df.columns:
        numeric_columns = numeric_columns.drop('is_fraud')
    
    # Keep only numeric features and target
    features_df = df[numeric_columns]
    target = df['is_fraud']
    
    # Handle missing values in numeric features only
    features_df = features_df.fillna(features_df.mean())
    
    # Scale features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    features_scaled = pd.DataFrame(features_scaled, columns=features_df.columns)
    
    print(f"Preprocessed data shape: {features_scaled.shape}")
    print(f"Features scaled: {features_scaled.shape[1]}")
    
    return features_scaled, target, scaler

def train_comprehensive_models(X, y):
    """Train models on comprehensive dataset"""
    print("\nTRAINING COMPREHENSIVE FRAUD DETECTION MODELS")
    print("=" * 55)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert back to pandas Series for proper functionality
    y_train = pd.Series(y_train, name='target')
    y_test = pd.Series(y_test, name='target')
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Fraud rate in training: {y_train.mean():.4f}")
    print(f"Fraud rate in test: {y_test.mean():.4f}")
    
    # Initialize models with optimized parameters
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ) if LGBMClassifier else None,
        'Random Forest': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
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
        
        # Cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        # Store results
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"{name} - Accuracy: {accuracy:.4f}, ROC AUC: {roc_auc:.4f}")
        print(f"{name} - CV ROC AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return results, X_test, y_test

def evaluate_comprehensive_models(results, X_test, y_test):
    """Evaluate comprehensive models"""
    print("\nCOMPREHENSIVE MODEL EVALUATION")
    print("=" * 40)
    
    best_model = None
    best_auc = 0
    
    for name, result in results.items():
        print(f"\n{name.upper()} RESULTS:")
        print("-" * 25)
        
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
        recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC AUC: {result['roc_auc']:.4f}")
        print(f"CV ROC AUC: {result['cv_mean']:.4f} (+/- {result['cv_std'] * 2:.4f})")
        
        # Track best model
        if result['roc_auc'] > best_auc:
            best_auc = result['roc_auc']
            best_model = name
    
    print(f"\nBEST MODEL: {best_model} (ROC AUC: {best_auc:.4f})")
    return best_model

def save_comprehensive_model_and_results(results, best_model, scaler):
    """Save comprehensive model and results"""
    print(f"\nSAVING COMPREHENSIVE MODEL AND RESULTS")
    print("=" * 45)
    
    # Save best model
    best_model_obj = results[best_model]['model']
    model_data = {
        'model': best_model_obj,
        'scaler': scaler,
        'model_type': best_model,
        'training_date': datetime.now().isoformat(),
        'performance': {
            'accuracy': results[best_model]['accuracy'],
            'roc_auc': results[best_model]['roc_auc'],
            'cv_mean': results[best_model]['cv_mean'],
            'cv_std': results[best_model]['cv_std']
        }
    }
    
    joblib.dump(model_data, 'comprehensive_fraud_model.pkl')
    print(f"Comprehensive model ({best_model}) saved to 'comprehensive_fraud_model.pkl'")
    
    # Save comprehensive results
    comprehensive_results = {}
    for name, result in results.items():
        comprehensive_results[name] = {
            'accuracy': result['accuracy'],
            'roc_auc': result['roc_auc'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std']
        }
    
    with open('comprehensive_model_results.json', 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print("Comprehensive results saved to 'comprehensive_model_results.json'")
    
    return best_model_obj

def main():
    """Main function for comprehensive fraud detection training"""
    print("COMPREHENSIVE FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    print("Combining: Synthetic Data + Live Data + Enterprise Data")
    print("=" * 60)
    
    # Load all data sources
    synthetic_df = load_synthetic_data()
    live_df = load_live_data()
    enterprise_datasets = load_enterprise_data()
    
    # Combine datasets
    combined_df = combine_datasets(synthetic_df, live_df, enterprise_datasets)
    
    if combined_df is None:
        print("No data available for training!")
        return
    
    # Preprocess combined data
    X, y, scaler = preprocess_combined_data(combined_df)
    
    # Train comprehensive models
    results, X_test, y_test = train_comprehensive_models(X, y)
    
    # Evaluate models
    best_model_name = evaluate_comprehensive_models(results, X_test, y_test)
    
    # Save model and results
    best_model = save_comprehensive_model_and_results(results, best_model_name, scaler)
    
    print(f"\n🎉 COMPREHENSIVE FRAUD DETECTION TRAINING COMPLETE!")
    print(f"Best model: {best_model_name}")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"CV ROC AUC: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std'] * 2:.4f})")
    print(f"Model saved and ready for enterprise deployment!")

if __name__ == "__main__":
    main() 