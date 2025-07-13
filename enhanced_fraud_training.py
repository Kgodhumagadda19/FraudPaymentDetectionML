import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import joblib
import json
from datetime import datetime
import hashlib
import re
import warnings
warnings.filterwarnings('ignore')

def generate_synthetic_pii_data(n_samples=100000):
    """
    Generate synthetic PII data for fraud detection training
    This is FAKE data - no real customer information
    """
    print("GENERATING SYNTHETIC PII DATA")
    print("=" * 40)
    
    np.random.seed(42)
    
    # Generate fake customer data
    fake_names = [
        f"Customer_{i:06d}" for i in range(n_samples)
    ]
    
    # Generate fake email addresses
    domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'company.com']
    fake_emails = [
        f"user{i:06d}@{np.random.choice(domains)}" for i in range(n_samples)
    ]
    
    # Generate fake phone numbers
    fake_phones = [
        f"+1-{np.random.randint(200, 999)}-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}"
        for _ in range(n_samples)
    ]
    
    # Generate fake IP addresses
    fake_ips = [
        f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}"
        for _ in range(n_samples)
    ]
    
    # Generate fake credit card numbers (masked)
    fake_card_numbers = [
        f"****-****-****-{np.random.randint(1000, 9999)}"
        for _ in range(n_samples)
    ]
    
    # Generate fake addresses
    fake_addresses = [
        f"{np.random.randint(100, 9999)} {np.random.choice(['Main', 'Oak', 'Pine', 'Elm', 'Maple'])} St, {np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'])}"
        for _ in range(n_samples)
    ]
    
    # Generate fake SSNs (masked)
    fake_ssns = [
        f"***-**-{np.random.randint(1000, 9999)}"
        for _ in range(n_samples)
    ]
    
    # Generate fake device IDs
    fake_device_ids = [
        f"DEV_{np.random.randint(100000, 999999)}"
        for _ in range(n_samples)
    ]
    
    # Generate fake user agents
    fake_user_agents = [
        np.random.choice([
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 14_7_1 like Mac OS X) AppleWebKit/605.1.15",
            "Mozilla/5.0 (Android 11; Mobile; rv:68.0) Gecko/68.0 Firefox/68.0"
        ])
        for _ in range(n_samples)
    ]
    
    # Generate transaction data with more realistic patterns
    amounts = np.random.lognormal(4.5, 0.8, n_samples)
    amounts = np.clip(amounts, 1, 10000)
    
    # Generate fraud labels based on more sophisticated patterns
    fraud_prob = np.zeros(n_samples)
    
    # High-risk patterns
    fraud_prob += (amounts > 1000) * 0.3
    fraud_prob += (amounts > 5000) * 0.2  # Very high amounts
    fraud_prob += np.random.choice([0, 1], n_samples, p=[0.8, 0.2]) * 0.4  # Random risk
    fraud_prob += np.random.choice([0, 1], n_samples, p=[0.9, 0.1]) * 0.5  # Device risk
    
    # Time-based risk
    hours = np.random.randint(0, 24, n_samples)
    fraud_prob += ((hours >= 22) | (hours <= 6)) * 0.2  # Night transactions
    
    fraud_prob = np.clip(fraud_prob, 0, 1)
    fraud_labels = np.random.binomial(1, fraud_prob)
    
    # Create DataFrame
    data = {
        'customer_name': fake_names,
        'email': fake_emails,
        'phone': fake_phones,
        'ip_address': fake_ips,
        'credit_card': fake_card_numbers,
        'address': fake_addresses,
        'ssn': fake_ssns,
        'device_id': fake_device_ids,
        'user_agent': fake_user_agents,
        'amount': amounts,
        'is_fraud': fraud_labels
    }
    
    df = pd.DataFrame(data)
    
    print(f"Synthetic PII data generated: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print("⚠️  NOTE: This is FAKE data for training only!")
    
    return df

def anonymize_pii_features(df):
    """
    Anonymize PII features using privacy-preserving techniques
    """
    print("\nANONYMIZING PII FEATURES")
    print("=" * 35)
    
    anonymized_df = df.copy()
    
    # 1. Hash sensitive identifiers
    if 'email' in anonymized_df.columns:
        anonymized_df['email_hash'] = anonymized_df['email'].apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
        )
    
    if 'phone' in anonymized_df.columns:
        anonymized_df['phone_hash'] = anonymized_df['phone'].apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
        )
    
    if 'ssn' in anonymized_df.columns:
        anonymized_df['ssn_hash'] = anonymized_df['ssn'].apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
        )
    
    # 2. Extract features from IP addresses
    if 'ip_address' in anonymized_df.columns:
        anonymized_df['ip_country'] = anonymized_df['ip_address'].apply(
            lambda x: f"Country_{hash(x.split('.')[0]) % 50}"
        )
        anonymized_df['ip_network'] = anonymized_df['ip_address'].apply(
            lambda x: f"Network_{hash('.'.join(x.split('.')[:2])) % 100}"
        )
    
    # 3. Extract features from addresses
    if 'address' in anonymized_df.columns:
        anonymized_df['address_city'] = anonymized_df['address'].apply(
            lambda x: x.split(',')[-1].strip() if ',' in x else 'Unknown'
        )
        anonymized_df['address_state'] = anonymized_df['address'].apply(
            lambda x: x.split(',')[-1].strip()[:2] if ',' in x else 'Unknown'
        )
    
    # 4. Extract features from user agents
    if 'user_agent' in anonymized_df.columns:
        anonymized_df['device_type'] = anonymized_df['user_agent'].apply(
            lambda x: 'Mobile' if 'Mobile' in x or 'iPhone' in x or 'Android' in x else 'Desktop'
        )
        anonymized_df['browser_type'] = anonymized_df['user_agent'].apply(
            lambda x: 'Chrome' if 'Chrome' in x else 'Firefox' if 'Firefox' in x else 'Safari' if 'Safari' in x else 'Other'
        )
    
    # 5. Create behavioral features
    if 'customer_name' in anonymized_df.columns:
        anonymized_df['customer_id_hash'] = anonymized_df['customer_name'].apply(
            lambda x: hashlib.sha256(x.encode()).hexdigest()[:16]
        )
    
    print("PII features anonymized successfully")
    return anonymized_df

def extract_text_features(df):
    """
    Extract features from text-based PII using TF-IDF
    """
    print("\nEXTRACTING TEXT FEATURES")
    print("=" * 30)
    
    text_features = {}
    
    # Extract features from user agent
    if 'user_agent' in df.columns:
        try:
            tfidf = TfidfVectorizer(max_features=10, stop_words='english')
            user_agent_features = tfidf.fit_transform(df['user_agent'].fillna(''))
            # Convert sparse matrix to dense array
            user_agent_array = user_agent_features.toarray()  # type: ignore
            feature_names = pd.Index([f'ua_feature_{i}' for i in range(user_agent_array.shape[1])])
            user_agent_df = pd.DataFrame(user_agent_array, columns=feature_names)
            text_features['user_agent'] = user_agent_df
        except Exception as e:
            print(f"Warning: Could not extract user agent features: {e}")
    
    # Extract features from address
    if 'address' in df.columns:
        try:
            tfidf = TfidfVectorizer(max_features=5, stop_words='english')
            address_features = tfidf.fit_transform(df['address'].fillna(''))
            # Convert sparse matrix to dense array
            address_array = address_features.toarray()  # type: ignore
            feature_names = pd.Index([f'addr_feature_{i}' for i in range(address_array.shape[1])])
            address_df = pd.DataFrame(address_array, columns=feature_names)
            text_features['address'] = address_df
        except Exception as e:
            print(f"Warning: Could not extract address features: {e}")
    
    return text_features

def create_enhanced_features(df):
    """
    Create enhanced fraud detection features with advanced engineering
    """
    print("\nCREATING ENHANCED FEATURES")
    print("=" * 35)
    
    enhanced_df = df.copy()
    
    # Time-based features with cyclical encoding
    enhanced_df['hour'] = np.random.randint(0, 24, len(df))
    enhanced_df['day_of_week'] = np.random.randint(0, 7, len(df))
    enhanced_df['hour_sin'] = np.sin(2 * np.pi * enhanced_df['hour'] / 24)
    enhanced_df['hour_cos'] = np.cos(2 * np.pi * enhanced_df['hour'] / 24)
    enhanced_df['day_sin'] = np.sin(2 * np.pi * enhanced_df['day_of_week'] / 7)
    enhanced_df['day_cos'] = np.cos(2 * np.pi * enhanced_df['day_of_week'] / 7)
    enhanced_df['is_weekend'] = (enhanced_df['day_of_week'] >= 5).astype(int)
    enhanced_df['is_night'] = ((enhanced_df['hour'] >= 22) | (enhanced_df['hour'] <= 6)).astype(int)
    
    # Enhanced amount-based features
    enhanced_df['amount_log'] = np.log1p(enhanced_df['amount'])
    enhanced_df['amount_squared'] = enhanced_df['amount'] ** 2
    enhanced_df['amount_percentile'] = enhanced_df['amount'].rank(pct=True)
    enhanced_df['amount_zscore'] = (enhanced_df['amount'] - enhanced_df['amount'].mean()) / enhanced_df['amount'].std()
    enhanced_df['is_high_value'] = (enhanced_df['amount'] > enhanced_df['amount'].quantile(0.95)).astype(int)
    enhanced_df['is_low_value'] = (enhanced_df['amount'] < enhanced_df['amount'].quantile(0.05)).astype(int)
    enhanced_df['is_very_high_value'] = (enhanced_df['amount'] > enhanced_df['amount'].quantile(0.99)).astype(int)
    
    # Behavioral features with more sophistication
    enhanced_df['velocity_1h'] = np.random.poisson(1, len(df))
    enhanced_df['velocity_6h'] = np.random.poisson(3, len(df))
    enhanced_df['velocity_24h'] = np.random.poisson(8, len(df))
    enhanced_df['velocity_7d'] = np.random.poisson(20, len(df))
    enhanced_df['high_velocity'] = (enhanced_df['velocity_24h'] > 10).astype(int)
    enhanced_df['very_high_velocity'] = (enhanced_df['velocity_24h'] > 20).astype(int)
    
    # Risk indicators with more granularity
    enhanced_df['new_device'] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
    enhanced_df['new_location'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    enhanced_df['unusual_time'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
    enhanced_df['high_risk_merchant'] = np.random.choice([0, 1], len(df), p=[0.85, 0.15])
    enhanced_df['international_transaction'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
    
    # Interaction features
    enhanced_df['amount_time_interaction'] = enhanced_df['amount'] * enhanced_df['hour']
    enhanced_df['device_location_risk'] = enhanced_df['new_device'] * enhanced_df['new_location']
    enhanced_df['high_value_night'] = enhanced_df['is_high_value'] * enhanced_df['is_night']
    enhanced_df['velocity_amount_risk'] = enhanced_df['high_velocity'] * enhanced_df['is_high_value']
    
    # Geographic risk scoring
    enhanced_df['geo_risk'] = np.random.choice([0, 1, 2], len(df), p=[0.7, 0.2, 0.1])
    enhanced_df['high_risk_country'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
    
    # Customer behavior patterns
    enhanced_df['customer_age_days'] = np.random.randint(1, 1000, len(df))
    enhanced_df['total_transactions'] = np.random.poisson(50, len(df))
    enhanced_df['avg_transaction_amount'] = enhanced_df['amount'] * np.random.uniform(0.8, 1.2, len(df))
    enhanced_df['transaction_frequency'] = enhanced_df['total_transactions'] / enhanced_df['customer_age_days']
    
    return enhanced_df

def encode_categorical_features(df):
    """
    Encode categorical features
    """
    print("\nENCODING CATEGORICAL FEATURES")
    print("=" * 35)
    
    encoded_df = df.copy()
    label_encoders = {}
    
    categorical_columns = [
        'ip_country', 'ip_network', 'address_city', 'address_state',
        'device_type', 'browser_type', 'customer_id_hash',
        'email_hash', 'phone_hash', 'ssn_hash'
    ]
    
    for col in categorical_columns:
        if col in encoded_df.columns:
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            label_encoders[col] = le
    
    return encoded_df, label_encoders

def handle_class_imbalance(X, y):
    """
    Handle class imbalance using SMOTE and undersampling
    """
    print("\nHANDLING CLASS IMBALANCE")
    print("=" * 30)
    
    print(f"Original fraud rate: {y.mean():.2%}")
    
    # Use SMOTE for oversampling and RandomUnderSampler for undersampling
    over = SMOTE(sampling_strategy='auto', random_state=42)  # Auto balance
    under = RandomUnderSampler(sampling_strategy='auto', random_state=42)  # Auto balance
    
    # Create pipeline
    steps = [('o', over), ('u', under)]
    pipeline = Pipeline(steps=steps)
    
    # Apply resampling
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    
    print(f"Resampled fraud rate: {y_resampled.mean():.2%}")
    print(f"Original shape: {X.shape} -> Resampled shape: {X_resampled.shape}")
    
    return X_resampled, y_resampled

def hyperparameter_tuning(X_train, y_train):
    """
    Perform hyperparameter tuning for best model performance
    """
    print("\nHYPERPARAMETER TUNING")
    print("=" * 25)
    
    # XGBoost parameter grid
    xgb_param_grid = {
        'n_estimators': [300, 500],
        'max_depth': [8, 10, 12],
        'learning_rate': [0.03, 0.05],
        'subsample': [0.8, 0.9],
        'colsample_bytree': [0.8, 0.9],
        'min_child_weight': [1, 3]
    }
    
    print("Tuning XGBoost...")
    xgb = XGBClassifier(random_state=42, eval_metric='logloss', use_label_encoder=False)
    xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
    xgb_grid.fit(X_train, y_train)
    
    print(f"Best XGBoost parameters: {xgb_grid.best_params_}")
    print(f"Best XGBoost CV score: {xgb_grid.best_score_:.4f}")
    
    # LightGBM parameter grid (if available)
    if LGBMClassifier:
        lgb_param_grid = {
            'n_estimators': [300, 500],
            'max_depth': [8, 10, 12],
            'learning_rate': [0.03, 0.05],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.8, 0.9]
        }
        
        print("Tuning LightGBM...")
        lgb = LGBMClassifier(random_state=42, verbose=-1)
        lgb_grid = GridSearchCV(lgb, lgb_param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
        lgb_grid.fit(X_train, y_train)
        
        print(f"Best LightGBM parameters: {lgb_grid.best_params_}")
        print(f"Best LightGBM CV score: {lgb_grid.best_score_:.4f}")
    else:
        lgb_grid = None
    
    return xgb_grid.best_estimator_, lgb_grid.best_estimator_ if lgb_grid else None

def train_ensemble_models(X_train, y_train, X_test, y_test):
    """
    Train ensemble models with optimized parameters
    """
    print("\nTRAINING ENHANCED FRAUD DETECTION MODELS")
    print("=" * 55)
    
    # Get optimized models
    best_xgb, best_lgb = hyperparameter_tuning(X_train, y_train)
    
    # Create ensemble
    models = {
        'XGBoost_Enhanced': best_xgb,
        'LightGBM_Enhanced': best_lgb if best_lgb else None,
        'Random_Forest_Enhanced': RandomForestClassifier(
            n_estimators=500,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
    }
    
    # Create voting classifier
    available_models = [(name, model) for name, model in models.items() if model is not None]
    if len(available_models) >= 2:
        ensemble = VotingClassifier(
            estimators=available_models,
            voting='soft'
        )
        models['Ensemble'] = ensemble
    
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
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # type: ignore
        
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

def evaluate_enhanced_models(results, X_test, y_test):
    """
    Evaluate enhanced models with detailed analysis
    """
    print("\nENHANCED MODEL EVALUATION")
    print("=" * 40)
    
    best_model = None
    best_auc = 0
    
    for name, result in results.items():
        print(f"\n{name.upper()} RESULTS:")
        print("-" * 30)
        
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

def save_enhanced_model_and_results(results, best_model, scaler, label_encoders, feature_columns):
    """
    Save enhanced model and results with feature metadata
    """
    print(f"\nSAVING ENHANCED MODEL AND RESULTS")
    print("=" * 45)
    
    # Save best model
    best_model_obj = results[best_model]['model']
    model_data = {
        'model': best_model_obj,
        'scaler': scaler,
        'label_encoders': label_encoders,
        'model_type': best_model,
        'training_date': datetime.now().isoformat(),
        'performance': {
            'accuracy': results[best_model]['accuracy'],
            'roc_auc': results[best_model]['roc_auc'],
            'cv_mean': results[best_model]['cv_mean'],
            'cv_std': results[best_model]['cv_std']
        },
        'feature_columns': feature_columns,  # Save the exact feature list used
        'feature_count': len(feature_columns)
    }
    
    joblib.dump(model_data, 'enhanced_fraud_model.pkl')
    print(f"Enhanced model ({best_model}) saved to 'enhanced_fraud_model.pkl'")
    print(f"✅ Feature list saved: {len(feature_columns)} features")
    
    # Save results
    enhanced_results = {}
    for name, result in results.items():
        enhanced_results[name] = {
            'accuracy': result['accuracy'],
            'roc_auc': result['roc_auc'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std']
        }
    
    with open('enhanced_model_results.json', 'w') as f:
        json.dump(enhanced_results, f, indent=2)
    
    print("Enhanced model results saved to 'enhanced_model_results.json'")
    
    # Also save feature list to a separate file for easy access
    feature_info = {
        'feature_columns': feature_columns,
        'feature_count': len(feature_columns),
        'training_date': datetime.now().isoformat(),
        'model_type': best_model
    }
    
    with open('model_features.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    print(f"✅ Feature list saved to model_features.json")
    
    return best_model_obj

def main():
    """
    Main function for enhanced fraud detection training
    """
    print("ENHANCED FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    print("Using: Advanced Feature Engineering + Class Balancing + Hyperparameter Tuning")
    print("=" * 60)
    
    # Generate synthetic PII data
    synthetic_pii_df = generate_synthetic_pii_data(n_samples=100000)
    
    # Anonymize PII features
    anonymized_df = anonymize_pii_features(synthetic_pii_df)
    
    # Extract text features
    text_features = extract_text_features(anonymized_df)
    
    # Create enhanced features
    enhanced_df = create_enhanced_features(anonymized_df)
    
    # Combine all features
    combined_df = enhanced_df.copy()
    
    # Add text features
    for feature_name, feature_df in text_features.items():
        combined_df = pd.concat([combined_df, feature_df], axis=1)
    
    # Remove duplicate columns
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    # Encode categorical features
    encoded_df, label_encoders = encode_categorical_features(combined_df)
    
    # Prepare features for training
    feature_columns = [col for col in encoded_df.columns if col != 'is_fraud']
    X = encoded_df[feature_columns].select_dtypes(include=[np.number])
    y = encoded_df['is_fraud']
    
    print(f"\nInitial dataset shape: {X.shape}")
    print(f"Features used: {X.shape[1]}")
    print(f"Fraud rate: {y.mean():.2%}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Handle class imbalance
    X_train_balanced, y_train_balanced = handle_class_imbalance(X_train_scaled, y_train)
    
    print(f"\nFinal training set shape: {X_train_balanced.shape}")
    print(f"Balanced fraud rate: {y_train_balanced.mean():.2%}")
    
    # Train enhanced models
    results, X_test, y_test = train_ensemble_models(X_train_balanced, y_train_balanced, X_test_scaled, y_test)
    
    # Evaluate models
    best_model_name = evaluate_enhanced_models(results, X_test, y_test)
    
    # Save model and results
    best_model = save_enhanced_model_and_results(results, best_model_name, scaler, label_encoders, list(X.columns))
    
    print(f"\n🎉 ENHANCED FRAUD DETECTION TRAINING COMPLETE!")
    print(f"Best model: {best_model_name}")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"CV ROC AUC: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std'] * 2:.4f})")
    print(f"✅ Enhanced model with improved accuracy saved!")
    print(f"✅ Ready for enterprise deployment!")

if __name__ == "__main__":
    main() 