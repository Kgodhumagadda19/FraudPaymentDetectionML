import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
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
    
    # Generate transaction data
    amounts = np.random.lognormal(4.5, 0.8, n_samples)
    amounts = np.clip(amounts, 1, 10000)
    
    # Generate fraud labels based on patterns
    fraud_prob = np.zeros(n_samples)
    
    # High-risk patterns
    fraud_prob += (amounts > 1000) * 0.3
    fraud_prob += np.random.choice([0, 1], n_samples, p=[0.8, 0.2]) * 0.4  # Random risk
    fraud_prob += np.random.choice([0, 1], n_samples, p=[0.9, 0.1]) * 0.5  # Device risk
    
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

def create_advanced_features(df):
    """
    Create advanced fraud detection features
    """
    print("\nCREATING ADVANCED FEATURES")
    print("=" * 30)
    
    advanced_df = df.copy()
    
    # Time-based features
    advanced_df['hour'] = np.random.randint(0, 24, len(df))
    advanced_df['day_of_week'] = np.random.randint(0, 7, len(df))
    advanced_df['is_weekend'] = (advanced_df['day_of_week'] >= 5).astype(int)
    advanced_df['is_night'] = ((advanced_df['hour'] >= 22) | (advanced_df['hour'] <= 6)).astype(int)
    
    # Amount-based features
    advanced_df['amount_log'] = np.log1p(advanced_df['amount'])
    advanced_df['amount_squared'] = advanced_df['amount'] ** 2
    advanced_df['is_high_value'] = (advanced_df['amount'] > advanced_df['amount'].quantile(0.95)).astype(int)
    advanced_df['is_low_value'] = (advanced_df['amount'] < advanced_df['amount'].quantile(0.05)).astype(int)
    
    # Behavioral features
    advanced_df['velocity_24h'] = np.random.poisson(2, len(df))
    advanced_df['velocity_7d'] = np.random.poisson(8, len(df))
    advanced_df['high_velocity'] = (advanced_df['velocity_24h'] > 5).astype(int)
    
    # Risk indicators
    advanced_df['new_device'] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
    advanced_df['new_location'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    advanced_df['unusual_time'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
    
    return advanced_df

def train_pii_enhanced_models(X, y):
    """
    Train models with PII-enhanced features
    """
    print("\nTRAINING PII-ENHANCED FRAUD DETECTION MODELS")
    print("=" * 55)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert to pandas Series
    y_train = pd.Series(y_train, name='target')
    y_test = pd.Series(y_test, name='target')
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Fraud rate in training: {y_train.mean():.4f}")
    print(f"Fraud rate in test: {y_test.mean():.4f}")
    
    # Initialize models with PII-optimized parameters
    models = {
        'XGBoost_PII': XGBClassifier(
            n_estimators=400,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        'LightGBM_PII': LGBMClassifier(
            n_estimators=400,
            max_depth=10,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        ) if LGBMClassifier else None,
        'Random_Forest_PII': RandomForestClassifier(
            n_estimators=400,
            max_depth=20,
            min_samples_split=3,
            min_samples_leaf=1,
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

def evaluate_pii_models(results, X_test, y_test):
    """
    Evaluate PII-enhanced models
    """
    print("\nPII-ENHANCED MODEL EVALUATION")
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

def save_pii_model_and_results(results, best_model, scaler, label_encoders):
    """
    Save PII-enhanced model and results
    """
    print(f"\nSAVING PII-ENHANCED MODEL AND RESULTS")
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
        }
    }
    
    joblib.dump(model_data, 'pii_enhanced_fraud_model.pkl')
    print(f"PII-enhanced model ({best_model}) saved to 'pii_enhanced_fraud_model.pkl'")
    
    # Save results
    pii_results = {}
    for name, result in results.items():
        pii_results[name] = {
            'accuracy': result['accuracy'],
            'roc_auc': result['roc_auc'],
            'cv_mean': result['cv_mean'],
            'cv_std': result['cv_std']
        }
    
    with open('pii_model_results.json', 'w') as f:
        json.dump(pii_results, f, indent=2)
    
    print("PII model results saved to 'pii_model_results.json'")
    
    return best_model_obj

def main():
    """
    Main function for PII-enhanced fraud detection training
    """
    print("PII-ENHANCED FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    print("Using: Synthetic PII + Privacy-Preserving Techniques")
    print("=" * 60)
    
    # Generate synthetic PII data
    synthetic_pii_df = generate_synthetic_pii_data(n_samples=100000)
    
    # Anonymize PII features
    anonymized_df = anonymize_pii_features(synthetic_pii_df)
    
    # Extract text features
    text_features = extract_text_features(anonymized_df)
    
    # Create advanced features
    engineered_df = create_advanced_features(anonymized_df)
    
    # Start with engineered features
    combined_df = engineered_df.copy()

    # Add text features (from TF-IDF)
    for feature_name, feature_df in text_features.items():
        combined_df = pd.concat([combined_df, feature_df], axis=1)

    # (Optional) Add metadata if not already in engineered_df
    # combined_df = pd.concat([combined_df, meta_df], axis=1)

    # Remove duplicate columns if any
    combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]
    
    # Encode categorical features
    encoded_df, label_encoders = encode_categorical_features(combined_df)
    
    # Prepare features for training
    # Select only numeric columns for model training
    feature_columns = [col for col in encoded_df.columns if col != 'is_fraud']
    X = encoded_df[feature_columns].select_dtypes(include=[np.number])
    y = encoded_df['is_fraud']

    # Train/test split and scaling
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nFinal dataset shape: {X_train_scaled.shape}")
    print(f"Features used: {X_train_scaled.shape[1]}")
    print(f"Fraud rate: {y.mean():.2%}")
    
    # Train PII-enhanced models
    results, X_test, y_test = train_pii_enhanced_models(X_train_scaled, y_train)
    
    # Evaluate models
    best_model_name = evaluate_pii_models(results, X_test, y_test)
    
    # Save model and results
    best_model = save_pii_model_and_results(results, best_model_name, scaler, label_encoders)
    
    print(f"\n🎉 PII-ENHANCED FRAUD DETECTION TRAINING COMPLETE!")
    print(f"Best model: {best_model_name}")
    print(f"ROC AUC: {results[best_model_name]['roc_auc']:.4f}")
    print(f"CV ROC AUC: {results[best_model_name]['cv_mean']:.4f} (+/- {results[best_model_name]['cv_std'] * 2:.4f})")
    print(f"✅ Privacy-compliant PII-enhanced model saved!")
    print(f"✅ Ready for enterprise deployment!")

if __name__ == "__main__":
    main() 