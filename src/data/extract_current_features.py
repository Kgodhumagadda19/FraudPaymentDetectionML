#!/usr/bin/env python3
"""
Extract feature list from current model by analyzing training data structure
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import json

def generate_synthetic_pii_data(n_samples=1000):
    """Generate a small sample to determine feature structure"""
    np.random.seed(42)
    
    # Generate synthetic PII data
    data = {
        'transaction_id': [f"TXN_{i:06d}" for i in range(n_samples)],
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1H'),
        'amount': np.random.lognormal(4.0, 0.8, n_samples),
        'user_id': [f"USER_{np.random.randint(1000, 9999)}" for _ in range(n_samples)],
        'email': [f"user{i}@example.com" for i in range(n_samples)],
        'phone': [f"+1-555-{np.random.randint(100, 999)}-{np.random.randint(1000, 9999)}" for _ in range(n_samples)],
        'ssn': [f"{np.random.randint(100, 999)}-{np.random.randint(10, 99)}-{np.random.randint(1000, 9999)}" for _ in range(n_samples)],
        'ip_address': [f"{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}" for _ in range(n_samples)],
        'address': [f"{np.random.randint(100, 9999)} {np.random.choice(['Main', 'Oak', 'Pine', 'Elm'])} St, {np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'])}" for _ in range(n_samples)],
        'user_agent': [f"Mozilla/5.0 ({np.random.choice(['Windows', 'Mac', 'Linux'])}) {np.random.choice(['Chrome', 'Firefox', 'Safari'])}/90.0" for _ in range(n_samples)],
        'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], n_samples),
        'browser_type': np.random.choice(['chrome', 'firefox', 'safari', 'edge'], n_samples),
        'is_fraud': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    }
    
    return pd.DataFrame(data)

def anonymize_pii_features(df):
    """Anonymize PII features"""
    anonymized_df = df.copy()
    
    # Hash sensitive identifiers
    if 'email' in anonymized_df.columns:
        anonymized_df['email_hash'] = anonymized_df['email'].apply(
            lambda x: hash(x) % 10000
        )
    
    if 'phone' in anonymized_df.columns:
        anonymized_df['phone_hash'] = anonymized_df['phone'].apply(
            lambda x: hash(x) % 10000
        )
    
    if 'ssn' in anonymized_df.columns:
        anonymized_df['ssn_hash'] = anonymized_df['ssn'].apply(
            lambda x: hash(x) % 10000
        )
    
    # Extract features from IP addresses
    if 'ip_address' in anonymized_df.columns:
        anonymized_df['ip_country'] = anonymized_df['ip_address'].apply(
            lambda x: f"Country_{hash(x.split('.')[0]) % 50}"
        )
        anonymized_df['ip_network'] = anonymized_df['ip_address'].apply(
            lambda x: f"Network_{hash('.'.join(x.split('.')[:2])) % 100}"
        )
    
    # Extract features from addresses
    if 'address' in anonymized_df.columns:
        anonymized_df['address_city'] = anonymized_df['address'].apply(
            lambda x: x.split(',')[-1].strip() if ',' in x else 'Unknown'
        )
        anonymized_df['address_state'] = anonymized_df['address'].apply(
            lambda x: x.split(',')[-1].strip()[:2] if ',' in x else 'Unknown'
        )
    
    return anonymized_df

def extract_text_features(df):
    """Extract features from text-based PII using TF-IDF"""
    text_features = {}
    
    # Extract features from user agent
    if 'user_agent' in df.columns:
        try:
            tfidf = TfidfVectorizer(max_features=10, stop_words='english')
            user_agent_features = tfidf.fit_transform(df['user_agent'].fillna(''))
            user_agent_array = user_agent_features.toarray()
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
            address_array = address_features.toarray()
            feature_names = pd.Index([f'addr_feature_{i}' for i in range(address_array.shape[1])])
            address_df = pd.DataFrame(address_array, columns=feature_names)
            text_features['address'] = address_df
        except Exception as e:
            print(f"Warning: Could not extract address features: {e}")
    
    return text_features

def create_enhanced_features(df):
    """Create enhanced fraud detection features"""
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
    
    # Behavioral features
    enhanced_df['velocity_1h'] = np.random.poisson(1, len(df))
    enhanced_df['velocity_6h'] = np.random.poisson(3, len(df))
    enhanced_df['velocity_24h'] = np.random.poisson(8, len(df))
    enhanced_df['velocity_7d'] = np.random.poisson(20, len(df))
    enhanced_df['high_velocity'] = (enhanced_df['velocity_24h'] > 10).astype(int)
    
    # Risk indicators
    enhanced_df['new_device'] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
    enhanced_df['new_location'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
    enhanced_df['unusual_time'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
    enhanced_df['high_risk_merchant'] = np.random.choice([0, 1], len(df), p=[0.85, 0.15])
    enhanced_df['foreign_transaction'] = np.random.choice([0, 1], len(df), p=[0.9, 0.1])
    
    return enhanced_df

def encode_categorical_features(df):
    """Encode categorical features"""
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

def main():
    """Extract feature list from current model structure"""
    print("EXTRACTING CURRENT MODEL FEATURE LIST")
    print("=" * 50)
    
    # Generate synthetic PII data
    synthetic_pii_df = generate_synthetic_pii_data(n_samples=1000)
    
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
    
    print(f"Feature count: {X.shape[1]}")
    print(f"Features used in current model:")
    print("-" * 40)
    
    for i, col in enumerate(X.columns):
        print(f"{i+1:2d}. {col}")
    
    # Save feature list
    feature_info = {
        'feature_columns': list(X.columns),
        'feature_count': X.shape[1],
        'extraction_date': pd.Timestamp.now().isoformat()
    }
    
    with open('current_model_features.json', 'w') as f:
        json.dump(feature_info, f, indent=2)
    
    print(f"\n✅ Feature list saved to current_model_features.json")
    print(f"✅ Total features: {X.shape[1]}")
    
    return list(X.columns)

if __name__ == "__main__":
    main() 