import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_fraud_detection_dataset(n_samples=10000, fraud_ratio=0.1):
    """
    Generate realistic fraud detection dataset
    """
    np.random.seed(42)
    
    # Generate transaction IDs
    transaction_ids = [f"TXN_{i:06d}" for i in range(n_samples)]
    
    # Generate timestamps (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    timestamps = [start_date + timedelta(
        seconds=random.randint(0, int((end_date - start_date).total_seconds()))
    ) for _ in range(n_samples)]
    
    # Generate user IDs
    user_ids = [f"USER_{random.randint(1000, 9999)}" for _ in range(n_samples)]
    
    # Generate merchant categories
    merchant_categories = [
        'electronics', 'clothing', 'food', 'travel', 'entertainment',
        'healthcare', 'automotive', 'home', 'sports', 'jewelry'
    ]
    
    # Generate transaction amounts (normal distribution with some outliers)
    amounts = np.random.lognormal(mean=4.5, sigma=0.8, size=n_samples)
    amounts = np.clip(amounts, 1, 10000)  # Limit between $1 and $10,000
    
    # Generate features
    data = {
        'transaction_id': transaction_ids,
        'timestamp': timestamps,
        'user_id': user_ids,
        'amount': amounts,
        'merchant_category': [random.choice(merchant_categories) for _ in range(n_samples)],
        'merchant_id': [f"MERCH_{random.randint(10000, 99999)}" for _ in range(n_samples)],
        'card_type': [random.choice(['visa', 'mastercard', 'amex', 'discover']) for _ in range(n_samples)],
        'card_present': [random.choice([True, False]) for _ in range(n_samples)],
        'distance_from_home': np.random.exponential(50, n_samples),  # km
        'distance_from_last_transaction': np.random.exponential(20, n_samples),  # km
        'ratio_to_median_purchase': np.random.lognormal(0, 0.5, n_samples),
        'repeat_retailer': [random.choice([True, False]) for _ in range(n_samples)],
        'used_chip': [random.choice([True, False]) for _ in range(n_samples)],
        'used_pin_number': [random.choice([True, False]) for _ in range(n_samples)],
        'online_order': [random.choice([True, False]) for _ in range(n_samples)],
        'hour_of_day': [ts.hour for ts in timestamps],
        'day_of_week': [ts.weekday() for ts in timestamps],
        'days_since_last_transaction': np.random.exponential(3, n_samples),
        'avg_amount_user': np.random.lognormal(4.0, 0.6, n_samples),
        'transaction_count_user': np.random.poisson(15, n_samples),
        'unique_merchants_user': np.random.poisson(8, n_samples),
        'foreign_transaction': [random.choice([True, False]) for _ in range(n_samples)],
        'high_risk_merchant': [random.choice([True, False]) for _ in range(n_samples)],
        'velocity_24h': np.random.poisson(2, n_samples),  # transactions in last 24h
        'velocity_7d': np.random.poisson(8, n_samples),   # transactions in last 7 days
    }
    
    df = pd.DataFrame(data)
    
    # Generate fraud labels based on patterns
    fraud_labels = generate_fraud_labels(df, fraud_ratio)
    df['is_fraud'] = fraud_labels
    
    return df

def generate_fraud_labels(df, fraud_ratio):
    """
    Generate realistic fraud labels based on patterns
    """
    n_samples = len(df)
    fraud_labels = np.zeros(n_samples, dtype=int)
    
    # Calculate fraud probability based on features
    fraud_prob = np.zeros(n_samples)
    
    # High amount transactions are more likely to be fraud
    fraud_prob += (df['amount'] > 1000) * 0.3
    
    # Foreign transactions are more likely to be fraud
    fraud_prob += df['foreign_transaction'] * 0.4
    
    # High velocity (many transactions quickly) indicates fraud
    fraud_prob += (df['velocity_24h'] > 5) * 0.5
    fraud_prob += (df['velocity_7d'] > 20) * 0.3
    
    # Large distance from home indicates fraud
    fraud_prob += (df['distance_from_home'] > 100) * 0.4
    
    # High risk merchants
    fraud_prob += df['high_risk_merchant'] * 0.3
    
    # Unusual amounts (very high ratio to median)
    fraud_prob += (df['ratio_to_median_purchase'] > 3) * 0.2
    
    # Online orders are slightly more risky
    fraud_prob += df['online_order'] * 0.1
    
    # Normalize probabilities
    fraud_prob = np.clip(fraud_prob, 0, 1)
    
    # Generate fraud labels
    fraud_labels = np.random.binomial(1, fraud_prob)
    
    # Ensure we have approximately the desired fraud ratio
    current_fraud_ratio = fraud_labels.mean()
    if current_fraud_ratio < fraud_ratio:
        # Add more fraud cases
        additional_fraud_needed = int((fraud_ratio - current_fraud_ratio) * n_samples)
        non_fraud_indices = np.where(fraud_labels == 0)[0]
        fraud_indices = np.random.choice(non_fraud_indices, 
                                       size=min(additional_fraud_needed, len(non_fraud_indices)), 
                                       replace=False)
        fraud_labels[fraud_indices] = 1
    
    return fraud_labels

def save_fraud_dataset():
    """
    Generate and save fraud detection dataset
    """
    print("Generating fraud detection dataset...")
    df = generate_fraud_detection_dataset(n_samples=50000, fraud_ratio=0.15)
    
    # Save to CSV
    df.to_csv('fraud_detection_data.csv', index=False)
    
    print(f"Dataset saved: fraud_detection_data.csv")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud rate: {df['is_fraud'].mean():.2%}")
    print(f"Features: {len(df.columns) - 1}")  # Excluding target
    
    # Show sample data
    print("\nSample data:")
    print(df.head())
    
    # Show fraud distribution
    print(f"\nFraud distribution:")
    print(df['is_fraud'].value_counts())
    
    return df

if __name__ == "__main__":
    save_fraud_dataset() 