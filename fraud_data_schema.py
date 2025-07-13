"""
Comprehensive Fraud Detection Data Schema
Covers all major fraud types: Credit Card, Payment Processing, Account Takeover, Merchant, Money Laundering
"""

# Core Transaction Data (Common across all fraud types)
CORE_TRANSACTION_FIELDS = {
    'transaction_id': 'str',           # Unique transaction identifier
    'timestamp': 'datetime',           # Transaction timestamp
    'amount': 'float',                 # Transaction amount
    'currency': 'str',                 # Transaction currency
    'status': 'str',                   # Transaction status (approved/declined/pending)
    'merchant_id': 'str',              # Merchant identifier
    'merchant_category': 'str',        # Merchant category code
    'merchant_name': 'str',            # Merchant business name
    'merchant_location': 'dict',       # Merchant location (lat, lng, country, city)
    'payment_method': 'str',           # Payment method type
    'user_id': 'str',                  # User/customer identifier
    'session_id': 'str',               # Session identifier
    'device_id': 'str',                # Device identifier
    'ip_address': 'str',               # IP address
    'user_agent': 'str',               # Browser/device information
}

# Credit Card Specific Data
CREDIT_CARD_FIELDS = {
    # Card Information
    'card_type': 'str',                # Visa, Mastercard, Amex, Discover
    'card_bin': 'str',                 # First 6 digits of card
    'card_last4': 'str',               # Last 4 digits of card
    'card_issuer': 'str',              # Card issuing bank
    'card_country': 'str',             # Card issuing country
    'card_age_days': 'int',            # Days since card was issued
    
    # Transaction Method
    'card_present': 'bool',            # Card physically present
    'used_chip': 'bool',               # Chip card used
    'used_contactless': 'bool',        # Contactless payment
    'used_pin': 'bool',                # PIN entered
    'used_signature': 'bool',          # Signature provided
    
    # Authorization Data
    'auth_code': 'str',                # Authorization code
    'cvv_result': 'str',               # CVV verification result
    'avs_result': 'str',               # Address verification result
    '3d_secure_used': 'bool',          # 3D Secure authentication
    '3d_secure_result': 'str',         # 3D Secure result
    
    # Geographic Data
    'transaction_location': 'dict',    # Transaction location (lat, lng, country, city)
    'cardholder_home_location': 'dict', # Cardholder's home location
    'distance_from_home_km': 'float',  # Distance from home
    'distance_from_last_txn_km': 'float', # Distance from last transaction
    'timezone_difference_hours': 'int', # Timezone difference
    
    # Behavioral Data
    'cardholder_avg_amount': 'float',  # Cardholder's average transaction amount
    'cardholder_transaction_count_24h': 'int', # Transactions in last 24h
    'cardholder_transaction_count_7d': 'int',  # Transactions in last 7 days
    'cardholder_preferred_categories': 'list', # Preferred merchant categories
    'cardholder_usual_times': 'dict',  # Usual transaction times
}

# Payment Processing Specific Data
PAYMENT_PROCESSING_FIELDS = {
    # Processing Details
    'processing_time_ms': 'int',       # Processing time in milliseconds
    'gateway_used': 'str',             # Payment gateway
    'processor_used': 'str',           # Payment processor
    'decline_reason': 'str',           # Reason for decline
    'retry_count': 'int',              # Number of retry attempts
    'previous_declines_24h': 'int',    # Previous declines in 24h
    
    # Bank Account Data (for ACH/wire)
    'bank_account_number': 'str',      # Bank account number (masked)
    'bank_routing_number': 'str',      # Bank routing number
    'bank_name': 'str',                # Bank name
    'account_type': 'str',             # Checking/savings
    'account_age_days': 'int',         # Account age in days
    
    # Risk Indicators
    'high_value_transaction': 'bool',  # Transaction above threshold
    'multiple_payment_methods': 'bool', # Multiple methods used
    'rapid_payment_attempts': 'bool',  # Rapid payment attempts
    'cross_border_transaction': 'bool', # Cross-border transaction
    'new_merchant_account': 'bool',    # New merchant account
}

# Account Takeover Specific Data
ACCOUNT_TAKEOVER_FIELDS = {
    # Authentication Data
    'login_attempts_24h': 'int',       # Login attempts in 24h
    'failed_password_attempts': 'int', # Failed password attempts
    'password_reset_requests_24h': 'int', # Password reset requests
    'mfa_used': 'bool',                # Multi-factor authentication used
    'mfa_method': 'str',               # MFA method (SMS, email, app)
    'mfa_bypassed': 'bool',            # MFA bypassed
    
    # Session Data
    'session_duration_minutes': 'int', # Session duration
    'session_location': 'dict',        # Session location
    'session_device': 'str',           # Session device
    'session_ip': 'str',               # Session IP address
    
    # Account Changes
    'recent_email_change': 'bool',     # Email changed recently
    'recent_phone_change': 'bool',     # Phone changed recently
    'recent_address_change': 'bool',   # Address changed recently
    'profile_update_frequency': 'int', # Profile updates per month
    
    # Behavioral Biometrics
    'typing_speed': 'float',           # Typing speed (characters per second)
    'mouse_movement_pattern': 'str',   # Mouse movement pattern hash
    'device_usage_pattern': 'str',     # Device usage pattern
    'navigation_pattern': 'str',       # Navigation pattern
    'copy_paste_behavior': 'bool',     # Copy-paste behavior detected
}

# Merchant Fraud Specific Data
MERCHANT_FRAUD_FIELDS = {
    # Business Information
    'business_registration_date': 'datetime', # Business registration date
    'business_type': 'str',            # Business type (LLC, Corp, etc.)
    'tax_id': 'str',                   # Tax identification number
    'business_address': 'dict',        # Business address
    'business_phone': 'str',           # Business phone
    'business_website': 'str',         # Business website
    'business_email': 'str',           # Business email
    
    # Financial Data
    'monthly_transaction_volume': 'float', # Monthly transaction volume
    'average_transaction_amount': 'float', # Average transaction amount
    'chargeback_rate': 'float',        # Chargeback rate percentage
    'refund_rate': 'float',            # Refund rate percentage
    'credit_score': 'int',             # Business credit score
    
    # Risk Indicators
    'high_chargeback_ratio': 'bool',   # High chargeback ratio
    'unusual_transaction_spikes': 'bool', # Unusual transaction spikes
    'geographic_anomalies': 'bool',    # Geographic anomalies
    'product_category_mismatch': 'bool', # Product category mismatch
    'customer_complaint_rate': 'float', # Customer complaint rate
    
    # Verification Data
    'website_legitimacy_score': 'float', # Website legitimacy score
    'social_media_presence': 'bool',   # Social media presence
    'customer_reviews_count': 'int',   # Number of customer reviews
    'customer_reviews_rating': 'float', # Average customer rating
    'business_license_verified': 'bool', # Business license verified
}

# Money Laundering Specific Data
MONEY_LAUNDERING_FIELDS = {
    # Transaction Chain Data
    'transaction_chain_id': 'str',     # Transaction chain identifier
    'chain_length': 'int',             # Length of transaction chain
    'chain_total_amount': 'float',     # Total amount in chain
    'chain_duration_hours': 'int',     # Chain duration in hours
    'structuring_detected': 'bool',    # Structuring pattern detected
    'smurfing_detected': 'bool',       # Smurfing pattern detected
    
    # Account Relationship Data
    'related_accounts': 'list',        # Related account IDs
    'account_ownership_pattern': 'str', # Account ownership pattern
    'shell_company_indicator': 'bool', # Shell company indicator
    'offshore_account_used': 'bool',   # Offshore account used
    'multiple_account_holders': 'bool', # Multiple account holders
    
    # Temporal Patterns
    'weekend_activity': 'bool',        # Weekend activity
    'holiday_activity': 'bool',        # Holiday activity
    'round_number_transaction': 'bool', # Round number transaction
    'regular_payment_pattern': 'bool', # Regular payment pattern
    'unusual_timing': 'bool',          # Unusual transaction timing
    
    # Geographic Patterns
    'high_risk_jurisdiction': 'bool',  # High-risk jurisdiction
    'cross_border_transfer': 'bool',   # Cross-border transfer
    'geographic_clustering': 'bool',   # Geographic clustering
    'travel_pattern_match': 'bool',    # Travel pattern match
    'location_risk_score': 'float',    # Location-based risk score
}

# Risk Scoring and Model Output
RISK_SCORING_FIELDS = {
    'overall_risk_score': 'float',     # Overall risk score (0-100)
    'credit_card_risk_score': 'float', # Credit card fraud risk
    'payment_processing_risk_score': 'float', # Payment processing risk
    'account_takeover_risk_score': 'float', # Account takeover risk
    'merchant_risk_score': 'float',    # Merchant fraud risk
    'money_laundering_risk_score': 'float', # Money laundering risk
    
    'risk_factors': 'list',            # List of risk factors
    'confidence_score': 'float',       # Model confidence (0-1)
    'recommended_action': 'str',       # Recommended action
    'fraud_probability': 'float',      # Probability of fraud (0-1)
}

# Complete Schema
COMPLETE_FRAUD_SCHEMA = {
    **CORE_TRANSACTION_FIELDS,
    **CREDIT_CARD_FIELDS,
    **PAYMENT_PROCESSING_FIELDS,
    **ACCOUNT_TAKEOVER_FIELDS,
    **MERCHANT_FRAUD_FIELDS,
    **MONEY_LAUNDERING_FIELDS,
    **RISK_SCORING_FIELDS
}

def get_required_fields_for_fraud_type(fraud_type):
    """
    Get required fields for specific fraud type
    """
    fraud_type_fields = {
        'credit_card': {**CORE_TRANSACTION_FIELDS, **CREDIT_CARD_FIELDS},
        'payment_processing': {**CORE_TRANSACTION_FIELDS, **PAYMENT_PROCESSING_FIELDS},
        'account_takeover': {**CORE_TRANSACTION_FIELDS, **ACCOUNT_TAKEOVER_FIELDS},
        'merchant': {**CORE_TRANSACTION_FIELDS, **MERCHANT_FRAUD_FIELDS},
        'money_laundering': {**CORE_TRANSACTION_FIELDS, **MONEY_LAUNDERING_FIELDS},
        'all': COMPLETE_FRAUD_SCHEMA
    }
    
    return fraud_type_fields.get(fraud_type, CORE_TRANSACTION_FIELDS)

def print_schema_summary():
    """
    Print summary of data requirements
    """
    print("FRAUD DETECTION DATA REQUIREMENTS SUMMARY")
    print("=" * 60)
    
    fraud_types = {
        'Credit Card Fraud': len(CREDIT_CARD_FIELDS),
        'Payment Processing Fraud': len(PAYMENT_PROCESSING_FIELDS),
        'Account Takeover Fraud': len(ACCOUNT_TAKEOVER_FIELDS),
        'Merchant Fraud': len(MERCHANT_FRAUD_FIELDS),
        'Money Laundering': len(MONEY_LAUNDERING_FIELDS)
    }
    
    print(f"Core Transaction Fields: {len(CORE_TRANSACTION_FIELDS)}")
    print(f"Risk Scoring Fields: {len(RISK_SCORING_FIELDS)}")
    print()
    
    for fraud_type, field_count in fraud_types.items():
        print(f"{fraud_type}: {field_count} specific fields")
    
    print(f"\nTotal Fields (Complete Schema): {len(COMPLETE_FRAUD_SCHEMA)}")
    print(f"Data Sources Required: 15+ different systems")
    print(f"Real-time Requirements: Sub-second processing")
    print(f"Data Retention: 7+ years for compliance")

if __name__ == "__main__":
    print_schema_summary() 