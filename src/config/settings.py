"""
Application settings and configuration
"""

import os
from typing import List, Optional

class Settings:
    """Application settings"""
    
    def __init__(self):
        # API Settings
        self.API_TITLE = "Enterprise Fraud Detection API"
        self.API_DESCRIPTION = "Real-time fraud detection for multiple fraud types"
        self.API_VERSION = "1.0.0"
        self.API_HOST = "0.0.0.0"
        self.API_PORT = 8000
        self.API_RELOAD = False
        
        # Model Settings
        self.MODEL_TYPE = "xgboost"
        self.FRAUD_TYPES = [
            'credit_card', 
            'payment_processing', 
            'account_takeover', 
            'merchant', 
            'money_laundering'
        ]
        self.MODEL_PATH = "models/"
        
        # Database Settings
        self.DATABASE_URL = os.getenv("DATABASE_URL")
        self.MONGODB_URL = os.getenv("MONGODB_URL")
        
        # Security Settings
        self.SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
        self.ALGORITHM = "HS256"
        self.ACCESS_TOKEN_EXPIRE_MINUTES = 30
        
        # Logging Settings
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        self.LOG_FILE = "logs/fraud_detection.log"
        
        # Feature Settings
        self.ENABLE_PII_ANONYMIZATION = True
        self.ENABLE_FEATURE_EXTRACTION = True
        self.MAX_FEATURES = 100
        
        # Performance Settings
        self.BATCH_SIZE = 1000
        self.MAX_WORKERS = 4
        
        # Monitoring Settings
        self.ENABLE_METRICS = True
        self.METRICS_PORT = 9090

# Global settings instance
settings = Settings()

# Environment-specific settings
def get_settings() -> Settings:
    """Get settings based on environment"""
    return settings 