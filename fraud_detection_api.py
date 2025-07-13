from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
import uvicorn
from fraud_detection_model import FraudDetectionModel
import joblib
import hashlib
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Enterprise Fraud Detection API",
    description="Real-time fraud detection for multiple fraud types",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize fraud detection model
fraud_model = None

# Pydantic models for API requests/responses
class TransactionData(BaseModel):
    # Core transaction data
    transaction_id: str
    timestamp: str
    amount: float
    user_id: str
    
    # PII data (will be anonymized)
    email: Optional[str] = None
    phone: Optional[str] = None
    ssn: Optional[str] = None
    ip_address: Optional[str] = None
    address: Optional[str] = None
    user_agent: Optional[str] = None
    
    # Device and browser info
    device_type: Optional[str] = "desktop"
    browser_type: Optional[str] = "chrome"
    
    # Additional transaction context
    merchant_id: Optional[str] = None
    merchant_category: Optional[str] = None
    distance_from_home_km: Optional[float] = None
    velocity_24h: Optional[int] = None
    foreign_transaction: Optional[bool] = False
    online_order: Optional[bool] = False
    high_risk_merchant: Optional[bool] = False
    transaction_count_user: Optional[int] = None
    card_present: Optional[bool] = True
    used_chip: Optional[bool] = True
    used_pin: Optional[bool] = False
    card_type: Optional[str] = None
    device_id: Optional[str] = None

class FraudPrediction(BaseModel):
    transaction_id: str
    fraud_probability: float
    risk_level: str
    recommended_action: str
    fraud_type: str
    confidence_score: float
    processing_time_ms: int
    timestamp: str

class BatchPredictionRequest(BaseModel):
    transactions: List[TransactionData]

class BatchPredictionResponse(BaseModel):
    predictions: List[FraudPrediction]
    total_transactions: int
    processing_time_ms: int
    fraud_count: int
    risk_distribution: Dict[str, int]

class ModelHealth(BaseModel):
    status: str
    model_type: str
    fraud_types: List[str]
    last_training: str
    total_models: int
    uptime_seconds: float

# Global variables
start_time = datetime.now()
model_loaded = False

def process_transaction_for_prediction(df):
    """Process transaction data to match model feature expectations"""
    processed_df = df.copy()
    
    # Anonymize PII features
    if 'email' in processed_df.columns:
        processed_df['email_hash'] = processed_df['email'].apply(
            lambda x: hash(str(x)) % 10000 if pd.notna(x) else 0
        )
    
    if 'phone' in processed_df.columns:
        processed_df['phone_hash'] = processed_df['phone'].apply(
            lambda x: hash(str(x)) % 10000 if pd.notna(x) else 0
        )
    
    if 'ssn' in processed_df.columns:
        processed_df['ssn_hash'] = processed_df['ssn'].apply(
            lambda x: hash(str(x)) % 10000 if pd.notna(x) else 0
        )
    
    # Extract features from IP addresses
    if 'ip_address' in processed_df.columns:
        processed_df['ip_country'] = processed_df['ip_address'].apply(
            lambda x: f"Country_{hash(str(x).split('.')[0]) % 50}" if pd.notna(x) and '.' in str(x) else "Country_0"
        )
        processed_df['ip_network'] = processed_df['ip_address'].apply(
            lambda x: f"Network_{hash('.'.join(str(x).split('.')[:2])) % 100}" if pd.notna(x) and '.' in str(x) else "Network_0"
        )
    
    # Extract features from addresses
    if 'address' in processed_df.columns:
        processed_df['address_city'] = processed_df['address'].apply(
            lambda x: x.split(',')[-1].strip() if pd.notna(x) and ',' in str(x) else 'Unknown'
        )
        processed_df['address_state'] = processed_df['address'].apply(
            lambda x: x.split(',')[-1].strip()[:2] if pd.notna(x) and ',' in str(x) else 'Unknown'
        )
    
    # Extract text features
    if 'user_agent' in processed_df.columns:
        try:
            tfidf = TfidfVectorizer(max_features=10, stop_words='english')
            user_agent_features = tfidf.fit_transform(processed_df['user_agent'].fillna(''))
            user_agent_array = user_agent_features.toarray()  # type: ignore
            for i in range(user_agent_array.shape[1]):
                processed_df[f'ua_feature_{i}'] = user_agent_array[:, i]
        except Exception as e:
            logger.warning(f"Could not extract user agent features: {e}")
            for i in range(10):
                processed_df[f'ua_feature_{i}'] = 0
    
    if 'address' in processed_df.columns:
        try:
            tfidf = TfidfVectorizer(max_features=5, stop_words='english')
            address_features = tfidf.fit_transform(processed_df['address'].fillna(''))
            address_array = address_features.toarray()  # type: ignore
            for i in range(address_array.shape[1]):
                processed_df[f'addr_feature_{i}'] = address_array[:, i]
        except Exception as e:
            logger.warning(f"Could not extract address features: {e}")
            for i in range(5):
                processed_df[f'addr_feature_{i}'] = 0
    
    # Create enhanced features
    if 'timestamp' in processed_df.columns:
        timestamps = pd.to_datetime(processed_df['timestamp'])
        processed_df['hour'] = timestamps.dt.hour
        processed_df['day_of_week'] = timestamps.dt.dayofweek
        processed_df['hour_sin'] = np.sin(2 * np.pi * processed_df['hour'] / 24)
        processed_df['hour_cos'] = np.cos(2 * np.pi * processed_df['hour'] / 24)
        processed_df['day_sin'] = np.sin(2 * np.pi * processed_df['day_of_week'] / 7)
        processed_df['day_cos'] = np.cos(2 * np.pi * processed_df['day_of_week'] / 7)
        processed_df['is_weekend'] = (processed_df['day_of_week'] >= 5).astype(int)
        processed_df['is_night'] = ((processed_df['hour'] >= 22) | (processed_df['hour'] <= 6)).astype(int)
    
    # Amount-based features
    if 'amount' in processed_df.columns:
        processed_df['amount_log'] = np.log1p(processed_df['amount'])
        processed_df['amount_squared'] = processed_df['amount'] ** 2
        processed_df['amount_percentile'] = processed_df['amount'].rank(pct=True)
        processed_df['amount_zscore'] = (processed_df['amount'] - processed_df['amount'].mean()) / processed_df['amount'].std()
        processed_df['is_high_value'] = (processed_df['amount'] > processed_df['amount'].quantile(0.95)).astype(int)
        processed_df['is_low_value'] = (processed_df['amount'] < processed_df['amount'].quantile(0.05)).astype(int)
        processed_df['is_very_high_value'] = (processed_df['amount'] > processed_df['amount'].quantile(0.99)).astype(int)
    
    # Behavioral features
    processed_df['velocity_1h'] = processed_df.get('velocity_24h', 0) // 24
    processed_df['velocity_6h'] = processed_df.get('velocity_24h', 0) // 4
    processed_df['velocity_7d'] = processed_df.get('velocity_24h', 0) * 7
    processed_df['high_velocity'] = (processed_df.get('velocity_24h', 0) > 10).astype(int)
    
    # Risk indicators
    processed_df['new_device'] = 0  # Default value
    processed_df['new_location'] = 0  # Default value
    processed_df['unusual_time'] = 0  # Default value
    processed_df['high_risk_merchant'] = processed_df.get('high_risk_merchant', False).astype(int)
    processed_df['foreign_transaction'] = processed_df.get('foreign_transaction', False).astype(int)
    
    # Encode categorical features
    categorical_columns = [
        'ip_country', 'ip_network', 'address_city', 'address_state',
        'device_type', 'browser_type', 'email_hash', 'phone_hash', 'ssn_hash'
    ]
    
    for col in categorical_columns:
        if col in processed_df.columns:
            le = LabelEncoder()
            processed_df[col] = le.fit_transform(processed_df[col].astype(str))
    
    # Select only numeric features that the model expects
    expected_features = [
        'amount', 'device_type', 'browser_type', 'email_hash', 'phone_hash', 'ssn_hash',
        'ip_country', 'ip_network', 'address_city', 'address_state', 'hour', 'day_of_week',
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'is_weekend', 'is_night',
        'amount_log', 'amount_squared', 'amount_percentile', 'amount_zscore',
        'is_high_value', 'is_low_value', 'is_very_high_value', 'velocity_1h',
        'velocity_6h', 'velocity_24h', 'velocity_7d', 'high_velocity', 'new_device',
        'new_location', 'unusual_time', 'high_risk_merchant', 'foreign_transaction'
    ] + [f'ua_feature_{i}' for i in range(10)] + [f'addr_feature_{i}' for i in range(5)]
    
    # Ensure all expected features exist
    for feature in expected_features:
        if feature not in processed_df.columns:
            processed_df[feature] = 0
    
    # Select only the expected features in the correct order
    processed_df = processed_df[expected_features]
    
    return processed_df

@app.on_event("startup")
async def startup_event():
    """Initialize the fraud detection model on startup"""
    global fraud_model, model_loaded
    try:
        fraud_model = FraudDetectionModel(model_type='lightgbm')
        # Try to load existing model
        try:
            fraud_model.load_model('api_compatible_model.pkl')
            logger.info("Loaded API-compatible fraud detection model")
        except FileNotFoundError:
            logger.warning("No compatible model found. Please train and convert a model first.")
        model_loaded = True
        logger.info("Fraud Detection API started successfully")
    except Exception as e:
        logger.error(f"Failed to initialize fraud detection model: {e}")
        model_loaded = False

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enterprise Fraud Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch_predict": "/batch-predict",
            "model_info": "/model-info",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=ModelHealth)
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - start_time).total_seconds()
    
    return ModelHealth(
        status="healthy" if model_loaded else "unhealthy",
        model_type=fraud_model.model_type if fraud_model else "none",
        fraud_types=fraud_model.fraud_types if fraud_model else [],
        last_training=fraud_model.model_metadata.get('credit_card', {}).get('training_date', 'never') if fraud_model else 'never',
        total_models=len(fraud_model.models) if fraud_model else 0,
        uptime_seconds=uptime
    )

@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(transaction: TransactionData):
    """Predict fraud for a single transaction"""
    if not model_loaded or not fraud_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Convert transaction to DataFrame
        transaction_dict = transaction.dict()
        df = pd.DataFrame([transaction_dict])
        
        # Process the data to match model expectations
        processed_df = process_transaction_for_prediction(df)
        
        # Make prediction
        prediction = fraud_model.predict_fraud(processed_df, 'credit_card')
        
        processing_time = (datetime.now() - start_time).microseconds // 1000
        
        return FraudPrediction(
            transaction_id=transaction.transaction_id,
            fraud_probability=prediction['fraud_probability'],
            risk_level=prediction['risk_level'],
            recommended_action=prediction['recommended_action'],
            fraud_type=prediction['fraud_type'],
            confidence_score=0.95,  # Placeholder
            processing_time_ms=processing_time,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/batch-predict", response_model=BatchPredictionResponse)
async def batch_predict_fraud(batch_request: BatchPredictionRequest):
    """Predict fraud for multiple transactions"""
    if not model_loaded or not fraud_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        # Convert transactions to DataFrame
        transactions_dict = [t.dict() for t in batch_request.transactions]
        df = pd.DataFrame(transactions_dict)
        
        # Process the data to match model expectations
        processed_df = process_transaction_for_prediction(df)
        
        # Make predictions
        predictions = []
        fraud_count = 0
        risk_distribution = {'LOW': 0, 'MEDIUM': 0, 'HIGH': 0, 'CRITICAL': 0}
        
        for i, transaction in processed_df.iterrows():
            transaction_df = pd.DataFrame([transaction])
            prediction = fraud_model.predict_fraud(transaction_df, 'credit_card')
            
            fraud_prediction = FraudPrediction(
                transaction_id=str(batch_request.transactions[i].transaction_id),
                fraud_probability=prediction['fraud_probability'],
                risk_level=prediction['risk_level'],
                recommended_action=prediction['recommended_action'],
                fraud_type=prediction['fraud_type'],
                confidence_score=0.95,
                processing_time_ms=0,
                timestamp=datetime.now().isoformat()
            )
            
            predictions.append(fraud_prediction)
            
            # Count fraud and risk levels
            if prediction['fraud_probability'] > 0.5:
                fraud_count += 1
            risk_distribution[prediction['risk_level']] += 1
        
        processing_time = (datetime.now() - start_time).microseconds // 1000
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_transactions=len(predictions),
            processing_time_ms=processing_time,
            fraud_count=fraud_count,
            risk_distribution=risk_distribution
        )
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the trained models"""
    if not model_loaded or not fraud_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": fraud_model.model_type,
        "fraud_types": fraud_model.fraud_types,
        "performance": fraud_model.get_model_performance(),
        "feature_importance": {
            fraud_type: (importance.to_dict('records')[:10] if importance is not None else [])
            for fraud_type in fraud_model.fraud_types
            for importance in [fraud_model.get_feature_importance(fraud_type)]
        }
    }

@app.post("/train")
async def train_model(background_tasks: BackgroundTasks):
    """Train the fraud detection model (background task)"""
    if not model_loaded or not fraud_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # This would typically be a background task
    # For now, we'll return a message
    return {
        "message": "Model training initiated",
        "status": "training",
        "estimated_time": "5-10 minutes"
    }

@app.get("/metrics")
async def get_metrics():
    """Get API metrics and performance statistics"""
    uptime = (datetime.now() - start_time).total_seconds()
    
    return {
        "uptime_seconds": uptime,
        "uptime_hours": uptime / 3600,
        "model_loaded": model_loaded,
        "total_models": len(fraud_model.models) if fraud_model else 0,
        "api_version": "1.0.0",
        "status": "healthy" if model_loaded else "unhealthy"
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {"error": "Endpoint not found", "detail": str(exc)}

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {"error": "Internal server error", "detail": str(exc)}

if __name__ == "__main__":
    uvicorn.run(
        "fraud_detection_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 