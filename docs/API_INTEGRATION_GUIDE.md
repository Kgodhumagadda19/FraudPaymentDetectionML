# Fraud Detection API Integration Guide

## Overview
The Enterprise Fraud Detection API provides real-time fraud detection capabilities for multiple fraud types including credit card fraud, payment processing fraud, account takeover, merchant fraud, and money laundering.

## Base URL
```
http://localhost:8000
```

## API Endpoints

### 1. Health Check
**GET** `/health`

Check if the API is running and healthy.

**Response:**
```json
{
  "status": "healthy",
  "model_type": "lightgbm",
  "fraud_types": ["credit_card", "payment_processing", "account_takeover", "merchant", "money_laundering"],
  "last_training": "2024-01-15T00:00:00Z",
  "total_models": 5,
  "uptime_seconds": 1234.5
}
```

### 2. Single Transaction Prediction
**POST** `/predict`

Predict fraud probability for a single transaction.

**Request Body:**
```json
{
  "transaction_id": "TXN_123456",
  "timestamp": "2024-01-15T14:30:00Z",
  "amount": 150.75,
  "user_id": "USER_1234",
  "merchant_id": "MERCH_001",
  "merchant_category": "electronics",
  "distance_from_home_km": 25.5,
  "velocity_24h": 3,
  "foreign_transaction": false,
  "online_order": true,
  "high_risk_merchant": false,
  "transaction_count_user": 15,
  "card_present": false,
  "used_chip": false,
  "used_pin": false,
  "card_type": "visa",
  "device_id": "DEV_001"
}
```

**Response:**
```json
{
  "transaction_id": "TXN_123456",
  "fraud_probability": 0.1000,
  "risk_level": "LOW",
  "recommended_action": "APPROVE",
  "fraud_type": "credit_card",
  "confidence_score": 0.95,
  "processing_time_ms": 92,
  "timestamp": "2024-01-15T14:30:01Z"
}
```

### 3. Batch Transaction Prediction
**POST** `/batch-predict`

Predict fraud probability for multiple transactions.

**Request Body:**
```json
{
  "transactions": [
    {
      "transaction_id": "TXN_001",
      "timestamp": "2024-01-15T14:30:00Z",
      "amount": 500.00,
      "user_id": "USER_001",
      "merchant_id": "MERCH_001",
      "merchant_category": "food",
      "distance_from_home_km": 5.0,
      "velocity_24h": 1,
      "foreign_transaction": false,
      "online_order": false,
      "high_risk_merchant": false,
      "transaction_count_user": 5,
      "card_present": true,
      "used_chip": true,
      "used_pin": false,
      "card_type": "debit",
      "device_id": "DEV_002"
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "transaction_id": "TXN_001",
      "fraud_probability": 0.1000,
      "risk_level": "LOW",
      "recommended_action": "APPROVE",
      "fraud_type": "credit_card",
      "confidence_score": 0.95,
      "processing_time_ms": 0,
      "timestamp": "2024-01-15T14:30:01Z"
    }
  ],
  "total_transactions": 1,
  "processing_time_ms": 73,
  "fraud_count": 0,
  "risk_distribution": {
    "LOW": 1,
    "MEDIUM": 0,
    "HIGH": 0,
    "CRITICAL": 0
  }
}
```

### 4. Model Information
**GET** `/model-info`

Get information about the trained models.

**Response:**
```json
{
  "model_type": "lightgbm",
  "fraud_types": ["credit_card", "payment_processing", "account_takeover", "merchant", "money_laundering"],
  "performance": {
    "credit_card": {
      "accuracy": 0.7945,
      "roc_auc": 0.4990
    }
  }
}
```

### 5. API Metrics
**GET** `/metrics`

Get API performance metrics.

**Response:**
```json
{
  "uptime_seconds": 1234.5,
  "uptime_hours": 0.34,
  "model_loaded": true,
  "total_models": 5,
  "api_version": "1.0.0",
  "status": "healthy"
}
```

## Integration Examples

### Python Client Example

```python
import requests
import json
from datetime import datetime

class FraudDetectionClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        
    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()
    
    def predict_single(self, transaction_data):
        """Predict fraud for a single transaction"""
        response = requests.post(
            f"{self.base_url}/predict",
            json=transaction_data,
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    
    def predict_batch(self, transactions):
        """Predict fraud for multiple transactions"""
        response = requests.post(
            f"{self.base_url}/batch-predict",
            json={"transactions": transactions},
            headers={"Content-Type": "application/json"}
        )
        return response.json()
    
    def get_model_info(self):
        """Get model information"""
        response = requests.get(f"{self.base_url}/model-info")
        return response.json()

# Usage example
client = FraudDetectionClient()

# Check health
health = client.health_check()
print(f"API Status: {health['status']}")

# Single prediction
transaction = {
    "transaction_id": "TXN_123456",
    "timestamp": datetime.now().isoformat(),
    "amount": 150.75,
    "user_id": "USER_1234",
    "merchant_id": "MERCH_001",
    "merchant_category": "electronics",
    "distance_from_home_km": 25.5,
    "velocity_24h": 3,
    "foreign_transaction": False,
    "online_order": True,
    "high_risk_merchant": False,
    "transaction_count_user": 15,
    "card_present": False,
    "used_chip": False,
    "used_pin": False,
    "card_type": "visa",
    "device_id": "DEV_001"
}

prediction = client.predict_single(transaction)
print(f"Fraud Probability: {prediction['fraud_probability']}")
print(f"Risk Level: {prediction['risk_level']}")
print(f"Recommended Action: {prediction['recommended_action']}")
```

### JavaScript/Node.js Client Example

```javascript
class FraudDetectionClient {
    constructor(baseUrl = 'http://localhost:8000') {
        this.baseUrl = baseUrl;
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        return await response.json();
    }
    
    async predictSingle(transactionData) {
        const response = await fetch(`${this.baseUrl}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(transactionData)
        });
        return await response.json();
    }
    
    async predictBatch(transactions) {
        const response = await fetch(`${this.baseUrl}/batch-predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ transactions })
        });
        return await response.json();
    }
    
    async getModelInfo() {
        const response = await fetch(`${this.baseUrl}/model-info`);
        return await response.json();
    }
}

// Usage example
const client = new FraudDetectionClient();

// Check health
client.healthCheck().then(health => {
    console.log(`API Status: ${health.status}`);
});

// Single prediction
const transaction = {
    transaction_id: "TXN_123456",
    timestamp: new Date().toISOString(),
    amount: 150.75,
    user_id: "USER_1234",
    merchant_id: "MERCH_001",
    merchant_category: "electronics",
    distance_from_home_km: 25.5,
    velocity_24h: 3,
    foreign_transaction: false,
    online_order: true,
    high_risk_merchant: false,
    transaction_count_user: 15,
    card_present: false,
    used_chip: false,
    used_pin: false,
    card_type: "visa",
    device_id: "DEV_001"
};

client.predictSingle(transaction).then(prediction => {
    console.log(`Fraud Probability: ${prediction.fraud_probability}`);
    console.log(`Risk Level: ${prediction.risk_level}`);
    console.log(`Recommended Action: ${prediction.recommended_action}`);
});
```

### cURL Examples

```bash
# Health check
curl -X GET "http://localhost:8000/health"

# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_123456",
    "timestamp": "2024-01-15T14:30:00Z",
    "amount": 150.75,
    "user_id": "USER_1234",
    "merchant_id": "MERCH_001",
    "merchant_category": "electronics",
    "distance_from_home_km": 25.5,
    "velocity_24h": 3,
    "foreign_transaction": false,
    "online_order": true,
    "high_risk_merchant": false,
    "transaction_count_user": 15,
    "card_present": false,
    "used_chip": false,
    "used_pin": false,
    "card_type": "visa",
    "device_id": "DEV_001"
  }'

# Batch prediction
curl -X POST "http://localhost:8000/batch-predict" \
  -H "Content-Type: application/json" \
  -d '{
    "transactions": [
      {
        "transaction_id": "TXN_001",
        "timestamp": "2024-01-15T14:30:00Z",
        "amount": 500.00,
        "user_id": "USER_001",
        "merchant_id": "MERCH_001",
        "merchant_category": "food",
        "distance_from_home_km": 5.0,
        "velocity_24h": 1,
        "foreign_transaction": false,
        "online_order": false,
        "high_risk_merchant": false,
        "transaction_count_user": 5,
        "card_present": true,
        "used_chip": true,
        "used_pin": false,
        "card_type": "debit",
        "device_id": "DEV_002"
      }
    ]
  }'
```

## Risk Levels and Actions

| Risk Level | Probability Range | Recommended Action | Description |
|------------|------------------|-------------------|-------------|
| LOW | 0.0 - 0.1 | APPROVE | Low risk transaction |
| MEDIUM | 0.1 - 0.3 | REVIEW | Moderate risk, requires review |
| HIGH | 0.3 - 0.7 | DECLINE | High risk, should be declined |
| CRITICAL | 0.7 - 1.0 | BLOCK_ACCOUNT | Critical risk, block account |

## Error Handling

The API returns standard HTTP status codes:

- **200**: Success
- **400**: Bad Request (invalid input)
- **500**: Internal Server Error (model/prediction error)
- **503**: Service Unavailable (model not loaded)

Error responses include details:
```json
{
  "detail": "Error description"
}
```

## Production Deployment

For production deployment, consider:

1. **HTTPS**: Use SSL/TLS encryption
2. **Authentication**: Implement API key or OAuth
3. **Rate Limiting**: Prevent abuse
4. **Load Balancing**: For high availability
5. **Monitoring**: Logs, metrics, alerts
6. **CORS**: Configure for web applications

## Interactive Documentation

Visit `http://localhost:8000/docs` for interactive API documentation with Swagger UI. 