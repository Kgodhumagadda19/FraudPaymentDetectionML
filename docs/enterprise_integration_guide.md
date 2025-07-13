# Enterprise Fraud Detection System - Integration Guide

## **Overview** 🎯

This guide provides step-by-step instructions for integrating the fraud detection system into enterprise applications. The system supports real-time fraud detection for multiple fraud types with sub-second response times.

## **System Architecture** 🏗️

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Enterprise    │    │   Fraud         │    │   Database      │
│   Application   │───▶│   Detection     │───▶│   (PostgreSQL/  │
│                 │    │   API           │    │    MongoDB)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Monitoring    │    │   Model         │    │   Logging       │
│   Dashboard     │    │   Training      │    │   System        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## **Prerequisites** ✅

### **System Requirements:**
- **Python 3.8+**
- **8GB+ RAM**
- **4+ CPU cores**
- **100GB+ storage**
- **Network connectivity**

### **Dependencies:**
```bash
pip install fastapi uvicorn pandas numpy scikit-learn xgboost lightgbm joblib
```

## **Step 1: Data Integration** 📊

### **1.1 Payment Processor Integration**

```python
# Example: Stripe Integration
import stripe
from fraud_detection_api import predict_fraud

stripe.api_key = 'your_stripe_secret_key'

def process_payment_with_fraud_check(payment_intent_id):
    # Get payment data from Stripe
    payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
    
    # Prepare transaction data
    transaction_data = {
        "transaction_id": payment_intent.id,
        "timestamp": datetime.now().isoformat(),
        "amount": payment_intent.amount / 100,  # Convert from cents
        "user_id": payment_intent.customer,
        "merchant_id": payment_intent.metadata.get('merchant_id'),
        "merchant_category": payment_intent.metadata.get('category'),
        "card_present": payment_intent.payment_method_details.card.present,
        "used_chip": payment_intent.payment_method_details.card.chip,
        "foreign_transaction": payment_intent.payment_method_details.card.country != 'US'
    }
    
    # Check for fraud
    fraud_result = predict_fraud(transaction_data)
    
    if fraud_result['risk_level'] in ['HIGH', 'CRITICAL']:
        # Decline transaction
        stripe.PaymentIntent.cancel(payment_intent_id)
        return {"status": "declined", "reason": "fraud_detected"}
    
    return {"status": "approved"}
```

### **1.2 Banking System Integration**

```python
# Example: ACH Transaction Processing
def process_ach_transaction(transaction_data):
    # Enrich transaction data
    enriched_data = enrich_transaction_data(transaction_data)
    
    # Fraud check
    fraud_result = predict_fraud(enriched_data)
    
    # Decision logic
    if fraud_result['fraud_probability'] > 0.7:
        return {"action": "block", "reason": "high_fraud_risk"}
    elif fraud_result['fraud_probability'] > 0.3:
        return {"action": "review", "reason": "moderate_risk"}
    else:
        return {"action": "approve", "reason": "low_risk"}
```

### **1.3 Real-time Data Streaming**

```python
# Example: Kafka Integration
from kafka import KafkaConsumer, KafkaProducer
import json

def fraud_detection_stream():
    consumer = KafkaConsumer('transactions', 
                           bootstrap_servers=['localhost:9092'],
                           value_deserializer=lambda m: json.loads(m.decode('utf-8')))
    
    producer = KafkaProducer(bootstrap_servers=['localhost:9092'],
                           value_serializer=lambda v: json.dumps(v).encode('utf-8'))
    
    for message in consumer:
        transaction = message.value
        
        # Fraud detection
        fraud_result = predict_fraud(transaction)
        
        # Send result to results topic
        producer.send('fraud_results', fraud_result)
```

## **Step 2: API Integration** 🔌

### **2.1 REST API Endpoints**

```bash
# Health Check
GET /health

# Single Transaction Prediction
POST /predict
{
    "transaction_id": "TXN_123456",
    "timestamp": "2024-01-15T10:30:00Z",
    "amount": 150.00,
    "user_id": "USER_001",
    "merchant_id": "MERCH_001",
    "merchant_category": "electronics",
    "distance_from_home_km": 25.5,
    "foreign_transaction": false
}

# Batch Prediction
POST /batch-predict
{
    "transactions": [
        {
            "transaction_id": "TXN_123456",
            "amount": 150.00,
            ...
        }
    ]
}
```

### **2.2 Client Integration Examples**

#### **JavaScript/Node.js:**
```javascript
const axios = require('axios');

async function checkFraud(transactionData) {
    try {
        const response = await axios.post('http://localhost:8000/predict', transactionData);
        return response.data;
    } catch (error) {
        console.error('Fraud check failed:', error);
        throw error;
    }
}

// Usage
const transaction = {
    transaction_id: "TXN_123456",
    amount: 150.00,
    user_id: "USER_001",
    merchant_id: "MERCH_001",
    merchant_category: "electronics"
};

const fraudResult = await checkFraud(transaction);
console.log('Fraud probability:', fraudResult.fraud_probability);
```

#### **Python:**
```python
import requests

def check_fraud(transaction_data):
    response = requests.post('http://localhost:8000/predict', json=transaction_data)
    return response.json()

# Usage
transaction = {
    "transaction_id": "TXN_123456",
    "amount": 150.00,
    "user_id": "USER_001",
    "merchant_id": "MERCH_001",
    "merchant_category": "electronics"
}

fraud_result = check_fraud(transaction)
print(f"Fraud probability: {fraud_result['fraud_probability']}")
```

#### **Java:**
```java
import org.springframework.web.client.RestTemplate;

public class FraudDetectionClient {
    private RestTemplate restTemplate = new RestTemplate();
    private String baseUrl = "http://localhost:8000";
    
    public FraudPrediction checkFraud(TransactionData transaction) {
        return restTemplate.postForObject(
            baseUrl + "/predict", 
            transaction, 
            FraudPrediction.class
        );
    }
}
```

## **Step 3: Database Integration** 🗄️

### **3.1 PostgreSQL Schema**

```sql
-- Transactions table
CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) UNIQUE NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    user_id VARCHAR(50) NOT NULL,
    merchant_id VARCHAR(50) NOT NULL,
    merchant_category VARCHAR(100),
    fraud_probability DECIMAL(5,4),
    risk_level VARCHAR(20),
    recommended_action VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Fraud alerts table
CREATE TABLE fraud_alerts (
    id SERIAL PRIMARY KEY,
    transaction_id VARCHAR(50) REFERENCES transactions(transaction_id),
    alert_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    description TEXT,
    status VARCHAR(20) DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model performance table
CREATE TABLE model_performance (
    id SERIAL PRIMARY KEY,
    model_type VARCHAR(50) NOT NULL,
    fraud_type VARCHAR(50) NOT NULL,
    accuracy DECIMAL(5,4),
    roc_auc DECIMAL(5,4),
    training_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

### **3.2 MongoDB Collections**

```javascript
// Transactions collection
db.createCollection("transactions", {
    validator: {
        $jsonSchema: {
            bsonType: "object",
            required: ["transaction_id", "timestamp", "amount", "user_id"],
            properties: {
                transaction_id: { bsonType: "string" },
                timestamp: { bsonType: "date" },
                amount: { bsonType: "double" },
                user_id: { bsonType: "string" },
                fraud_probability: { bsonType: "double" },
                risk_level: { bsonType: "string" }
            }
        }
    }
});

// Create indexes
db.transactions.createIndex({ "transaction_id": 1 }, { unique: true });
db.transactions.createIndex({ "timestamp": 1 });
db.transactions.createIndex({ "user_id": 1 });
db.transactions.createIndex({ "fraud_probability": 1 });
```

## **Step 4: Monitoring & Alerting** 📊

### **4.1 Prometheus Metrics**

```python
from prometheus_client import Counter, Histogram, Gauge

# Metrics
fraud_predictions_total = Counter('fraud_predictions_total', 'Total fraud predictions')
fraud_detected_total = Counter('fraud_detected_total', 'Total fraud detected')
prediction_duration = Histogram('prediction_duration_seconds', 'Prediction duration')
model_accuracy = Gauge('model_accuracy', 'Model accuracy')

# Usage in API
@app.post("/predict")
async def predict_fraud(transaction: TransactionData):
    start_time = time.time()
    
    # Make prediction
    prediction = fraud_model.predict_fraud(transaction)
    
    # Record metrics
    fraud_predictions_total.inc()
    prediction_duration.observe(time.time() - start_time)
    
    if prediction['fraud_probability'] > 0.5:
        fraud_detected_total.inc()
    
    return prediction
```

### **4.2 Grafana Dashboard**

```json
{
  "dashboard": {
    "title": "Fraud Detection Metrics",
    "panels": [
      {
        "title": "Predictions per Second",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(fraud_predictions_total[5m])",
            "legendFormat": "Predictions/sec"
          }
        ]
      },
      {
        "title": "Fraud Detection Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(fraud_detected_total[5m]) / rate(fraud_predictions_total[5m])",
            "legendFormat": "Fraud Rate"
          }
        ]
      }
    ]
  }
}
```

### **4.3 Alerting Rules**

```yaml
# alertmanager.yml
groups:
  - name: fraud_detection_alerts
    rules:
      - alert: HighFraudRate
        expr: rate(fraud_detected_total[5m]) / rate(fraud_predictions_total[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High fraud detection rate"
          description: "Fraud rate is {{ $value }}%"
      
      - alert: ModelAccuracyLow
        expr: model_accuracy < 0.8
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Model accuracy is low"
          description: "Model accuracy is {{ $value }}"
```

## **Step 5: Security & Compliance** 🔒

### **5.1 Authentication & Authorization**

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.post("/predict")
async def predict_fraud(transaction: TransactionData, user=Depends(verify_token)):
    # Check user permissions
    if not has_permission(user, "fraud_detection"):
        raise HTTPException(status_code=403, detail="Insufficient permissions")
    
    return fraud_model.predict_fraud(transaction)
```

### **5.2 Data Encryption**

```python
from cryptography.fernet import Fernet

# Encrypt sensitive data
def encrypt_sensitive_data(data):
    key = Fernet.generate_key()
    f = Fernet(key)
    encrypted_data = f.encrypt(json.dumps(data).encode())
    return encrypted_data, key

# Decrypt sensitive data
def decrypt_sensitive_data(encrypted_data, key):
    f = Fernet(key)
    decrypted_data = f.decrypt(encrypted_data)
    return json.loads(decrypted_data.decode())
```

### **5.3 Audit Logging**

```python
import logging
from datetime import datetime

# Configure audit logging
audit_logger = logging.getLogger('audit')
audit_logger.setLevel(logging.INFO)

def log_audit_event(event_type, user_id, transaction_id, details):
    audit_logger.info({
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        'user_id': user_id,
        'transaction_id': transaction_id,
        'details': details,
        'ip_address': request.client.host
    })

@app.post("/predict")
async def predict_fraud(transaction: TransactionData, user=Depends(verify_token)):
    # Log prediction request
    log_audit_event('fraud_prediction', user['user_id'], transaction.transaction_id, {
        'amount': transaction.amount,
        'merchant_category': transaction.merchant_category
    })
    
    prediction = fraud_model.predict_fraud(transaction)
    
    # Log prediction result
    log_audit_event('fraud_result', user['user_id'], transaction.transaction_id, {
        'fraud_probability': prediction['fraud_probability'],
        'risk_level': prediction['risk_level']
    })
    
    return prediction
```

## **Step 6: Deployment** 🚀

### **6.1 Docker Deployment**

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "fraud_detection_api:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  fraud-detection-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/fraud_detection
    depends_on:
      - db
    volumes:
      - ./models:/app/models
  
  db:
    image: postgres:13
    environment:
      - POSTGRES_DB=fraud_detection
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### **6.2 Kubernetes Deployment**

```yaml
# fraud-detection-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: fraud-detection-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: fraud-detection-api
  template:
    metadata:
      labels:
        app: fraud-detection-api
    spec:
      containers:
      - name: fraud-detection-api
        image: fraud-detection:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: fraud-detection-secrets
              key: database-url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: fraud-detection-service
spec:
  selector:
    app: fraud-detection-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## **Step 7: Testing** 🧪

### **7.1 Unit Tests**

```python
import pytest
from fraud_detection_model import FraudDetectionModel

def test_fraud_detection_model():
    model = FraudDetectionModel(model_type='xgboost')
    
    # Test data
    test_data = pd.DataFrame({
        'transaction_id': ['TXN_001'],
        'amount': [1000.0],
        'distance_from_home_km': [200.0],
        'foreign_transaction': [True],
        'is_fraud': [1]
    })
    
    # Train model
    X_test, y_test, y_pred, y_pred_proba = model.train_model(test_data, 'credit_card')
    
    # Assertions
    assert len(y_pred) == len(y_test)
    assert all(isinstance(pred, (int, np.integer)) for pred in y_pred)

def test_api_endpoints():
    from fastapi.testclient import TestClient
    from fraud_detection_api import app
    
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    
    # Test prediction endpoint
    transaction_data = {
        "transaction_id": "TXN_001",
        "amount": 100.0,
        "user_id": "USER_001",
        "merchant_id": "MERCH_001",
        "merchant_category": "electronics"
    }
    
    response = client.post("/predict", json=transaction_data)
    assert response.status_code == 200
    assert "fraud_probability" in response.json()
```

### **7.2 Load Testing**

```python
import asyncio
import aiohttp
import time

async def load_test():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i in range(1000):
            transaction_data = {
                "transaction_id": f"TXN_{i}",
                "amount": 100.0 + i,
                "user_id": f"USER_{i % 100}",
                "merchant_id": f"MERCH_{i % 50}",
                "merchant_category": "electronics"
            }
            task = session.post('http://localhost:8000/predict', json=transaction_data)
            tasks.append(task)
        
        start_time = time.time()
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        print(f"Processed {len(responses)} requests in {end_time - start_time:.2f} seconds")
        print(f"Average response time: {(end_time - start_time) / len(responses) * 1000:.2f} ms")

# Run load test
asyncio.run(load_test())
```

## **Step 8: Production Checklist** ✅

### **Before Going Live:**

- [ ] **Model Training**: Train models with production data
- [ ] **Performance Testing**: Load test with expected traffic
- [ ] **Security Review**: Penetration testing and security audit
- [ ] **Compliance Check**: Ensure GDPR, PCI DSS compliance
- [ ] **Monitoring Setup**: Prometheus, Grafana, alerting
- [ ] **Backup Strategy**: Database backups and disaster recovery
- [ ] **Documentation**: API documentation and runbooks
- [ ] **Team Training**: Train operations team
- [ ] **Rollback Plan**: Plan for quick rollback if issues arise

### **Post-Deployment:**

- [ ] **Performance Monitoring**: Monitor response times and throughput
- [ ] **Accuracy Tracking**: Track model accuracy over time
- [ ] **Fraud Pattern Analysis**: Analyze new fraud patterns
- [ ] **Model Retraining**: Regular model retraining schedule
- [ ] **Security Updates**: Regular security patches
- [ ] **Capacity Planning**: Monitor resource usage and scale as needed

## **Support & Maintenance** 🛠️

### **Contact Information:**
- **Technical Support**: support@frauddetection.com
- **Documentation**: https://docs.frauddetection.com
- **GitHub**: https://github.com/frauddetection/enterprise

### **Regular Maintenance:**
- **Weekly**: Performance review and optimization
- **Monthly**: Model retraining and accuracy assessment
- **Quarterly**: Security audit and compliance review
- **Annually**: Architecture review and capacity planning

---

**This integration guide provides a comprehensive framework for deploying the fraud detection system in enterprise environments. Follow each step carefully and ensure proper testing before going live.** 