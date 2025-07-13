#!/usr/bin/env python3
"""
Test the API with the correct feature structure
"""

import requests
import json
from datetime import datetime

def test_api_prediction():
    """Test the API prediction endpoint"""
    
    # Test transaction data with all required fields
    test_transaction = {
        "transaction_id": "TXN_123456",
        "timestamp": "2024-01-15T14:30:00Z",
        "amount": 150.75,
        "user_id": "USER_1234",
        "email": "user@example.com",
        "phone": "+1-555-123-4567",
        "ssn": "123-45-6789",
        "ip_address": "192.168.1.100",
        "address": "123 Main St, New York, NY",
        "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "device_type": "desktop",
        "browser_type": "chrome",
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
    
    try:
        # Make prediction request
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_transaction,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Prediction Successful!")
            print(f"Transaction ID: {result['transaction_id']}")
            print(f"Fraud Probability: {result['fraud_probability']:.4f}")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Recommended Action: {result['recommended_action']}")
            print(f"Processing Time: {result['processing_time_ms']}ms")
            return True
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Error details: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_api_health():
    """Test the API health endpoint"""
    
    try:
        response = requests.get("http://localhost:8000/health")
        
        if response.status_code == 200:
            health = response.json()
            print("✅ API Health Check:")
            print(f"Status: {health['status']}")
            print(f"Model Type: {health['model_type']}")
            print(f"Fraud Types: {health['fraud_types']}")
            print(f"Total Models: {health['total_models']}")
            print(f"Uptime: {health['uptime_seconds']:.1f} seconds")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    
    # Create multiple test transactions
    test_transactions = [
        {
            "transaction_id": f"TXN_{i:06d}",
            "timestamp": "2024-01-15T14:30:00Z",
            "amount": 100.0 + i * 10,
            "user_id": f"USER_{i}",
            "email": f"user{i}@example.com",
            "phone": f"+1-555-{i:03d}-4567",
            "ssn": f"123-45-{i:04d}",
            "ip_address": f"192.168.1.{i % 255}",
            "address": f"{i} Main St, New York, NY",
            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "device_type": "desktop",
            "browser_type": "chrome",
            "merchant_id": f"MERCH_{i:03d}",
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
            "device_id": f"DEV_{i:03d}"
        }
        for i in range(1, 4)  # Test with 3 transactions
    ]
    
    try:
        # Make batch prediction request
        response = requests.post(
            "http://localhost:8000/batch-predict",
            json={"transactions": test_transactions},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Batch Prediction Successful!")
            print(f"Total Transactions: {result['total_transactions']}")
            print(f"Fraud Count: {result['fraud_count']}")
            print(f"Processing Time: {result['processing_time_ms']}ms")
            print(f"Risk Distribution: {result['risk_distribution']}")
            
            print("\nIndividual Predictions:")
            for i, pred in enumerate(result['predictions']):
                print(f"  {i+1}. {pred['transaction_id']}: {pred['fraud_probability']:.4f} ({pred['risk_level']})")
            
            return True
        else:
            print(f"❌ Batch API Error: {response.status_code}")
            print(f"Error details: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Make sure it's running on localhost:8000")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🧪 TESTING FRAUD DETECTION API")
    print("=" * 50)
    
    # Test health endpoint
    print("\n1. Testing Health Endpoint...")
    health_ok = test_api_health()
    
    if not health_ok:
        print("\n❌ Health check failed. Please start the API first:")
        print("   python fraud_detection_api.py")
        return
    
    # Test single prediction
    print("\n2. Testing Single Prediction...")
    prediction_ok = test_api_prediction()
    
    # Test batch prediction
    print("\n3. Testing Batch Prediction...")
    batch_ok = test_batch_prediction()
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    print(f"Single Prediction: {'✅ PASS' if prediction_ok else '❌ FAIL'}")
    print(f"Batch Prediction: {'✅ PASS' if batch_ok else '❌ FAIL'}")
    
    if all([health_ok, prediction_ok, batch_ok]):
        print("\n🎉 All tests passed! API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the API logs for details.")

if __name__ == "__main__":
    main() 