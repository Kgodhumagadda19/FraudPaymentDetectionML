import requests
import json
import time
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Test the health check endpoint"""
    print("🏥 Testing Health Endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data['status']}")
            print(f"   Model Type: {data['model_type']}")
            print(f"   Fraud Types: {data['fraud_types']}")
            print(f"   Uptime: {data['uptime_seconds']:.1f} seconds")
            return True
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False

def test_single_prediction():
    """Test single transaction prediction"""
    print("\n🔍 Testing Single Transaction Prediction...")
    
    # Sample transaction data
    transaction = {
        "transaction_id": "TXN_TEST_001",
        "timestamp": datetime.now().isoformat(),
        "amount": 1500.00,
        "user_id": "USER_12345",
        "merchant_id": "MERCH_67890",
        "merchant_category": "electronics",
        "distance_from_home_km": 25.5,
        "velocity_24h": 3,
        "foreign_transaction": False,
        "online_order": True,
        "high_risk_merchant": False,
        "transaction_count_user": 12,
        "card_present": False,
        "used_chip": False,
        "used_pin": False,
        "card_type": "credit",
        "ip_address": "192.168.1.100",
        "device_id": "DEV_001"
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=transaction)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction Successful!")
            print(f"   Transaction ID: {data['transaction_id']}")
            print(f"   Fraud Probability: {data['fraud_probability']:.4f}")
            print(f"   Risk Level: {data['risk_level']}")
            print(f"   Recommended Action: {data['recommended_action']}")
            print(f"   Processing Time: {data['processing_time_ms']} ms")
            return True
        else:
            print(f"❌ Prediction Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Prediction Error: {e}")
        return False

def test_batch_prediction():
    """Test batch transaction prediction"""
    print("\n📦 Testing Batch Transaction Prediction...")
    
    # Sample batch of transactions
    transactions = [
        {
            "transaction_id": "TXN_BATCH_001",
            "timestamp": datetime.now().isoformat(),
            "amount": 500.00,
            "user_id": "USER_001",
            "merchant_id": "MERCH_001",
            "merchant_category": "food",
            "distance_from_home_km": 5.0,
            "velocity_24h": 1,
            "foreign_transaction": False,
            "online_order": False,
            "high_risk_merchant": False,
            "transaction_count_user": 5,
            "card_present": True,
            "used_chip": True,
            "used_pin": False,
            "card_type": "debit",
            "ip_address": "192.168.1.101",
            "device_id": "DEV_002"
        },
        {
            "transaction_id": "TXN_BATCH_002",
            "timestamp": datetime.now().isoformat(),
            "amount": 2500.00,
            "user_id": "USER_002",
            "merchant_id": "MERCH_002",
            "merchant_category": "electronics",
            "distance_from_home_km": 150.0,
            "velocity_24h": 8,
            "foreign_transaction": True,
            "online_order": True,
            "high_risk_merchant": True,
            "transaction_count_user": 25,
            "card_present": False,
            "used_chip": False,
            "used_pin": False,
            "card_type": "credit",
            "ip_address": "203.0.113.1",
            "device_id": "DEV_003"
        }
    ]
    
    batch_request = {"transactions": transactions}
    
    try:
        response = requests.post(f"{BASE_URL}/batch-predict", json=batch_request)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch Prediction Successful!")
            print(f"   Total Transactions: {data['total_transactions']}")
            print(f"   Fraud Count: {data['fraud_count']}")
            print(f"   Processing Time: {data['processing_time_ms']} ms")
            print(f"   Risk Distribution: {data['risk_distribution']}")
            
            # Show individual predictions
            for i, pred in enumerate(data['predictions']):
                print(f"   Transaction {i+1}: {pred['transaction_id']}")
                print(f"     Fraud Probability: {pred['fraud_probability']:.4f}")
                print(f"     Risk Level: {pred['risk_level']}")
            
            return True
        else:
            print(f"❌ Batch Prediction Failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Batch Prediction Error: {e}")
        return False

def test_model_info():
    """Test model information endpoint"""
    print("\n📊 Testing Model Information...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model Info Retrieved!")
            print(f"   Model Type: {data['model_type']}")
            print(f"   Fraud Types: {data['fraud_types']}")
            
            # Show performance metrics
            if 'performance' in data:
                print(f"   Performance Metrics:")
                for fraud_type, metrics in data['performance'].items():
                    print(f"     {fraud_type}:")
                    print(f"       Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                    print(f"       ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
            
            return True
        else:
            print(f"❌ Model Info Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Model Info Error: {e}")
        return False

def test_api_documentation():
    """Test API documentation endpoint"""
    print("\n📚 Testing API Documentation...")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print(f"✅ API Documentation Available!")
            print(f"   Visit: {BASE_URL}/docs")
            return True
        else:
            print(f"❌ API Documentation Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ API Documentation Error: {e}")
        return False

def main():
    """Run all API tests"""
    print("🚀 ENTERPRISE FRAUD DETECTION API TESTING")
    print("=" * 50)
    
    # Wait a moment for API to start
    print("⏳ Waiting for API to start...")
    time.sleep(3)
    
    # Run tests
    tests = [
        test_health_endpoint,
        test_single_prediction,
        test_batch_prediction,
        test_model_info,
        test_api_documentation
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\n📈 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! API is ready for enterprise use.")
        print(f"\n🌐 API Endpoints:")
        print(f"   Health Check: {BASE_URL}/health")
        print(f"   Single Prediction: {BASE_URL}/predict")
        print(f"   Batch Prediction: {BASE_URL}/batch-predict")
        print(f"   Model Info: {BASE_URL}/model-info")
        print(f"   Documentation: {BASE_URL}/docs")
    else:
        print("⚠️  Some tests failed. Check API logs for details.")

if __name__ == "__main__":
    main() 