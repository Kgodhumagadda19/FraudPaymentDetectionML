#!/usr/bin/env python3
"""
Comprehensive API test that shows all working functionality
"""

import requests
import json
from datetime import datetime
import time

def test_all_endpoints():
    """Test all API endpoints and provide comprehensive results"""
    
    print("🚀 COMPREHENSIVE FRAUD DETECTION API TESTING")
    print("=" * 60)
    
    base_url = "http://localhost:8000"
    results = {}
    
    # Test 1: Root endpoint
    print("\n1️⃣  Testing Root Endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint working")
            print(f"   API: {data['message']}")
            print(f"   Version: {data['version']}")
            print(f"   Status: {data['status']}")
            results['root'] = True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            results['root'] = False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        results['root'] = False
    
    # Test 2: Health check
    print("\n2️⃣  Testing Health Check...")
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed")
            print(f"   Status: {data['status']}")
            print(f"   Model Type: {data['model_type']}")
            print(f"   Fraud Types: {len(data['fraud_types'])} types")
            print(f"   Total Models: {data['total_models']}")
            print(f"   Uptime: {data['uptime_seconds']:.1f} seconds")
            results['health'] = True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            results['health'] = False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        results['health'] = False
    
    # Test 3: Model information
    print("\n3️⃣  Testing Model Information...")
    try:
        response = requests.get(f"{base_url}/model-info")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Model info retrieved")
            print(f"   Model Type: {data['model_type']}")
            print(f"   Fraud Types: {data['fraud_types']}")
            
            if 'performance' in data and data['performance']:
                print(f"   Performance Metrics:")
                for fraud_type, metrics in data['performance'].items():
                    print(f"     {fraud_type}:")
                    print(f"       Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                    print(f"       ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
            
            results['model_info'] = True
        else:
            print(f"❌ Model info failed: {response.status_code}")
            results['model_info'] = False
    except Exception as e:
        print(f"❌ Model info error: {e}")
        results['model_info'] = False
    
    # Test 4: Metrics
    print("\n4️⃣  Testing Metrics...")
    try:
        response = requests.get(f"{base_url}/metrics")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Metrics retrieved")
            print(f"   Uptime: {data['uptime_seconds']:.1f} seconds")
            print(f"   Model Loaded: {data['model_loaded']}")
            print(f"   Total Models: {data['total_models']}")
            print(f"   API Version: {data['api_version']}")
            results['metrics'] = True
        else:
            print(f"❌ Metrics failed: {response.status_code}")
            results['metrics'] = False
    except Exception as e:
        print(f"❌ Metrics error: {e}")
        results['metrics'] = False
    
    # Test 5: API Documentation
    print("\n5️⃣  Testing API Documentation...")
    try:
        response = requests.get(f"{base_url}/docs")
        if response.status_code == 200:
            print(f"✅ API documentation available")
            print(f"   Visit: {base_url}/docs")
            results['docs'] = True
        else:
            print(f"❌ API documentation failed: {response.status_code}")
            results['docs'] = False
    except Exception as e:
        print(f"❌ API documentation error: {e}")
        results['docs'] = False
    
    # Test 6: Prediction endpoint (expected to fail)
    print("\n6️⃣  Testing Prediction Endpoint...")
    try:
        test_transaction = {
            "transaction_id": "TXN_TEST_001",
            "timestamp": datetime.now().isoformat(),
            "amount": 150.00,
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
            "device_id": "DEV_001"
        }
        
        response = requests.post(f"{base_url}/predict", json=test_transaction)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Prediction successful!")
            print(f"   Fraud Probability: {data['fraud_probability']:.4f}")
            print(f"   Risk Level: {data['risk_level']}")
            print(f"   Processing Time: {data['processing_time_ms']} ms")
            results['prediction'] = True
        else:
            print(f"❌ Prediction failed: {response.status_code}")
            print(f"   Error: {response.text[:100]}...")
            results['prediction'] = False
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        results['prediction'] = False
    
    # Test 7: Batch prediction endpoint (expected to fail)
    print("\n7️⃣  Testing Batch Prediction Endpoint...")
    try:
        test_transactions = [
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
                "device_id": "DEV_002"
            }
        ]
        
        response = requests.post(f"{base_url}/batch-predict", json={"transactions": test_transactions})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Batch prediction successful!")
            print(f"   Total Transactions: {data['total_transactions']}")
            print(f"   Fraud Count: {data['fraud_count']}")
            print(f"   Processing Time: {data['processing_time_ms']} ms")
            results['batch_prediction'] = True
        else:
            print(f"❌ Batch prediction failed: {response.status_code}")
            print(f"   Error: {response.text[:100]}...")
            results['batch_prediction'] = False
    except Exception as e:
        print(f"❌ Batch prediction error: {e}")
        results['batch_prediction'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 COMPREHENSIVE TEST RESULTS")
    print("=" * 60)
    
    working_endpoints = []
    failing_endpoints = []
    
    for endpoint, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{endpoint.replace('_', ' ').title()}: {status}")
        
        if result:
            working_endpoints.append(endpoint)
        else:
            failing_endpoints.append(endpoint)
    
    print(f"\n📈 SUMMARY:")
    print(f"   Working Endpoints: {len(working_endpoints)}/{len(results)}")
    print(f"   Failing Endpoints: {len(failing_endpoints)}/{len(results)}")
    
    if working_endpoints:
        print(f"\n✅ WORKING ENDPOINTS:")
        for endpoint in working_endpoints:
            print(f"   • {endpoint.replace('_', ' ').title()}")
    
    if failing_endpoints:
        print(f"\n❌ FAILING ENDPOINTS:")
        for endpoint in failing_endpoints:
            print(f"   • {endpoint.replace('_', ' ').title()}")
    
    print(f"\n🎯 API STATUS: {'🟢 FULLY OPERATIONAL' if len(failing_endpoints) == 0 else '🟡 PARTIALLY OPERATIONAL' if len(working_endpoints) > len(failing_endpoints) else '🔴 NOT OPERATIONAL'}")
    
    if 'prediction' in failing_endpoints or 'batch_prediction' in failing_endpoints:
        print(f"\n⚠️  PREDICTION ISSUE:")
        print(f"   The prediction endpoints are failing due to a feature mismatch.")
        print(f"   The model expects 62 features but receives different numbers.")
        print(f"   This is a known issue that can be resolved by:")
        print(f"   1. Retraining the model with the correct feature set")
        print(f"   2. Modifying the feature extraction in the API")
        print(f"   3. Using a model with shape checking disabled")
    
    print(f"\n🌐 API DOCUMENTATION: {base_url}/docs")
    print(f"📊 API METRICS: {base_url}/metrics")
    print(f"🏥 API HEALTH: {base_url}/health")

if __name__ == "__main__":
    test_all_endpoints() 