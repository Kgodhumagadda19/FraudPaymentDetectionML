#!/usr/bin/env python3
"""
Fraud Detection API Client Library
A simple client for integrating with the Enterprise Fraud Detection API
"""

import requests
import json
from datetime import datetime
from typing import Dict, List, Optional, Union
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FraudDetectionClient:
    """
    Client for the Enterprise Fraud Detection API
    """
    
    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        """
        Initialize the fraud detection client
        
        Args:
            base_url: Base URL of the API
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'FraudDetectionClient/1.0.0'
        })
    
    def health_check(self) -> Dict:
        """
        Check if the API is healthy
        
        Returns:
            Health status information
        """
        try:
            response = self.session.get(f"{self.base_url}/health", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            raise
    
    def predict_single(self, transaction_data: Dict) -> Dict:
        """
        Predict fraud for a single transaction
        
        Args:
            transaction_data: Transaction data dictionary
            
        Returns:
            Prediction result with fraud probability and risk assessment
        """
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=transaction_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Single prediction failed: {e}")
            raise
    
    def predict_batch(self, transactions: List[Dict]) -> Dict:
        """
        Predict fraud for multiple transactions
        
        Args:
            transactions: List of transaction data dictionaries
            
        Returns:
            Batch prediction results
        """
        try:
            response = self.session.post(
                f"{self.base_url}/batch-predict",
                json={"transactions": transactions},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Batch prediction failed: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """
        Get information about the trained models
        
        Returns:
            Model information and performance metrics
        """
        try:
            response = self.session.get(f"{self.base_url}/model-info", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Model info request failed: {e}")
            raise
    
    def get_metrics(self) -> Dict:
        """
        Get API performance metrics
        
        Returns:
            API metrics and performance statistics
        """
        try:
            response = self.session.get(f"{self.base_url}/metrics", timeout=self.timeout)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Metrics request failed: {e}")
            raise
    
    def create_transaction(
        self,
        transaction_id: str,
        amount: float,
        user_id: str,
        timestamp: Optional[str] = None,
        merchant_id: Optional[str] = None,
        merchant_category: Optional[str] = None,
        distance_from_home_km: Optional[float] = None,
        velocity_24h: Optional[int] = None,
        foreign_transaction: bool = False,
        online_order: bool = False,
        high_risk_merchant: bool = False,
        transaction_count_user: Optional[int] = None,
        card_present: bool = True,
        used_chip: bool = True,
        used_pin: bool = False,
        card_type: Optional[str] = None,
        device_id: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Create a transaction data dictionary with proper formatting
        
        Args:
            transaction_id: Unique transaction identifier
            amount: Transaction amount
            user_id: User identifier
            timestamp: Transaction timestamp (ISO format)
            merchant_id: Merchant identifier
            merchant_category: Merchant category
            distance_from_home_km: Distance from user's home location
            velocity_24h: Number of transactions in last 24 hours
            foreign_transaction: Whether transaction is foreign
            online_order: Whether it's an online order
            high_risk_merchant: Whether merchant is high risk
            transaction_count_user: Total transactions for user
            card_present: Whether card was physically present
            used_chip: Whether chip was used
            used_pin: Whether PIN was used
            card_type: Type of card used
            device_id: Device identifier
            **kwargs: Additional transaction fields
            
        Returns:
            Formatted transaction data dictionary
        """
        if timestamp is None:
            timestamp = datetime.now().isoformat()
        
        transaction = {
            "transaction_id": transaction_id,
            "timestamp": timestamp,
            "amount": amount,
            "user_id": user_id,
            "merchant_id": merchant_id,
            "merchant_category": merchant_category,
            "distance_from_home_km": distance_from_home_km,
            "velocity_24h": velocity_24h,
            "foreign_transaction": foreign_transaction,
            "online_order": online_order,
            "high_risk_merchant": high_risk_merchant,
            "transaction_count_user": transaction_count_user,
            "card_present": card_present,
            "used_chip": used_chip,
            "used_pin": used_pin,
            "card_type": card_type,
            "device_id": device_id
        }
        
        # Add any additional fields
        transaction.update(kwargs)
        
        # Remove None values
        transaction = {k: v for k, v in transaction.items() if v is not None}
        
        return transaction
    
    def is_high_risk(self, prediction: Dict, threshold: float = 0.3) -> bool:
        """
        Check if a prediction indicates high risk
        
        Args:
            prediction: Prediction result from API
            threshold: Risk threshold (default 0.3)
            
        Returns:
            True if high risk, False otherwise
        """
        return prediction.get('fraud_probability', 0) > threshold
    
    def get_risk_summary(self, prediction: Dict) -> str:
        """
        Get a human-readable risk summary
        
        Args:
            prediction: Prediction result from API
            
        Returns:
            Risk summary string
        """
        prob = prediction.get('fraud_probability', 0)
        risk_level = prediction.get('risk_level', 'UNKNOWN')
        action = prediction.get('recommended_action', 'UNKNOWN')
        
        return f"Risk Level: {risk_level} ({prob:.1%}), Action: {action}"
    
    def close(self):
        """Close the client session"""
        self.session.close()

# Example usage and testing
def main():
    """Example usage of the FraudDetectionClient"""
    
    # Initialize client
    client = FraudDetectionClient()
    
    try:
        # Check API health
        print("🔍 Checking API health...")
        health = client.health_check()
        print(f"✅ API Status: {health['status']}")
        print(f"   Model Type: {health['model_type']}")
        print(f"   Fraud Types: {health['fraud_types']}")
        print(f"   Uptime: {health['uptime_seconds']:.1f} seconds")
        
        # Get model information
        print("\n📊 Getting model information...")
        model_info = client.get_model_info()
        print(f"✅ Model Type: {model_info['model_type']}")
        print(f"   Fraud Types: {model_info['fraud_types']}")
        
        # Create a test transaction
        print("\n💳 Testing single prediction...")
        transaction = client.create_transaction(
            transaction_id="TXN_TEST_001",
            amount=150.75,
            user_id="USER_1234",
            merchant_id="MERCH_001",
            merchant_category="electronics",
            distance_from_home_km=25.5,
            velocity_24h=3,
            foreign_transaction=False,
            online_order=True,
            high_risk_merchant=False,
            transaction_count_user=15,
            card_present=False,
            used_chip=False,
            used_pin=False,
            card_type="visa",
            device_id="DEV_001"
        )
        
        # Make prediction
        prediction = client.predict_single(transaction)
        print(f"✅ Prediction successful!")
        print(f"   Transaction ID: {prediction['transaction_id']}")
        print(f"   Fraud Probability: {prediction['fraud_probability']:.4f}")
        print(f"   Risk Level: {prediction['risk_level']}")
        print(f"   Recommended Action: {prediction['recommended_action']}")
        print(f"   Processing Time: {prediction['processing_time_ms']} ms")
        
        # Check if high risk
        is_high = client.is_high_risk(prediction)
        print(f"   High Risk: {'Yes' if is_high else 'No'}")
        
        # Get risk summary
        summary = client.get_risk_summary(prediction)
        print(f"   Summary: {summary}")
        
        # Test batch prediction
        print("\n📦 Testing batch prediction...")
        transactions = [
            client.create_transaction(
                transaction_id="TXN_BATCH_001",
                amount=500.00,
                user_id="USER_001",
                merchant_category="food",
                distance_from_home_km=5.0,
                velocity_24h=1,
                online_order=False,
                card_present=True,
                used_chip=True,
                card_type="debit"
            ),
            client.create_transaction(
                transaction_id="TXN_BATCH_002",
                amount=2500.00,
                user_id="USER_002",
                merchant_category="electronics",
                distance_from_home_km=150.0,
                velocity_24h=8,
                foreign_transaction=True,
                online_order=True,
                high_risk_merchant=True,
                transaction_count_user=25,
                card_present=False,
                card_type="credit"
            )
        ]
        
        batch_result = client.predict_batch(transactions)
        print(f"✅ Batch prediction successful!")
        print(f"   Total Transactions: {batch_result['total_transactions']}")
        print(f"   Fraud Count: {batch_result['fraud_count']}")
        print(f"   Processing Time: {batch_result['processing_time_ms']} ms")
        print(f"   Risk Distribution: {batch_result['risk_distribution']}")
        
        # Get API metrics
        print("\n📈 Getting API metrics...")
        metrics = client.get_metrics()
        print(f"✅ API Metrics:")
        print(f"   Uptime: {metrics['uptime_seconds']:.1f} seconds")
        print(f"   Model Loaded: {metrics['model_loaded']}")
        print(f"   Total Models: {metrics['total_models']}")
        print(f"   API Version: {metrics['api_version']}")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        client.close()

if __name__ == "__main__":
    main() 