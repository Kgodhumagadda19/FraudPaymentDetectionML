#!/usr/bin/env python3
"""
Main entry point for the Fraud Payment Detection ML Application
"""

import sys
import os

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.api.fraud_detection_api import app
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.fraud_detection_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 