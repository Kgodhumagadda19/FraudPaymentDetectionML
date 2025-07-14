
# Production configuration for Fraud Detection API
import os
import uvicorn
import sys

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.api.fraud_detection_api import app

if __name__ == "__main__":
    # Get port from environment variable (for cloud platforms)
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "src.api.fraud_detection_api:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # Reduced for cloud platforms
        reload=False,
        log_level="info",
        access_log=True
    )
