
# Production configuration for Fraud Detection API
import os
import uvicorn
from fraud_detection_api import app

if __name__ == "__main__":
    # Get port from environment variable (for cloud platforms)
    port = int(os.environ.get("PORT", 8000))
    
    uvicorn.run(
        "fraud_detection_api:app",
        host="0.0.0.0",
        port=port,
        workers=1,  # Reduced for cloud platforms
        reload=False,
        log_level="info",
        access_log=True
    )
