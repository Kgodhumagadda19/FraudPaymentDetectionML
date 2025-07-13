#!/bin/bash
# Fraud Detection API Deployment Script

set -e

echo "🚀 Deploying Fraud Detection API..."

# Check if running in production mode
if [ "$1" = "production" ]; then
    echo "📦 Production deployment mode"
    
    # Create logs directory
    mkdir -p logs
    
    # Set environment variables
    export PYTHONUNBUFFERED=1
    
    # Start the API
    echo "🔄 Starting API server..."
    python production_server.py
    
elif [ "$1" = "docker" ]; then
    echo "🐳 Docker deployment mode"
    
    # Build and run with Docker Compose
    docker-compose up --build -d
    
    echo "✅ API deployed with Docker"
    echo "📊 Check status: docker-compose ps"
    echo "📋 View logs: docker-compose logs -f"
    
elif [ "$1" = "systemd" ]; then
    echo "⚙️  Systemd deployment mode"
    
    # Copy service file
    sudo cp fraud-detection-api.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable fraud-detection-api
    sudo systemctl start fraud-detection-api
    
    echo "✅ API deployed with systemd"
    echo "📊 Check status: sudo systemctl status fraud-detection-api"
    echo "📋 View logs: sudo journalctl -u fraud-detection-api -f"
    
else
    echo "❌ Invalid deployment mode. Use: production, docker, or systemd"
    echo "Usage: ./deploy.sh [production|docker|systemd]"
    exit 1
fi

echo "🎉 Deployment completed!"
