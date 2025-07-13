#!/usr/bin/env python3
"""
Deployment script for the Fraud Detection API
Helps with production deployment configuration
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

def create_production_config(host="0.0.0.0", port=8000, workers=4):
    """Create production configuration"""
    
    config = f"""
# Production configuration for Fraud Detection API
import uvicorn
from fraud_detection_api import app

if __name__ == "__main__":
    uvicorn.run(
        "fraud_detection_api:app",
        host="{host}",
        port={port},
        workers={workers},
        reload=False,
        log_level="info",
        access_log=True
    )
"""
    
    with open("production_server.py", "w") as f:
        f.write(config)
    
    print(f"✅ Created production_server.py with host={host}, port={port}, workers={workers}")

def create_dockerfile():
    """Create a Dockerfile for containerized deployment"""
    
    dockerfile = """# Fraud Detection API Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY *.py ./
COPY *.pkl ./
COPY *.json ./

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["python", "production_server.py"]
"""
    
    with open("Dockerfile", "w") as f:
        f.write(dockerfile)
    
    print("✅ Created Dockerfile")

def create_docker_compose():
    """Create docker-compose.yml for easy deployment"""
    
    compose = """version: '3.8'

services:
  fraud-detection-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - PYTHONUNBUFFERED=1
    volumes:
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
"""
    
    with open("docker-compose.yml", "w") as f:
        f.write(compose)
    
    print("✅ Created docker-compose.yml")

def create_nginx_config():
    """Create nginx configuration for reverse proxy"""
    
    nginx_config = """server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
    
    # Health check endpoint
    location /health {
        proxy_pass http://localhost:8000/health;
        access_log off;
    }
}
"""
    
    with open("nginx.conf", "w") as f:
        f.write(nginx_config)
    
    print("✅ Created nginx.conf")

def create_systemd_service():
    """Create systemd service file"""
    
    service = """[Unit]
Description=Fraud Detection API
After=network.target

[Service]
Type=exec
User=www-data
Group=www-data
WorkingDirectory=/path/to/your/api
Environment=PATH=/path/to/your/api/venv/bin
ExecStart=/path/to/your/api/venv/bin/python production_server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    with open("fraud-detection-api.service", "w") as f:
        f.write(service)
    
    print("✅ Created fraud-detection-api.service")
    print("⚠️  Remember to update the paths in the service file")

def create_environment_file():
    """Create environment configuration file"""
    
    env_config = """# Fraud Detection API Environment Configuration

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# Model Configuration
MODEL_PATH=api_compatible_model.pkl
MODEL_TYPE=lightgbm

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/api.log

# Security (for production)
# API_KEY=your-secret-api-key
# CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Database (if needed)
# DATABASE_URL=postgresql://user:password@localhost/fraud_db

# Monitoring
# PROMETHEUS_ENABLED=true
# METRICS_PORT=9090
"""
    
    with open(".env.example", "w") as f:
        f.write(env_config)
    
    print("✅ Created .env.example")

def create_deployment_script():
    """Create deployment helper script"""
    
    script = """#!/bin/bash
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
"""
    
    with open("deploy.sh", "w") as f:
        f.write(script)
    
    # Make executable
    os.chmod("deploy.sh", 0o755)
    
    print("✅ Created deploy.sh")

def main():
    parser = argparse.ArgumentParser(description="Deploy Fraud Detection API")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--all", action="store_true", help="Create all deployment files")
    
    args = parser.parse_args()
    
    print("🚀 Fraud Detection API Deployment Setup")
    print("=" * 50)
    
    # Create production configuration
    create_production_config(args.host, args.port, args.workers)
    
    if args.all:
        print("\n📦 Creating all deployment files...")
        create_dockerfile()
        create_docker_compose()
        create_nginx_config()
        create_systemd_service()
        create_environment_file()
        create_deployment_script()
        
        print("\n✅ All deployment files created!")
        print("\n📋 Next steps:")
        print("1. Review and customize the generated files")
        print("2. Update paths in fraud-detection-api.service")
        print("3. Configure nginx.conf with your domain")
        print("4. Set up environment variables in .env")
        print("5. Run: ./deploy.sh [production|docker|systemd]")
    else:
        print("\n💡 Run with --all to create all deployment files")
        print("   python deploy_api.py --all")

if __name__ == "__main__":
    main() 