# Fraud Payment Detection ML

A comprehensive machine learning system for detecting fraudulent payment transactions in real-time.

## 🏗️ Project Structure

```
FraudPaymentDetectionML/
├── src/                    # Source code
│   ├── api/               # API-related code
│   │   ├── fraud_detection_api.py    # Main FastAPI application
│   │   └── fraud_detection_client.py # API client library
│   ├── models/            # ML models and training
│   │   ├── fraud_detection_model.py  # Core model class
│   │   ├── enhanced_fraud_training.py # Advanced training pipeline
│   │   ├── synthetic_pii_fraud_training.py # Synthetic data training
│   │   ├── comprehensive_fraud_training.py # Comprehensive training
│   │   └── train_fraud_model.py      # Basic training script
│   ├── data/              # Data processing and schemas
│   │   ├── fraud_detection_data.py   # Data loading utilities
│   │   ├── fraud_data_schema.py      # Data schemas
│   │   ├── extract_current_features.py # Feature extraction
│   │   └── analyze_model_features.py # Feature analysis
│   ├── utils/             # Utility functions
│   │   ├── fix_model_compatibility.py # Model compatibility fixes
│   │   ├── fix_prediction_api.py     # API fixes
│   │   ├── print_model_features.py   # Model inspection
│   │   ├── inspect_model_file.py     # Model file inspection
│   │   └── find_fraud_datasets.py    # Dataset discovery
│   └── config/            # Configuration files
│       ├── model_features.json       # Model feature definitions
│       ├── current_model_features.json # Current model features
│       ├── enhanced_model_results.json # Model results
│       ├── fraud_dataset_info.json   # Dataset information
│       └── model_results.json        # Model performance results
├── tests/                 # Test files
│   ├── test_api.py                    # API tests
│   ├── test_api_simple.py             # Simple API tests
│   ├── test_api_comprehensive.py      # Comprehensive API tests
│   └── test_api_features.py           # Feature tests
├── docs/                  # Documentation
│   ├── API_INTEGRATION_GUIDE.md       # API integration guide
│   ├── enterprise_integration_guide.md # Enterprise integration
│   └── enterprise_pii_compliance_guide.md # PII compliance guide
├── deployment/            # Deployment configurations
│   ├── Dockerfile                     # Docker configuration
│   ├── docker-compose.yml             # Docker Compose
│   ├── nginx.conf                     # Nginx configuration
│   ├── fraud-detection-api.service    # Systemd service
│   ├── deploy.sh                      # Deployment script
│   └── deploy_api.py                  # Deployment automation
├── scripts/               # Utility scripts
│   ├── production_server.py           # Production server
│   ├── Procfile                       # Heroku Procfile
│   └── railway.json                   # Railway configuration
├── data/                  # Data files (if any)
├── logs/                  # Log files
├── main.py                # Application entry point
├── requirements.txt       # Python dependencies
└── .gitignore            # Git ignore rules
```

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- pip
- Docker (optional, for containerized deployment)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd FraudPaymentDetectionML
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   python main.py
   ```

The API will be available at `http://localhost:8000`

## 📚 Documentation

- **API Documentation**: Available at `http://localhost:8000/docs` when running
- **Integration Guide**: See `docs/API_INTEGRATION_GUIDE.md`
- **Enterprise Guide**: See `docs/enterprise_integration_guide.md`
- **PII Compliance**: See `docs/enterprise_pii_compliance_guide.md`

## 🔧 Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test file
python -m pytest tests/test_api.py
```

### Training Models

```bash
# Basic training
python src/models/train_fraud_model.py

# Enhanced training with synthetic data
python src/models/enhanced_fraud_training.py

# Comprehensive training
python src/models/comprehensive_fraud_training.py
```

### Code Organization

- **API Layer** (`src/api/`): FastAPI application and client libraries
- **Model Layer** (`src/models/`): Machine learning models and training pipelines
- **Data Layer** (`src/data/`): Data processing, schemas, and feature extraction
- **Utils** (`src/utils/`): Utility functions and helper scripts
- **Config** (`src/config/`): Configuration files and model metadata

## 🐳 Docker Deployment

### Build and Run

```bash
# Build the image
docker build -f deployment/Dockerfile -t fraud-detection-api .

# Run the container
docker run -p 8000:8000 fraud-detection-api
```

### Using Docker Compose

```bash
cd deployment
docker-compose up -d
```

## 🔒 Security Features

- **PII Anonymization**: All personally identifiable information is anonymized
- **Privacy-Preserving**: Uses hashing and feature extraction instead of raw PII
- **Enterprise-Ready**: Includes compliance guides and security best practices

## 📊 Model Performance

The system supports multiple fraud detection models:
- **XGBoost**: High-performance gradient boosting
- **LightGBM**: Fast gradient boosting framework
- **Random Forest**: Ensemble learning method
- **Gradient Boosting**: Sequential ensemble method

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the documentation in the `docs/` folder
- Review the API documentation at `/docs` endpoint
- Open an issue on GitHub

## 🔄 Version History

- **v1.0.0**: Initial release with basic fraud detection
- **v1.1.0**: Enhanced with synthetic data training
- **v1.2.0**: Added enterprise features and PII compliance
- **v1.3.0**: Improved project structure and documentation 