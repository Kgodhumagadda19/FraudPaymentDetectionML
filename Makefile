.PHONY: help install test lint format clean run build deploy

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt
	pip install -e .

test: ## Run tests
	python -m pytest tests/ -v

test-coverage: ## Run tests with coverage
	python -m pytest tests/ --cov=src --cov-report=html --cov-report=term-missing

lint: ## Run linting
	flake8 src/ tests/
	mypy src/

format: ## Format code
	black src/ tests/
	isort src/ tests/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf build/
	rm -rf dist/

run: ## Run the application
	python main.py

run-dev: ## Run the application in development mode
	uvicorn src.api.fraud_detection_api:app --reload --host 0.0.0.0 --port 8000

build: ## Build Docker image
	docker build -f deployment/Dockerfile -t fraud-detection-api .

deploy: ## Deploy using Docker Compose
	cd deployment && docker-compose up -d

deploy-stop: ## Stop Docker Compose deployment
	cd deployment && docker-compose down

train: ## Train the model
	python src/models/enhanced_fraud_training.py

train-basic: ## Train basic model
	python src/models/train_fraud_model.py

docs: ## Generate documentation
	cd docs && mkdocs build

serve-docs: ## Serve documentation
	cd docs && mkdocs serve

check: ## Run all checks (lint, test, format)
	make lint
	make test
	make format

setup-dev: ## Setup development environment
	python -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .[dev] 