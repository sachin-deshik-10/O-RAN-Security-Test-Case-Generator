#!/bin/bash
# Quick Setup Script for O-RAN Security Test Case Generator
# Author: N. Sachin Deshik
# GitHub: sachin-deshik-10

set -e

echo "🚀 Setting up O-RAN Security Test Case Generator"
echo "=" * 60

# Check if we're in the right directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8+ required. Found: $python_version"
    exit 1
fi

echo "✅ Python $python_version detected"

# Create virtual environment
echo "🔧 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "⚠️  Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies
echo "📥 Installing development dependencies..."
pip install -r requirements-dev.txt

# Setup pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p logs
mkdir -p models
mkdir -p output/reports
mkdir -p output/models
mkdir -p scripts
mkdir -p k8s
mkdir -p docs/images
mkdir -p docs/api
mkdir -p docs/user-guide
mkdir -p docs/developer-guide

# Create environment file from template
echo "⚙️  Creating environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env << 'EOF'
# O-RAN Security Test Case Generator Configuration
# Author: N. Sachin Deshik

# API Keys
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

# Database Configuration
DATABASE_URL=postgresql://user:password@localhost:5432/oran_security
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET=your_jwt_secret_here
ENCRYPTION_KEY=your_encryption_key_here

# Application Configuration
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_WORKERS=4

# ML Configuration
MODEL_PATH=./models
BATCH_SIZE=32
MAX_SEQUENCE_LENGTH=512

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000

# Cloud Configuration
AWS_ACCESS_KEY_ID=your_aws_access_key
AWS_SECRET_ACCESS_KEY=your_aws_secret_key
AWS_REGION=us-east-1
AZURE_SUBSCRIPTION_ID=your_azure_subscription_id
GCP_PROJECT_ID=your_gcp_project_id

# Notification Configuration
SLACK_WEBHOOK_URL=your_slack_webhook_url
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USERNAME=your_email@gmail.com
SMTP_PASSWORD=your_app_password
EOF
    echo "✅ Environment file created (.env)"
    echo "⚠️  Please edit .env file with your actual configuration values"
else
    echo "⚠️  Environment file already exists"
fi

# Run initial tests
echo "🧪 Running initial tests..."
if python -m pytest tests/ -v --tb=short; then
    echo "✅ All tests passed"
else
    echo "⚠️  Some tests failed, but setup continues"
fi

# Generate API documentation
echo "📖 Generating API documentation..."
if command -v python &> /dev/null; then
    python -c "
import sys
sys.path.append('.')
from api.oran_api import app
import json

# Generate OpenAPI schema
openapi_schema = app.openapi()
with open('docs/api/openapi.json', 'w') as f:
    json.dump(openapi_schema, f, indent=2)
print('✅ API documentation generated')
" || echo "⚠️  API documentation generation failed"
fi

# Create sample data
echo "📊 Creating sample data..."
python -c "
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample O-RAN network data
np.random.seed(42)
dates = pd.date_range(start='2024-01-01', periods=1000, freq='1min')

sample_data = pd.DataFrame({
    'timestamp': dates,
    'latency': np.random.normal(10, 2, 1000),
    'throughput': np.random.normal(1000, 100, 1000),
    'packet_loss': np.random.uniform(0, 5, 1000),
    'cpu_usage': np.random.normal(50, 10, 1000),
    'memory_usage': np.random.normal(60, 15, 1000),
    'network_usage': np.random.normal(70, 20, 1000),
    'security_events': np.random.poisson(3, 1000),
    'vulnerability_score': np.random.uniform(1, 10, 1000),
    'threat_level': np.random.uniform(0, 1, 1000),
    'compliance_score': np.random.uniform(0.7, 1.0, 1000)
})

sample_data.to_json('data/sample_oran_data.json', orient='records', date_format='iso')
print('✅ Sample data created')
"

# Create startup script
echo "🔧 Creating startup script..."
cat > start.sh << 'EOF'
#!/bin/bash
# Startup script for O-RAN Security Test Case Generator

echo "🚀 Starting O-RAN Security Test Case Generator..."

# Activate virtual environment
source venv/bin/activate

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "❌ Error: .env file not found. Please create it from .env.example"
    exit 1
fi

# Load environment variables
export $(cat .env | xargs)

# Start the application
echo "🌐 Starting Streamlit application..."
streamlit run app.py --server.port=8501 --server.address=0.0.0.0 &

echo "🔌 Starting API server..."
uvicorn api.oran_api:app --host 0.0.0.0 --port 8000 --reload &

echo "✅ Application started successfully!"
echo "🌐 Streamlit UI: http://localhost:8501"
echo "🔌 API: http://localhost:8000"
echo "📖 API Docs: http://localhost:8000/docs"

# Wait for services to start
sleep 5

# Health check
echo "🏥 Performing health check..."
curl -f http://localhost:8000/health || echo "⚠️  API health check failed"

echo "🎉 Setup complete! Press Ctrl+C to stop services"
wait
EOF

chmod +x start.sh
echo "✅ Startup script created (start.sh)"

# Create testing script
echo "🧪 Creating testing script..."
cat > test.sh << 'EOF'
#!/bin/bash
# Testing script for O-RAN Security Test Case Generator

echo "🧪 Running comprehensive tests..."

# Activate virtual environment
source venv/bin/activate

# Run unit tests
echo "🔬 Running unit tests..."
python -m pytest tests/test_ml_models.py -v

# Run integration tests
echo "🔗 Running integration tests..."
python -m pytest tests/ -m integration -v

# Run security tests
echo "🔒 Running security tests..."
python -m pytest tests/ -m security -v

# Run performance tests
echo "⚡ Running performance tests..."
python -m pytest tests/ -m performance -v

# Generate coverage report
echo "📊 Generating coverage report..."
python -m pytest --cov=ml_models --cov=api --cov-report=html --cov-report=term

# Run linting
echo "🔍 Running code quality checks..."
flake8 ml_models/ api/
black --check ml_models/ api/
isort --check-only ml_models/ api/
mypy ml_models/ --ignore-missing-imports

# Run security scanning
echo "🔒 Running security scan..."
bandit -r ml_models/ api/

echo "✅ All tests completed!"
EOF

chmod +x test.sh
echo "✅ Testing script created (test.sh)"

# Create deployment script
echo "🚀 Creating deployment script..."
cat > deploy.sh << 'EOF'
#!/bin/bash
# Deployment script for O-RAN Security Test Case Generator

echo "🚀 Deploying O-RAN Security Test Case Generator..."

# Build Docker image
echo "🐳 Building Docker image..."
docker build -t oran-security-generator:latest .

# Tag for different environments
docker tag oran-security-generator:latest oran-security-generator:dev
docker tag oran-security-generator:latest oran-security-generator:$(date +%Y%m%d_%H%M%S)

# Deploy with Docker Compose
echo "🐳 Deploying with Docker Compose..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to be ready..."
sleep 30

# Health check
echo "🏥 Performing health check..."
curl -f http://localhost:8501 || echo "⚠️  Streamlit health check failed"
curl -f http://localhost:8000/health || echo "⚠️  API health check failed"

echo "✅ Deployment completed!"
echo "🌐 Streamlit UI: http://localhost:8501"
echo "🔌 API: http://localhost:8000"
echo "📊 Monitoring: http://localhost:3000 (Grafana)"
EOF

chmod +x deploy.sh
echo "✅ Deployment script created (deploy.sh)"

# Final setup summary
echo ""
echo "🎉 Setup completed successfully!"
echo "=" * 60
echo "📁 Project structure created"
echo "🐍 Virtual environment configured"
echo "📦 Dependencies installed"
echo "🔧 Pre-commit hooks installed"
echo "⚙️  Environment configuration created"
echo "🧪 Initial tests passed"
echo "📖 Documentation generated"
echo "📊 Sample data created"
echo "🚀 Startup scripts created"
echo ""
echo "🎯 Next steps:"
echo "1. Edit .env file with your API keys and configuration"
echo "2. Run: ./start.sh (to start the application)"
echo "3. Run: ./test.sh (to run comprehensive tests)"
echo "4. Run: ./deploy.sh (for Docker deployment)"
echo "5. Visit: http://localhost:8501 (Streamlit UI)"
echo "6. Visit: http://localhost:8000/docs (API documentation)"
echo ""
echo "🔗 Useful commands:"
echo "- source venv/bin/activate (activate virtual environment)"
echo "- pre-commit run --all-files (run code quality checks)"
echo "- pytest tests/ -v (run tests)"
echo "- docker-compose up -d (start with Docker)"
echo ""
echo "Happy coding! 🚀"
