#!/usr/bin/env python3
"""
Quick Project Setup Script (No GitHub CLI required)
Author: N. Sachin Deshik
GitHub: sachin-deshik-10
"""

import os
import subprocess
import sys
from pathlib import Path


def main():
    print("🚀 O-RAN Security Test Case Generator")
    print("Quick Setup Script")
    print("Author: N. Sachin Deshik")
    print("=" * 60)
    
    project_root = Path.cwd()
    print(f"📁 Project Root: {project_root}")
    
    # Check if we're in the right directory
    if not (project_root / "app.py").exists():
        print("❌ Error: Please run this script from the project root directory")
        sys.exit(1)
    
    print("✅ Project structure validated")
    
    # Initialize git repository if not exists
    if not (project_root / ".git").exists():
        print("🔧 Initializing Git repository...")
        try:
            subprocess.run(["git", "init"], cwd=project_root, check=True)
            print("✅ Git repository initialized")
        except Exception as e:
            print(f"❌ Git initialization failed: {e}")
    else:
        print("✅ Git repository already exists")
    
    # Create .gitignore if not exists
    if not (project_root / ".gitignore").exists():
        print("📝 Creating .gitignore...")
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Logs
*.log
logs/

# Environment
.env
.env.local
.env.production

# Models
models/
*.pkl
*.joblib
*.h5

# Data
data/raw/
data/processed/
*.csv
*.json

# Output
output/
results/
reports/
temp/
tmp/

# Secrets
secrets/
*.pem
*.key
*.crt
"""
        with open(project_root / ".gitignore", "w") as f:
            f.write(gitignore_content)
        print("✅ .gitignore created")
    
    # Create necessary directories
    directories = [
        "logs", "models", "output/reports", "output/models", 
        "scripts", "docs/images", "docs/api"
    ]
    
    print("📁 Creating project directories...")
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"   ✅ {dir_name}/")
    
    # Create sample environment file
    env_file = project_root / ".env.example"
    if not env_file.exists():
        print("⚙️  Creating .env.example...")
        env_content = """# O-RAN Security Test Case Generator Configuration
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

# Application Configuration
ENVIRONMENT=development
DEBUG=True
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# ML Configuration
MODEL_PATH=./models
BATCH_SIZE=32
"""
        with open(env_file, "w") as f:
            f.write(env_content)
        print("✅ .env.example created")
    
    # Check Git configuration
    print("🔧 Checking Git configuration...")
    try:
        result = subprocess.run(["git", "config", "user.name"], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("⚠️  Git user.name not configured")
            print("   Run: git config --global user.name 'N. Sachin Deshik'")
        else:
            print(f"✅ Git user.name: {result.stdout.strip()}")
    except Exception as e:
        print(f"⚠️  Could not check Git config: {e}")
    
    # Add files to git
    print("📦 Adding files to Git...")
    try:
        subprocess.run(["git", "add", "."], cwd=project_root, check=True)
        print("✅ Files added to Git")
    except Exception as e:
        print(f"❌ Failed to add files to Git: {e}")
    
    # Create initial commit
    print("💾 Creating initial commit...")
    try:
        subprocess.run([
            "git", "commit", "-m", 
            "🚀 Initial commit: Advanced O-RAN Security Test Case Generator with AI/ML capabilities"
        ], cwd=project_root, check=True)
        print("✅ Initial commit created")
    except Exception as e:
        print(f"⚠️  Commit failed (possibly nothing to commit): {e}")
    
    print("\n🎉 Setup completed successfully!")
    print("=" * 60)
    print("🌐 GitHub Repository Setup:")
    print("1. Go to https://github.com/new")
    print("2. Repository name: O-RAN-Security-Test-Case-Generator")
    print("3. Description: AI/ML-powered O-RAN security test case generator")
    print("4. Set to Public")
    print("5. Don't initialize with README (we already have one)")
    print("6. Create repository")
    print("7. Run the following commands:")
    print(f"   git remote add origin https://github.com/sachin-deshik-10/O-RAN-Security-Test-Case-Generator.git")
    print(f"   git branch -M main")
    print(f"   git push -u origin main")
    print("\n📋 Next Steps:")
    print("1. Configure your API keys in .env file")
    print("2. Run: python app.py (to start Streamlit app)")
    print("3. Run: python -m uvicorn api.oran_api:app --reload (to start API)")
    print("4. Run: python -m pytest tests/ (to run tests)")
    print("\n🔗 Documentation:")
    print("• README.md - Main documentation")
    print("• docs/architecture.md - System architecture")
    print("• ENHANCEMENT_ROADMAP.md - Future plans")
    print("• CODE_REVIEW.md - Code review checklist")
    print("\n🚀 Happy coding!")


if __name__ == "__main__":
    main()
