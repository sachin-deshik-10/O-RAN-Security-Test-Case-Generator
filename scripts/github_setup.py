#!/usr/bin/env python3
"""
GitHub Repository Setup Script
Author: N. Sachin Deshik
GitHub: sachin-deshik-10
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional
import json
import requests
from dataclasses import dataclass


@dataclass
class GitHubConfig:
    """GitHub repository configuration"""
    owner: str = "sachin-deshik-10"
    repo: str = "O-RAN-Security-Test-Case-Generator"
    description: str = "AI/ML-powered O-RAN security test case generator with advanced threat detection and analysis capabilities"
    topics: List[str] = None
    private: bool = False
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = [
                "machine-learning",
                "artificial-intelligence", 
                "cybersecurity",
                "o-ran",
                "security-testing",
                "threat-detection",
                "deep-learning",
                "fastapi",
                "streamlit",
                "python",
                "docker",
                "kubernetes",
                "ci-cd",
                "monitoring",
                "telecommunications"
            ]


class GitHubRepositorySetup:
    """GitHub repository setup and configuration"""
    
    def __init__(self, config: GitHubConfig):
        self.config = config
        self.repo_path = Path.cwd()
        self.github_token = os.getenv("GITHUB_TOKEN")
        
    def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met"""
        print("🔍 Checking prerequisites...")
        
        # Check if git is installed
        try:
            subprocess.run(["git", "--version"], capture_output=True, check=True)
            print("✅ Git is installed")
        except subprocess.CalledProcessError:
            print("❌ Git is not installed. Please install Git first.")
            return False
        
        # Check if GitHub CLI is installed
        try:
            subprocess.run(["gh", "--version"], capture_output=True, check=True)
            print("✅ GitHub CLI is installed")
        except subprocess.CalledProcessError:
            print("⚠️  GitHub CLI is not installed. Some features may not work.")
        
        # Check if we're in a git repository
        if not (self.repo_path / ".git").exists():
            print("⚠️  Not in a Git repository. Will initialize one.")
            
        return True
        
    def initialize_git_repository(self) -> bool:
        """Initialize Git repository if not exists"""
        print("🔧 Initializing Git repository...")
        
        try:
            if not (self.repo_path / ".git").exists():
                subprocess.run(["git", "init"], cwd=self.repo_path, check=True)
                print("✅ Git repository initialized")
            else:
                print("✅ Git repository already exists")
                
            # Configure git user if not set
            try:
                subprocess.run(["git", "config", "user.name"], 
                             capture_output=True, check=True)
            except subprocess.CalledProcessError:
                subprocess.run(["git", "config", "user.name", "N. Sachin Deshik"], 
                             cwd=self.repo_path)
                print("✅ Git user name configured")
                
            try:
                subprocess.run(["git", "config", "user.email"], 
                             capture_output=True, check=True)
            except subprocess.CalledProcessError:
                email = input("Enter your email address: ")
                subprocess.run(["git", "config", "user.email", email], 
                             cwd=self.repo_path)
                print("✅ Git user email configured")
                
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error initializing Git repository: {e}")
            return False
    
    def create_gitignore(self) -> bool:
        """Create comprehensive .gitignore file"""
        print("📝 Creating .gitignore file...")
        
        gitignore_content = """# O-RAN Security Test Case Generator .gitignore
# Author: N. Sachin Deshik

# Python
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
share/python-wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.py,cover
.hypothesis/
.pytest_cache/
cover/

# Translations
*.mo
*.pot

# Django
*.log
local_settings.py
db.sqlite3
db.sqlite3-journal

# Flask
instance/
.webassets-cache

# Scrapy
.scrapy

# Sphinx documentation
docs/_build/

# PyBuilder
.pybuilder/
target/

# Jupyter Notebook
.ipynb_checkpoints

# IPython
profile_default/
ipython_config.py

# pyenv
.python-version

# pipenv
Pipfile.lock

# poetry
poetry.lock

# pdm
.pdm.toml

# PEP 582
__pypackages__/

# Celery
celerybeat-schedule
celerybeat.pid

# SageMath parsed files
*.sage.py

# Environments
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Spyder project settings
.spyderproject
.spyproject

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# Pyre type checker
.pyre/

# pytype static type analyzer
.pytype/

# Cython debug symbols
cython_debug/

# PyCharm
.idea/

# VS Code
.vscode/
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json
!.vscode/*.code-snippets

# Local History for Visual Studio Code
.history/

# Built Visual Studio Code Extensions
*.vsix

# macOS
.DS_Store
.AppleDouble
.LSOverride

# Windows
Thumbs.db
Thumbs.db:encryptable
ehthumbs.db
ehthumbs_vista.db
*.tmp
*.temp
*.log
*.lock
*.pid
*.seed
*.pid.lock
Desktop.ini
$RECYCLE.BIN/
*.cab
*.msi
*.msix
*.msm
*.msp
*.lnk

# Linux
*~

# Node.js
node_modules/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*
.pnpm-debug.log*

# Docker
.dockerignore
Dockerfile.*
docker-compose.override.yml
docker-compose.*.yml
!docker-compose.yml

# Kubernetes
*.kubeconfig
kube-config-*

# Logs
logs/
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*
lerna-debug.log*

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/
*.lcov

# nyc test coverage
.nyc_output

# Grunt intermediate storage
.grunt

# Bower dependency directory
bower_components

# node-waf configuration
.lock-wscript

# Compiled binary addons
build/Release

# Dependency directories
node_modules/
jspm_packages/

# Snowpack dependency directory
web_modules/

# TypeScript cache
*.tsbuildinfo

# Optional npm cache directory
.npm

# Optional eslint cache
.eslintcache

# Optional stylelint cache
.stylelintcache

# Microbundle cache
.rpt2_cache/
.rts2_cache_cjs/
.rts2_cache_es/
.rts2_cache_umd/

# Optional REPL history
.node_repl_history

# Output of 'npm pack'
*.tgz

# Yarn Integrity file
.yarn-integrity

# dotenv environment variable files
.env
.env.development.local
.env.test.local
.env.production.local
.env.local

# parcel-bundler cache
.cache
.parcel-cache

# Next.js build output
.next
out

# Nuxt.js build / generate output
.nuxt
dist

# Gatsby files
.cache/
public

# Vuepress build output
.vuepress/dist

# Docusaurus cache and build directory
.docusaurus

# Serverless directories
.serverless/

# FuseBox cache
.fusebox/

# DynamoDB Local files
.dynamodb/

# TernJS port file
.tern-port

# Stores VSCode versions used for testing VSCode extensions
.vscode-test

# yarn v2
.yarn/cache
.yarn/unplugged
.yarn/build-state.yml
.yarn/install-state.gz
.pnp.*

# ML Models
models/
*.pkl
*.joblib
*.h5
*.pb
*.onnx
*.tflite

# Data
data/raw/
data/processed/
data/external/
*.csv
*.json
*.parquet
*.h5
*.hdf5

# Outputs
output/
results/
reports/
logs/
temp/
tmp/

# Secrets
secrets/
.secrets
*.pem
*.key
*.crt
*.p12
*.pfx

# Monitoring
prometheus/
grafana/
.monitoring

# Cloud
.aws/
.azure/
.gcp/
terraform.tfstate*
terraform.tfvars
*.terraform

# IDE
.idea/
.vscode/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Custom
config/local.yaml
config/secrets.yaml
.env.local
.env.production
.env.staging
"""

        try:
            with open(self.repo_path / ".gitignore", "w") as f:
                f.write(gitignore_content)
            print("✅ .gitignore file created")
            return True
        except Exception as e:
            print(f"❌ Error creating .gitignore: {e}")
            return False
    
    def add_and_commit_files(self) -> bool:
        """Add and commit all files"""
        print("📦 Adding and committing files...")
        
        try:
            # Add all files
            subprocess.run(["git", "add", "."], cwd=self.repo_path, check=True)
            
            # Commit files
            commit_message = "🚀 Initial commit: Advanced O-RAN Security Test Case Generator with AI/ML capabilities"
            subprocess.run(["git", "commit", "-m", commit_message], 
                         cwd=self.repo_path, check=True)
            
            print("✅ Files committed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error committing files: {e}")
            return False
    
    def setup_remote_repository(self) -> bool:
        """Setup remote GitHub repository"""
        print("🌐 Setting up remote GitHub repository...")
        
        try:
            # Check if GitHub CLI is available
            subprocess.run(["gh", "--version"], capture_output=True, check=True)
            
            # Create repository using GitHub CLI
            cmd = [
                "gh", "repo", "create",
                f"{self.config.owner}/{self.config.repo}",
                "--description", self.config.description,
                "--public" if not self.config.private else "--private",
                "--source", ".",
                "--remote", "origin",
                "--push"
            ]
            
            subprocess.run(cmd, cwd=self.repo_path, check=True)
            print("✅ GitHub repository created and pushed")
            
            # Add topics/labels
            if self.config.topics:
                topics_str = " ".join(self.config.topics)
                subprocess.run([
                    "gh", "repo", "edit", 
                    f"{self.config.owner}/{self.config.repo}",
                    "--add-topic", topics_str
                ], cwd=self.repo_path, check=True)
                print("✅ Repository topics added")
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error setting up remote repository: {e}")
            print("💡 Please create the repository manually on GitHub")
            return False
    
    def setup_branch_protection(self) -> bool:
        """Setup branch protection rules"""
        print("🔒 Setting up branch protection...")
        
        try:
            # Enable branch protection for main branch
            subprocess.run([
                "gh", "repo", "edit",
                f"{self.config.owner}/{self.config.repo}",
                "--enable-wiki", "false",
                "--enable-issues", "true",
                "--enable-projects", "true"
            ], cwd=self.repo_path, check=True)
            
            print("✅ Branch protection configured")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Could not setup branch protection: {e}")
            return False
    
    def create_issue_templates(self) -> bool:
        """Create issue templates"""
        print("📋 Creating issue templates...")
        
        try:
            # Create .github directory
            github_dir = self.repo_path / ".github"
            github_dir.mkdir(exist_ok=True)
            
            issue_template_dir = github_dir / "ISSUE_TEMPLATE"
            issue_template_dir.mkdir(exist_ok=True)
            
            # Bug report template
            bug_template = """---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: sachin-deshik-10

---

## Bug Description
A clear and concise description of what the bug is.

## Steps to Reproduce
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

## Expected Behavior
A clear and concise description of what you expected to happen.

## Actual Behavior
A clear and concise description of what actually happened.

## Screenshots
If applicable, add screenshots to help explain your problem.

## Environment
- OS: [e.g. Ubuntu 20.04, Windows 10, macOS 12]
- Python Version: [e.g. 3.9.7]
- Browser: [e.g. Chrome 95, Firefox 94]
- Application Version: [e.g. 1.0.0]

## Additional Context
Add any other context about the problem here.

## Logs
Please include relevant logs or error messages:
```
Paste logs here
```

## Possible Solution
If you have ideas on how to fix this, please describe them here.
"""

            with open(issue_template_dir / "bug_report.md", "w") as f:
                f.write(bug_template)
            
            # Feature request template
            feature_template = """---
name: Feature Request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: sachin-deshik-10

---

## Feature Description
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

## Proposed Solution
A clear and concise description of what you want to happen.

## Alternative Solutions
A clear and concise description of any alternative solutions or features you've considered.

## Use Case
Describe the use case for this feature. How would it be used?

## Benefits
What benefits would this feature provide?

## Implementation Ideas
If you have ideas on how to implement this feature, please describe them here.

## Additional Context
Add any other context or screenshots about the feature request here.

## Priority
- [ ] Low
- [ ] Medium
- [ ] High
- [ ] Critical

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3
"""

            with open(issue_template_dir / "feature_request.md", "w") as f:
                f.write(feature_template)
            
            print("✅ Issue templates created")
            return True
            
        except Exception as e:
            print(f"❌ Error creating issue templates: {e}")
            return False
    
    def create_pull_request_template(self) -> bool:
        """Create pull request template"""
        print("🔄 Creating pull request template...")
        
        try:
            github_dir = self.repo_path / ".github"
            github_dir.mkdir(exist_ok=True)
            
            pr_template = """# Pull Request

## Description
Please provide a brief description of the changes in this pull request.

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring
- [ ] Security fix

## Related Issues
Closes # (issue number)

## Changes Made
- [ ] Change 1
- [ ] Change 2
- [ ] Change 3

## Testing
- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] Manual testing completed
- [ ] Performance tests pass (if applicable)
- [ ] Security tests pass (if applicable)

## Code Quality
- [ ] Code follows the project's style guidelines
- [ ] Self-review of code completed
- [ ] Code is properly commented
- [ ] Documentation updated (if applicable)
- [ ] No new linting errors introduced

## Security
- [ ] No sensitive data exposed
- [ ] Input validation implemented
- [ ] Security best practices followed
- [ ] Dependencies updated to latest secure versions

## Performance
- [ ] No performance degradation
- [ ] Memory usage optimized
- [ ] Database queries optimized (if applicable)
- [ ] Caching implemented where appropriate

## Screenshots (if applicable)
Please add screenshots to help explain your changes.

## Checklist
- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
- [ ] Any dependent changes have been merged and published in downstream modules

## Additional Notes
Add any additional notes or context about the pull request here.

## Reviewer Guidelines
Please review the following:
1. Code quality and adherence to project standards
2. Test coverage and quality
3. Documentation updates
4. Security implications
5. Performance impact
6. Breaking changes
"""

            with open(github_dir / "pull_request_template.md", "w") as f:
                f.write(pr_template)
            
            print("✅ Pull request template created")
            return True
            
        except Exception as e:
            print(f"❌ Error creating pull request template: {e}")
            return False
    
    def create_security_policy(self) -> bool:
        """Create security policy"""
        print("🔒 Creating security policy...")
        
        try:
            security_content = """# Security Policy

## Supported Versions

We take security seriously and provide security updates for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in the O-RAN Security Test Case Generator, please report it to us privately. We appreciate your responsible disclosure.

### How to Report

1. **Email**: Send details to the project maintainer
2. **GitHub Security Advisory**: Use GitHub's private vulnerability reporting
3. **Direct Message**: Contact @sachin-deshik-10 on GitHub

### What to Include

Please include the following information in your report:

- Description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Suggested fix (if available)
- Your contact information

### Response Timeline

- **Acknowledgment**: We will acknowledge receipt of your report within 48 hours
- **Initial Assessment**: We will provide an initial assessment within 5 business days
- **Resolution**: We aim to resolve critical vulnerabilities within 30 days
- **Disclosure**: We will coordinate with you on responsible disclosure

### Security Measures

Our project implements the following security measures:

- **Input Validation**: All user inputs are validated and sanitized
- **Authentication**: JWT-based authentication with proper token validation
- **Authorization**: Role-based access control (RBAC)
- **Data Protection**: Sensitive data is encrypted at rest and in transit
- **Dependency Management**: Regular updates and security scanning
- **Code Analysis**: Static Application Security Testing (SAST)
- **Container Security**: Secure container images with minimal attack surface

### Security Best Practices

When using this project:

1. **Environment Variables**: Never commit sensitive information to version control
2. **API Keys**: Use environment variables for API keys and secrets
3. **Network Security**: Deploy behind proper network security controls
4. **Regular Updates**: Keep dependencies and the application updated
5. **Monitoring**: Implement proper logging and monitoring
6. **Access Control**: Follow principle of least privilege

### Scope

This security policy applies to:

- The main application code
- API endpoints
- Web interface
- Container images
- Documentation
- CI/CD pipeline

### Out of Scope

The following are typically out of scope:

- Denial of service attacks
- Social engineering attacks
- Physical security
- Third-party dependencies (report to respective maintainers)

### Recognition

We appreciate security researchers who help make our project safer. With your permission, we will:

- Acknowledge your contribution in our security acknowledgments
- Provide credit in our release notes
- Consider monetary rewards for significant vulnerabilities (subject to evaluation)

### Security Updates

Security updates will be:

- Released as patch versions
- Announced in our security advisories
- Documented in our changelog
- Communicated to users through appropriate channels

### Contact Information

**Project Maintainer**: N. Sachin Deshik  
**GitHub**: [@sachin-deshik-10](https://github.com/sachin-deshik-10)  
**Project**: O-RAN Security Test Case Generator

---

Thank you for helping keep our project and our users secure!
"""

            with open(self.repo_path / "SECURITY.md", "w") as f:
                f.write(security_content)
            
            print("✅ Security policy created")
            return True
            
        except Exception as e:
            print(f"❌ Error creating security policy: {e}")
            return False
    
    def run_setup(self) -> bool:
        """Run complete GitHub repository setup"""
        print("🚀 Starting GitHub Repository Setup")
        print("=" * 60)
        
        success = True
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            return False
        
        # Step 2: Initialize Git repository
        if not self.initialize_git_repository():
            success = False
        
        # Step 3: Create .gitignore
        if not self.create_gitignore():
            success = False
        
        # Step 4: Create GitHub templates
        if not self.create_issue_templates():
            success = False
        
        if not self.create_pull_request_template():
            success = False
        
        if not self.create_security_policy():
            success = False
        
        # Step 5: Add and commit files
        if not self.add_and_commit_files():
            success = False
        
        # Step 6: Setup remote repository
        if not self.setup_remote_repository():
            success = False
        
        # Step 7: Setup branch protection (optional)
        self.setup_branch_protection()
        
        if success:
            print("\n🎉 GitHub Repository Setup Completed Successfully!")
            print("=" * 60)
            print(f"🌐 Repository URL: https://github.com/{self.config.owner}/{self.config.repo}")
            print(f"📖 Documentation: https://github.com/{self.config.owner}/{self.config.repo}#readme")
            print(f"🔌 API Docs: https://github.com/{self.config.owner}/{self.config.repo}/blob/main/docs/api/")
            print(f"🚀 GitHub Actions: https://github.com/{self.config.owner}/{self.config.repo}/actions")
            print(f"📊 Insights: https://github.com/{self.config.owner}/{self.config.repo}/pulse")
            print("\n📋 Next Steps:")
            print("1. Visit your repository on GitHub")
            print("2. Enable GitHub Pages (if needed)")
            print("3. Configure repository secrets for CI/CD")
            print("4. Set up branch protection rules")
            print("5. Add collaborators if needed")
            print("6. Create your first issue or pull request")
        else:
            print("\n⚠️  Setup completed with some issues")
            print("Please check the messages above and fix any problems")
        
        return success


def main():
    """Main function"""
    print("🚀 O-RAN Security Test Case Generator")
    print("GitHub Repository Setup Script")
    print("Author: N. Sachin Deshik")
    print("=" * 60)
    
    # Configuration
    config = GitHubConfig()
    
    # Setup
    setup = GitHubRepositorySetup(config)
    success = setup.run_setup()
    
    if success:
        print("\n✅ Repository setup completed successfully!")
        print("🎯 You can now start using your GitHub repository.")
    else:
        print("\n❌ Repository setup completed with errors.")
        print("💡 Please check the error messages and fix any issues.")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
