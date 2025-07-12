# Contributing to O-RAN Security Test Case Generator

Thank you for your interest in contributing to the O-RAN Security Test Case Generator! This document provides guidelines for contributing to this project.

## 🎯 Code of Conduct

This project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of machine learning and O-RAN networks
- Familiarity with TensorFlow/PyTorch (for ML contributions)

### Development Setup

1. **Fork the Repository**

   ```bash
   git clone https://github.com/sachin-deshik-10/O-RAN-Security-Test-Case-Generator.git
   cd O-RAN-Security-Test-Case-Generator
   ```

2. **Create Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Install Pre-commit Hooks**

   ```bash
   pre-commit install
   ```

## 📋 How to Contribute

### 🐛 Reporting Bugs

1. **Check existing issues** to avoid duplicates
2. **Use the bug report template** when creating new issues
3. **Include detailed information**:
   - OS and Python version
   - Error messages and stack traces
   - Steps to reproduce
   - Expected vs actual behavior

### 💡 Suggesting Enhancements

1. **Check existing feature requests** to avoid duplicates
2. **Use the feature request template**
3. **Provide detailed description**:
   - Use case and motivation
   - Proposed implementation approach
   - Potential impact on existing functionality

### 🔧 Code Contributions

#### Branch Naming Convention

- `feature/description` - New features
- `bugfix/description` - Bug fixes
- `hotfix/description` - Critical fixes
- `refactor/description` - Code refactoring
- `docs/description` - Documentation updates

#### Pull Request Process

1. **Create Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**
   - Follow coding standards
   - Add tests for new functionality
   - Update documentation

3. **Run Tests**

   ```bash
   pytest
   flake8
   mypy ml_models/
   ```

4. **Commit Changes**

   ```bash
   git add .
   git commit -m "feat: add new anomaly detection model"
   ```

5. **Push Changes**

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Create Pull Request**
   - Use the PR template
   - Reference related issues
   - Provide clear description

## 🎨 Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://pycqa.github.io/isort/) for import sorting
- Maximum line length: 88 characters

### Code Quality Tools

```bash
# Format code
black .
isort .

# Check linting
flake8 .

# Type checking
mypy ml_models/

# Security scanning
bandit -r ml_models/
```

### Documentation Standards

- Use [Google Style Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include type hints for all functions
- Add examples in docstrings for public APIs

Example:

```python
def analyze_security_threats(
    network_data: pd.DataFrame,
    model_type: str = "ensemble"
) -> Dict[str, Any]:
    """
    Analyze security threats in O-RAN network data.
    
    Args:
        network_data: DataFrame containing network metrics and logs
        model_type: Type of ML model to use ('ensemble', 'transformer', 'rnn')
        
    Returns:
        Dictionary containing threat analysis results with keys:
        - 'threats': List of detected threats
        - 'severity': Overall threat severity score
        - 'recommendations': List of security recommendations
        
    Raises:
        ValueError: If network_data is empty or invalid
        
    Example:
        >>> data = pd.read_json('network_data.json')
        >>> results = analyze_security_threats(data, model_type='ensemble')
        >>> print(f"Threats detected: {len(results['threats'])}")
    """
```

## 🧪 Testing Guidelines

### Test Types

1. **Unit Tests** - Test individual functions/methods
2. **Integration Tests** - Test component interactions
3. **End-to-End Tests** - Test complete workflows
4. **Performance Tests** - Test ML model performance

### Writing Tests

```python
import pytest
import pandas as pd
from ml_models.oran_ml_models import ORANAnomalyDetector

class TestORANAnomalyDetector:
    def setup_method(self):
        """Set up test fixtures."""
        self.detector = ORANAnomalyDetector()
        self.sample_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=100, freq='1min'),
            'latency': np.random.normal(10, 2, 100),
            'throughput': np.random.normal(1000, 100, 100),
            'cpu_usage': np.random.normal(50, 10, 100)
        })
    
    def test_prepare_data(self):
        """Test data preparation functionality."""
        X, timestamps = self.detector.prepare_data(self.sample_data)
        assert X.shape[0] == len(self.sample_data)
        assert timestamps.shape[0] == len(self.sample_data)
    
    def test_train_isolation_forest(self):
        """Test Isolation Forest training."""
        X, _ = self.detector.prepare_data(self.sample_data)
        self.detector.train_isolation_forest(X)
        assert 'isolation_forest' in self.detector.models
        assert 'isolation_forest' in self.detector.scalers
```

### Test Coverage

- Maintain minimum 80% code coverage
- Test both success and failure scenarios
- Include edge cases and boundary conditions

## 📊 ML Model Contributions

### Adding New Models

1. **Create Model Class** in `ml_models/`
2. **Implement Required Methods**:
   - `train()` - Model training
   - `predict()` - Make predictions
   - `evaluate()` - Model evaluation
   - `save_model()` - Model persistence
   - `load_model()` - Model loading

3. **Add Model Integration** in `ORANMLPipeline`
4. **Create Comprehensive Tests**
5. **Update Documentation**

### Model Performance Requirements

- **Accuracy**: Minimum 85% on validation set
- **Inference Time**: < 100ms per prediction
- **Memory Usage**: < 2GB during training
- **Scalability**: Handle datasets up to 1M samples

### Model Documentation

Include in model docstring:

- Model architecture description
- Training data requirements
- Performance metrics
- Usage examples
- Limitations and assumptions

## 📚 Documentation Contributions

### Types of Documentation

1. **API Documentation** - Function/class references
2. **User Guides** - How-to guides and tutorials
3. **Developer Guides** - Architecture and implementation details
4. **Research Papers** - Technical background and methodology

### Documentation Standards

- Use Markdown for general documentation
- Use Sphinx for API documentation
- Include code examples and diagrams
- Keep documentation up-to-date with code changes

## 🔄 CI/CD Pipeline

### Automated Checks

- **Code Quality**: flake8, black, isort
- **Type Checking**: mypy
- **Security**: bandit, safety
- **Testing**: pytest with coverage
- **Documentation**: sphinx build

### Release Process

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create release PR
4. Tag release after merge
5. Automated deployment to PyPI

## 🏆 Recognition

### Contributors

All contributors are recognized in:

- Contributors section of README
- Release notes
- Annual contributor report

### Types of Contributions

- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation
- 🧪 Testing
- 🎨 Design
- 🔧 Infrastructure
- 🌐 Translation

## 📞 Getting Help

### Communication Channels

- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions and ideas
- **Discord** - Real-time chat and collaboration
- **Email** - Direct contact for sensitive issues

### Mentorship Program

New contributors can request mentorship:

- Guidance on first contributions
- Code review and feedback
- Learning resources and best practices
- Career advice in AI/ML and telecommunications

## 📋 Contribution Checklist

Before submitting your contribution:

- [ ] Code follows project style guidelines
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] Changes are backward compatible
- [ ] Performance impact assessed
- [ ] Security implications considered
- [ ] PR description is clear and complete
- [ ] Related issues are referenced
- [ ] Changelog updated (if applicable)

## 🙏 Thank You

Your contributions make this project better for everyone in the O-RAN and security communities. Every contribution, no matter how small, is valued and appreciated!

---

*This contributing guide is inspired by best practices from the open-source community and tailored for the O-RAN Security Test Case Generator project.*
