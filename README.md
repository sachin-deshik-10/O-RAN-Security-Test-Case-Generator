# Open RAN Security Test Case Generator

A comprehensive O-RAN security analysis tool that generates security test cases using AI and provides quantitative dataset analysis from O-RAN free sources.

## ✅ COMPLETED FEATURES

### 🔧 Core Migration

- **✅ Complete Gemini 2.5 Flash Migration**: Replaced all OpenAI API usage with Google Gemini
- **✅ Environment Setup**: Updated `.env` and `.env.template` for Gemini-only operation
- **✅ Dependencies**: Updated `Pipfile` with all required data analysis libraries
- **✅ Testing**: Comprehensive test suite for all components

### 🔍 O-RAN Dataset Analysis & Quantitative Assessment

- **✅ Multi-Dataset Support**: Analyze 10+ O-RAN datasets simultaneously
  - Performance metrics (Near-RT RIC, E2 traces, xApp performance)
  - Security metrics (vulnerability assessments, threat analysis)
  - Configuration data (O-RAN components, RIC configuration)
  - Standards compliance (ASVS, CAPEC, CWE)

- **✅ Security Analysis**:
  - Vulnerability assessment with CVSS scoring
  - Threat pattern analysis and categorization
  - Security score calculation (0-10 scale)
  - Component-level security analysis

- **✅ Performance Analysis**:
  - Latency, throughput, and availability metrics
  - Resource utilization tracking (CPU, memory, network)
  - SLA compliance monitoring
  - Performance trend analysis

- **✅ Interactive Visualizations**:
  - Security dashboards with gauge charts
  - Performance trend graphs
  - Vulnerability distribution charts
  - Component analysis visualizations

### 📊 O-RAN Free Source Data Collection

- **✅ O-RAN Alliance Specifications**: Automated collection of official O-RAN specs
- **✅ 3GPP Standards**: Integration with 3GPP technical specifications
- **✅ Cybersecurity Frameworks**: NIST, MITRE ATT&CK, CWE, CAPEC integration
- **✅ Export Capabilities**: JSON and CSV format support
- **✅ Metadata Tracking**: Complete audit trail of collected data

### 🎯 Enhanced User Interface

- **✅ Multi-Page Navigation**: Seamless switching between:
  - Security Test Case Generator
  - O-RAN Dataset Analysis
  - Data Collection Interface
- **✅ Interactive Controls**: Real-time analysis configuration
- **✅ Progress Monitoring**: Live updates during data collection and analysis
- **✅ Results Export**: One-click export of analysis results

### 🧪 Comprehensive Testing

- **✅ Integration Testing**: End-to-end system verification
- **✅ Component Testing**: Individual module validation
- **✅ API Testing**: Gemini integration verification
- **✅ Data Validation**: Dataset integrity checking

## 🚀 GETTING STARTED

### Prerequisites

- Python 3.8+
- Gemini API key from Google AI Studio

### Quick Setup

```bash
# Clone repository
git clone <repository-url>
cd O-RAN-Security-Test-Case-Generator-main

# Install dependencies
pip install -r requirements.txt
# OR use pipenv
pipenv install && pipenv shell

# Set up environment
cp .env.template .env
# Edit .env and add your GEMINI_API_KEY

# Run comprehensive tests
python test_integration.py

# Launch application
streamlit run app.py
```

## 📈 USAGE EXAMPLES

### Security Test Case Generation

1. Navigate to the main page
2. Input your O-RAN use case scenario
3. Select relevant attack patterns (CAPEC, CWE)
4. Choose security requirements (ASVS)
5. Generate AI-powered test cases

### Dataset Analysis

1. Navigate to "O-RAN Dataset Analysis"
2. Select datasets: Performance, Security, E2 Traces, etc.
3. Configure analysis options:
   - Basic Statistics ✓
   - Security Metrics ✓
   - Performance Metrics ✓
   - Interactive Visualizations ✓
4. View comprehensive analysis results

### Data Collection

1. Navigate to "Data Collection"
2. Select sources: O-RAN Alliance, 3GPP, NIST, etc.
3. Configure collection parameters
4. Start automated data collection
5. Export results in JSON/CSV format

## 📊 QUANTITATIVE RESULTS

### Sample Analysis Output

```json
{
  "security_analysis": {
    "total_vulnerabilities": 50,
    "critical_issues": 12,
    "security_score": 4.4,
    "cvss_distribution": {
      "critical": 12,
      "high": 18,
      "medium": 15,
      "low": 5
    }
  },
  "performance_analysis": {
    "avg_latency": 4.8,
    "throughput": 1050.5,
    "availability": 99.7,
    "resource_utilization": {
      "cpu": 68.5,
      "memory": 72.1,
      "network": 45.3
    }
  }
}
```

### Collected Data Sources

- **O-RAN Alliance**: 4 specifications (E2AP, A1AP, O1, xApp SDK)
- **3GPP Standards**: 15+ relevant technical specifications
- **NIST Framework**: 108 cybersecurity controls
- **MITRE ATT&CK**: 185 ICS-specific attack techniques
- **CWE Database**: 900+ weakness patterns
- **CAPEC**: 544 attack patterns

## 🔧 TESTING

### Run All Tests

```bash
# Comprehensive integration test
python test_integration.py

# Individual component tests
python test_gemini.py           # Gemini API integration
python test_data_analysis.py    # Data analysis functionality
python test_data_collection.py  # Data collection functionality
python test_comprehensive.py    # Additional comprehensive tests
```

### Test Results

- **✅ 8/8 Integration Tests Passed** (100% success rate)
- **✅ All imports successful**
- **✅ Data files validated**
- **✅ Gemini API integration working**
- **✅ NLTK data available**
- **✅ Analysis functionality verified**
- **✅ Data collection operational**

## 🎯 WHAT'S NEXT: RECOMMENDED ENHANCEMENTS

### 1. **Advanced Machine Learning Integration**

- **Anomaly Detection**: ML-based detection of unusual O-RAN behavior patterns
- **Predictive Analytics**: Forecast security threats and performance issues
- **Automated Classification**: AI-powered categorization of vulnerabilities and threats
- **Real-time Monitoring**: Continuous analysis of live O-RAN data streams

### 2. **Enhanced Data Sources**

- **O-RAN SC Integration**: Connect to O-RAN Software Community live repositories
- **Real-time Telemetry**: Integration with actual O-RAN deployments
- **Vendor-specific Data**: Support for major O-RAN vendor datasets
- **5G Core Integration**: Extend analysis to 5G Core network components

### 3. **Advanced Analytics & Reporting**

- **Compliance Dashboards**: Automated compliance reporting for O-RAN standards
- **Risk Assessment**: Comprehensive risk scoring and mitigation recommendations
- **Executive Reports**: High-level security and performance summaries
- **Trend Analysis**: Historical analysis and forecasting capabilities

### 4. **Integration & Automation**

- **API Endpoints**: RESTful API for programmatic access to analysis capabilities
- **CI/CD Integration**: Automated security testing in O-RAN development pipelines
- **SIEM Integration**: Export findings to security information and event management systems
- **Ticketing System**: Automated issue creation for identified vulnerabilities

### 5. **User Experience Enhancements**

- **Role-based Access**: Different interfaces for developers, security analysts, and executives
- **Custom Dashboards**: User-configurable analysis dashboards
- **Notification System**: Real-time alerts for critical security issues
- **Mobile Interface**: Responsive design for mobile access

### 6. **Cloud & Scalability**

- **Azure Integration**: Deploy on Azure with auto-scaling capabilities
- **Multi-tenant Support**: Support for multiple O-RAN deployments
- **Big Data Processing**: Handle large-scale O-RAN datasets efficiently
- **Global Deployment**: Multi-region deployment for global O-RAN analysis

### 7. **Advanced Security Features**

- **Threat Intelligence**: Integration with commercial threat intelligence feeds
- **Attack Simulation**: Automated attack scenario generation and testing
- **Incident Response**: Automated response recommendations for security incidents
- **Forensic Analysis**: Deep-dive analysis capabilities for security investigations

## 🏗️ ARCHITECTURE OVERVIEW

### Current Implementation

```
O-RAN Security Test Case Generator
├── Core Application (app.py)
│   ├── Security Test Case Generation
│   ├── Dataset Analysis Interface
│   └── Data Collection Interface
├── Data Analysis Engine (oran_data_analyzer.py)
│   ├── Statistical Analysis
│   ├── Security Metrics
│   ├── Performance Metrics
│   └── Visualization Generation
├── Data Collection Engine (collect_oran_data.py)
│   ├── O-RAN Alliance Specs
│   ├── 3GPP Standards
│   ├── Cybersecurity Frameworks
│   └── Export Capabilities
├── Testing Framework
│   ├── Integration Tests
│   ├── Component Tests
│   └── API Tests
└── Data Repository (./data/)
    ├── O-RAN Components
    ├── Security Standards
    ├── Performance Data
    └── Compliance Frameworks
```

### Recommended Future Architecture

```
Cloud-Native O-RAN Security Platform
├── Frontend (React/Vue.js)
│   ├── Executive Dashboard
│   ├── Analyst Interface
│   └── Developer Console
├── API Gateway
│   ├── Authentication
│   ├── Rate Limiting
│   └── Request Routing
├── Microservices
│   ├── Data Collection Service
│   ├── Analysis Engine
│   ├── ML/AI Service
│   └── Notification Service
├── Data Layer
│   ├── Time-series Database
│   ├── Graph Database
│   └── Document Store
└── Integration Layer
    ├── O-RAN SC Connector
    ├── Cloud APIs
    └── SIEM Integrations
```

## 🤝 CONTRIBUTING

### Adding New Data Sources

1. Extend `ORANDataCollector` class with new collection methods
2. Add data source configuration to the UI
3. Implement data validation and transformation
4. Add comprehensive tests

### Adding New Analysis Metrics

1. Extend `ORANDataAnalyzer` class with new analysis methods
2. Add visualization components using Plotly
3. Update the Streamlit interface
4. Add unit tests and integration tests

### Adding New Visualizations

1. Create new Plotly chart components
2. Integrate with the analysis dashboard
3. Add export functionality
4. Test across different datasets

## 📄 LICENSE

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 ACKNOWLEDGMENTS

- O-RAN Alliance for open specifications
- 3GPP for technical standards
- NIST for cybersecurity framework
- MITRE Corporation for ATT&CK framework
- Google for Gemini AI API
- Streamlit for the web framework
