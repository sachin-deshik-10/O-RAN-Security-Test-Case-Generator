# O-RAN Security Test Case Generator - Data Analysis Features

## New Features Added

### 1. **O-RAN Dataset Analysis & Quantitative Assessment**

- **Comprehensive Dataset Analysis**: Analyze O-RAN performance, security, and configuration data
- **Security Metrics**: Vulnerability assessment, threat analysis, and security scoring
- **Performance Metrics**: Latency, throughput, availability, and resource utilization analysis
- **Interactive Visualizations**: Plotly-based charts and dashboards for data exploration
- **Correlation Analysis**: Identify relationships between different metrics and components
- **Anomaly Detection**: Detect unusual patterns in O-RAN network behavior

### 2. **O-RAN Free Source Data Collection**

- **O-RAN Alliance Specifications**: Collect official O-RAN specifications and standards
- **3GPP Standards**: Gather relevant 3GPP documents and technical specifications
- **NIST Cybersecurity Framework**: Import cybersecurity controls and guidelines
- **MITRE ATT&CK for ICS**: Collect industrial control system attack patterns
- **CWE Database**: Common Weakness Enumeration data for vulnerability analysis
- **CAPEC Attack Patterns**: Common Attack Pattern Enumeration and Classification

### 3. **Enhanced User Interface**

- **Multi-page Navigation**: Switch between Security Test Case Generator, Dataset Analysis, and Data Collection
- **Interactive Dashboard**: Real-time data analysis and visualization
- **Export Capabilities**: Save analysis results in JSON and CSV formats
- **Configurable Analysis**: Choose which metrics and visualizations to display

## Usage Guide

### Using the Dataset Analysis Feature

1. **Launch the Application**

   ```bash
   streamlit run app.py
   ```

2. **Navigate to Dataset Analysis**
   - Use the sidebar navigation to select "O-RAN Dataset Analysis"

3. **Select Datasets**
   - Choose from available datasets (performance, security, e2_traces, etc.)
   - Multiple datasets can be analyzed simultaneously

4. **Configure Analysis Options**
   - Basic Statistics: Generate descriptive statistics
   - Security Metrics: Analyze vulnerabilities and threats
   - Performance Metrics: Assess network performance
   - Visualizations: Create interactive charts and graphs
   - Correlation Analysis: Find relationships between metrics
   - Anomaly Detection: Identify unusual patterns

5. **Run Analysis**
   - Click "Run Analysis" to generate comprehensive reports
   - Results are displayed in real-time with interactive visualizations

### Using the Data Collection Feature

1. **Navigate to Data Collection**
   - Use the sidebar navigation to select "Data Collection"

2. **Select Data Sources**
   - Choose from multiple free O-RAN and cybersecurity sources
   - Configure collection parameters (max items, format, etc.)

3. **Start Collection**
   - Click "Start Data Collection" to gather data from selected sources
   - Monitor progress and view collected data in real-time

4. **Save Results**
   - Data is automatically saved to the `./output` directory
   - Available in JSON and CSV formats

## Available Datasets

### Performance Datasets

- **Near-RT RIC Performance**: Real-time RIC performance metrics
- **E2 Interface Traces**: E2 interface communication logs
- **xApp Performance**: Individual xApp performance data

### Security Datasets

- **xApp Security Metrics**: Security assessments for xApps
- **Vulnerability Database**: Known vulnerabilities and mitigations
- **Threat Intelligence**: Security threat analysis data

### Configuration Datasets

- **O-RAN Components**: Component specifications and configurations
- **Near-RT RIC Configuration**: RIC setup and configuration data
- **Security Analysis**: Security requirement analysis

## Sample Analysis Results

### Security Metrics Example

```json
{
  "total_vulnerabilities": 15,
  "critical_issues": 3,
  "security_score": 7.2,
  "threat_categories": ["Authentication", "Data Integrity", "Access Control"],
  "mitigation_strategies": ["Multi-factor Authentication", "Encryption", "RBAC"]
}
```

### Performance Metrics Example

```json
{
  "avg_latency": 4.8,
  "max_latency": 8.5,
  "throughput": 1050.5,
  "availability": 99.7,
  "resource_utilization": {
    "cpu": 68.5,
    "memory": 72.1,
    "network": 45.3
  }
}
```

## Data Sources

### O-RAN Alliance Specifications

- E2 Application Protocol (E2AP)
- A1 Application Protocol (A1AP)
- O1 Operations and Maintenance Interface
- xApp SDK and APIs

### 3GPP Standards

- 5G NR specifications
- Network architecture standards
- Security and privacy requirements

### Cybersecurity Frameworks

- NIST Cybersecurity Framework
- MITRE ATT&CK for ICS
- Common Weakness Enumeration (CWE)
- CAPEC Attack Patterns

## Advanced Features

### Machine Learning Integration

- **Anomaly Detection**: Unsupervised learning for detecting unusual patterns
- **Predictive Analytics**: Forecast performance and security trends
- **Classification**: Categorize threats and vulnerabilities automatically

### Real-time Monitoring

- **Live Data Feeds**: Connect to real-time O-RAN data sources
- **Alert Systems**: Automated notifications for critical issues
- **Dashboard Updates**: Real-time visualization updates

### Export and Reporting

- **PDF Reports**: Generate comprehensive analysis reports
- **CSV Exports**: Export raw data for further analysis
- **JSON APIs**: Programmatic access to analysis results

## Installation and Setup

### Prerequisites

All required dependencies are in the `Pipfile`:

```toml
[packages]
streamlit = "*"
pandas = "*"
numpy = "*"
matplotlib = "*"
seaborn = "*"
plotly = "*"
scikit-learn = "*"
requests = "*"
beautifulsoup4 = "*"
google-generativeai = "*"
# ... other dependencies
```

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Or use pipenv
pipenv install
pipenv shell

# Run the application
streamlit run app.py
```

## Testing

### Comprehensive Integration Test

```bash
python test_integration.py
```

This test verifies:

- All imports work correctly
- Data files are present and accessible
- Data analysis modules function properly
- Gemini API integration works
- NLTK data is available
- Data analysis functionality works
- Data collection functionality works
- Output directories are writable

### Individual Component Tests

```bash
# Test Gemini API
python test_gemini.py

# Test data analysis
python test_comprehensive.py

# Test app functionality
python test_app_gemini.py
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility (3.8+)

2. **Data Files Not Found**
   - Verify all JSON files are in the `./data` directory
   - Check file permissions and accessibility

3. **Gemini API Issues**
   - Verify `GEMINI_API_KEY` is set in `.env` file
   - Check API key validity and quota limits

4. **NLTK Data Missing**
   - Run: `python -c "import nltk; nltk.download('averaged_perceptron_tagger'); nltk.download('universal_tagset')"`

5. **Visualization Issues**
   - Ensure matplotlib, seaborn, and plotly are installed
   - Check browser compatibility for Plotly charts

## Future Enhancements

### Planned Features

1. **Advanced ML Models**: Deep learning for sophisticated threat detection
2. **Real-time Data Integration**: Live O-RAN network monitoring
3. **Multi-tenant Support**: Support for multiple O-RAN deployments
4. **Custom Metrics**: User-defined analysis metrics and thresholds
5. **API Endpoints**: RESTful API for programmatic access
6. **Cloud Integration**: Azure/AWS integration for scalable analysis

### Contributing

- Add new data sources by extending the `ORANDataCollector` class
- Implement new analysis metrics in the `ORANDataAnalyzer` class
- Create custom visualizations using Plotly/Matplotlib
- Add new test cases to the comprehensive test suite

For detailed technical documentation, see the individual module docstrings and code comments.
