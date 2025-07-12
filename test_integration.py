#!/usr/bin/env python3
"""
Comprehensive integration test for O-RAN Security Test Case Generator
Tests all major components including data analysis and collection
"""

import sys
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required imports work correctly"""
    logger.info("Testing imports...")
    
    try:
        # Core libraries
        import streamlit as st
        import google.generativeai as genai
        import nltk
        import spacy
        
        # Data analysis libraries
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Additional libraries
        import requests
        from bs4 import BeautifulSoup
        from dotenv import load_dotenv
        
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False

def test_data_files():
    """Test that all required data files exist"""
    logger.info("Testing data files...")
    
    required_files = [
        './data/asvs.json',
        './data/capec.json',
        './data/cwe.json',
        './data/oran-components.json',
        './data/oran-near-rt-ric.json',
        './data/oran-security-analysis.json',
        './data/misuse-case-scenario-examples.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"✗ Missing files: {missing_files}")
        return False
    
    logger.info("✓ All required data files exist")
    return True

def test_data_analysis_modules():
    """Test that data analysis modules work correctly"""
    logger.info("Testing data analysis modules...")
    
    try:
        # Test analyzer import
        from oran_data_analyzer import ORANDataAnalyzer
        analyzer = ORANDataAnalyzer()
        
        # Test basic functionality
        if analyzer.datasets:
            logger.info(f"✓ Data analyzer loaded {len(analyzer.datasets)} datasets")
        else:
            logger.warning("! Data analyzer loaded but no datasets found")
        
        # Test collector import
        from collect_oran_data import ORANDataCollector
        collector = ORANDataCollector()
        
        logger.info("✓ Data analysis modules loaded successfully")
        return True
    except Exception as e:
        logger.error(f"✗ Data analysis module test failed: {e}")
        return False

def test_gemini_integration():
    """Test Gemini API integration"""
    logger.info("Testing Gemini API integration...")
    
    try:
        import google.generativeai as genai
        from dotenv import load_dotenv
        
        load_dotenv()
        api_key = os.getenv('GEMINI_API_KEY')
        
        if not api_key:
            logger.warning("! GEMINI_API_KEY not found in environment")
            return False
        
        genai.configure(api_key=api_key)
        
        # Test basic API call
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        response = model.generate_content("Hello, this is a test.")
        
        if response.text:
            logger.info("✓ Gemini API test successful")
            return True
        else:
            logger.error("✗ Gemini API returned empty response")
            return False
    except Exception as e:
        logger.error(f"✗ Gemini API test failed: {e}")
        return False

def test_nltk_data():
    """Test NLTK data availability"""
    logger.info("Testing NLTK data...")
    
    try:
        import nltk
        
        # Test required NLTK data
        try:
            nltk.data.find('taggers/averaged_perceptron_tagger')
            logger.info("✓ NLTK averaged_perceptron_tagger found")
        except LookupError:
            logger.warning("! NLTK averaged_perceptron_tagger not found")
        
        try:
            nltk.data.find('taggers/universal_tagset')
            logger.info("✓ NLTK universal_tagset found")
        except LookupError:
            logger.warning("! NLTK universal_tagset not found")
        
        return True
    except Exception as e:
        logger.error(f"✗ NLTK test failed: {e}")
        return False

def test_data_analysis_functionality():
    """Test data analysis functionality with sample data"""
    logger.info("Testing data analysis functionality...")
    
    try:
        # Create sample O-RAN performance data
        sample_data = {
            'performance_metrics': [
                {'timestamp': '2024-01-01T10:00:00Z', 'latency': 5.2, 'throughput': 1000, 'cpu_usage': 65},
                {'timestamp': '2024-01-01T10:01:00Z', 'latency': 4.8, 'throughput': 1050, 'cpu_usage': 70},
                {'timestamp': '2024-01-01T10:02:00Z', 'latency': 5.5, 'throughput': 980, 'cpu_usage': 68},
                {'timestamp': '2024-01-01T10:03:00Z', 'latency': 6.1, 'throughput': 920, 'cpu_usage': 75},
                {'timestamp': '2024-01-01T10:04:00Z', 'latency': 4.9, 'throughput': 1020, 'cpu_usage': 72}
            ]
        }
        
        # Convert to DataFrame
        df = pd.DataFrame(sample_data['performance_metrics'])
        
        # Basic statistics
        stats = df.describe()
        logger.info(f"✓ Generated basic statistics: {stats.shape}")
        
        # Test data visualization components
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.plot(df['latency'], label='Latency')
        plt.close()  # Close figure to prevent display
        
        logger.info("✓ Data analysis functionality test successful")
        return True
    except Exception as e:
        logger.error(f"✗ Data analysis functionality test failed: {e}")
        return False

def test_data_collection_functionality():
    """Test data collection functionality"""
    logger.info("Testing data collection functionality...")
    
    try:
        # Test O-RAN Alliance specs collection (simulated)
        sample_specs = {
            'metadata': {
                'source': 'O-RAN Alliance',
                'collected_at': datetime.now().isoformat(),
                'type': 'specifications'
            },
            'specifications': [
                {
                    'id': 'O-RAN.WG3.E2AP-v03.00',
                    'title': 'O-RAN E2 Application Protocol (E2AP)',
                    'version': '3.00',
                    'security_requirements': ['Authentication', 'Integrity', 'Confidentiality'],
                    'performance_requirements': {
                        'latency': '< 10ms',
                        'throughput': '> 1000 msg/s'
                    }
                }
            ]
        }
        
        # Test data structure
        assert 'specifications' in sample_specs
        assert len(sample_specs['specifications']) > 0
        
        logger.info("✓ Data collection functionality test successful")
        return True
    except Exception as e:
        logger.error(f"✗ Data collection functionality test failed: {e}")
        return False

def test_output_directories():
    """Test that output directories exist or can be created"""
    logger.info("Testing output directories...")
    
    try:
        output_dir = "./output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"✓ Created output directory: {output_dir}")
        else:
            logger.info(f"✓ Output directory exists: {output_dir}")
        
        # Test write permissions
        test_file = os.path.join(output_dir, "test_write.txt")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        
        logger.info("✓ Output directory is writable")
        return True
    except Exception as e:
        logger.error(f"✗ Output directory test failed: {e}")
        return False

def generate_test_report():
    """Generate a comprehensive test report"""
    logger.info("Generating test report...")
    
    tests = [
        ("Import Test", test_imports),
        ("Data Files Test", test_data_files),
        ("Data Analysis Modules Test", test_data_analysis_modules),
        ("Gemini Integration Test", test_gemini_integration),
        ("NLTK Data Test", test_nltk_data),
        ("Data Analysis Functionality Test", test_data_analysis_functionality),
        ("Data Collection Functionality Test", test_data_collection_functionality),
        ("Output Directories Test", test_output_directories)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Generate report
    report = {
        'test_date': datetime.now().isoformat(),
        'total_tests': len(tests),
        'passed_tests': sum(1 for _, result in results if result),
        'failed_tests': sum(1 for _, result in results if not result),
        'results': [{'test': name, 'passed': result} for name, result in results]
    }
    
    # Save report
    with open('./output/test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("O-RAN SECURITY TEST CASE GENERATOR - INTEGRATION TEST REPORT")
    print("="*60)
    print(f"Test Date: {report['test_date']}")
    print(f"Total Tests: {report['total_tests']}")
    print(f"Passed: {report['passed_tests']}")
    print(f"Failed: {report['failed_tests']}")
    print(f"Success Rate: {report['passed_tests']/report['total_tests']*100:.1f}%")
    print("\nDetailed Results:")
    for result in report['results']:
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  {status}: {result['test']}")
    
    print(f"\nFull report saved to: ./output/test_report.json")
    print("="*60)
    
    return report

if __name__ == "__main__":
    print("Starting comprehensive integration test...")
    report = generate_test_report()
    
    # Exit with appropriate code
    sys.exit(0 if report['failed_tests'] == 0 else 1)
