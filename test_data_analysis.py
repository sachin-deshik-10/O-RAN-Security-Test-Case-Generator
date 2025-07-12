#!/usr/bin/env python3
"""
Test the data analysis functionality
"""

from oran_data_analyzer import ORANDataAnalyzer
import json
import pandas as pd

def test_data_analysis():
    print("Testing O-RAN Data Analysis...")
    
    # Initialize analyzer
    analyzer = ORANDataAnalyzer()
    
    print(f"\n1. Loaded Datasets: {len(analyzer.datasets)}")
    for dataset_name in analyzer.datasets.keys():
        print(f"   - {dataset_name}")
    
    # Test basic statistics
    print("\n2. Testing Basic Statistics...")
    for dataset_name in list(analyzer.datasets.keys())[:3]:  # Test first 3 datasets
        try:
            stats = analyzer.get_basic_statistics(dataset_name)
            if stats:
                print(f"   {dataset_name}: Generated statistics for {len(stats)} metrics")
            else:
                print(f"   {dataset_name}: No statistics generated")
        except Exception as e:
            print(f"   {dataset_name}: Error - {e}")
    
    # Test security metrics
    print("\n3. Testing Security Metrics...")
    for dataset_name in ['security', 'vulnerabilities', 'oran_security_analysis']:
        if dataset_name in analyzer.datasets:
            try:
                security_metrics = analyzer.analyze_security_metrics(dataset_name)
                if security_metrics:
                    print(f"   {dataset_name}: Security Score = {security_metrics.get('security_score', 'N/A')}")
                    print(f"   {dataset_name}: Total Vulnerabilities = {security_metrics.get('total_vulnerabilities', 'N/A')}")
                else:
                    print(f"   {dataset_name}: No security metrics generated")
            except Exception as e:
                print(f"   {dataset_name}: Error - {e}")
    
    # Test performance metrics
    print("\n4. Testing Performance Metrics...")
    for dataset_name in ['performance', 'e2_traces']:
        if dataset_name in analyzer.datasets:
            try:
                perf_metrics = analyzer.analyze_performance_metrics(dataset_name)
                if perf_metrics:
                    print(f"   {dataset_name}: Avg Latency = {perf_metrics.get('avg_latency', 'N/A')}")
                    print(f"   {dataset_name}: Throughput = {perf_metrics.get('throughput', 'N/A')}")
                    print(f"   {dataset_name}: Availability = {perf_metrics.get('availability', 'N/A')}")
                else:
                    print(f"   {dataset_name}: No performance metrics generated")
            except Exception as e:
                print(f"   {dataset_name}: Error - {e}")
    
    # Test visualization creation
    print("\n5. Testing Visualization Creation...")
    for dataset_name in ['security', 'performance']:
        if dataset_name in analyzer.datasets:
            try:
                fig = analyzer.create_security_dashboard(dataset_name)
                if fig:
                    print(f"   {dataset_name}: Security dashboard created successfully")
                else:
                    print(f"   {dataset_name}: No visualization created")
            except Exception as e:
                print(f"   {dataset_name}: Error - {e}")
    
    # Test comprehensive analysis
    print("\n6. Testing Comprehensive Analysis...")
    try:
        analysis_results = analyzer.run_comprehensive_analysis()
        if analysis_results:
            print(f"   Comprehensive analysis completed: {len(analysis_results)} datasets analyzed")
            for dataset_name, results in analysis_results.items():
                if results:
                    print(f"   - {dataset_name}: {len(results)} analysis results")
        else:
            print("   No comprehensive analysis results")
    except Exception as e:
        print(f"   Error in comprehensive analysis: {e}")
    
    # Save analysis results
    print("\n7. Saving Analysis Results...")
    try:
        results_summary = {
            'datasets_analyzed': len(analyzer.datasets),
            'dataset_names': list(analyzer.datasets.keys()),
            'analysis_timestamp': pd.Timestamp.now().isoformat(),
            'analysis_results': analyzer.analysis_results
        }
        
        with open('./output/test_analysis_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print("   Analysis results saved to ./output/test_analysis_results.json")
    except Exception as e:
        print(f"   Error saving results: {e}")
    
    print("\nData analysis test completed successfully!")
    
    return analyzer

if __name__ == "__main__":
    test_data_analysis()
