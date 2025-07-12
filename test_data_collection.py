#!/usr/bin/env python3
"""
Test the data collection functionality
"""

from collect_oran_data import ORANDataCollector
import json

def test_data_collection():
    print("Testing O-RAN Data Collection...")
    
    # Initialize collector
    collector = ORANDataCollector()
    
    # Test O-RAN Alliance specifications collection
    print("\n1. Collecting O-RAN Alliance Specifications...")
    specs_data = collector.collect_oran_alliance_specs()
    print(f"   Collected {len(specs_data.get('specifications', []))} specifications")
    
    if specs_data.get('specifications'):
        print(f"   Sample spec: {specs_data['specifications'][0]['title']}")
        print(f"   Working Group: {specs_data['specifications'][0]['working_group']}")
        print(f"   Security Requirements: {len(specs_data['specifications'][0]['security_requirements'])}")
    
    # Test 3GPP standards collection
    print("\n2. Collecting 3GPP Standards...")
    try:
        standards_data = collector.collect_3gpp_standards()
        print(f"   Collected {len(standards_data.get('standards', []))} standards")
        if standards_data.get('standards'):
            print(f"   Sample standard: {standards_data['standards'][0]['title']}")
    except Exception as e:
        print(f"   Error collecting 3GPP standards: {e}")
    
    # Test NIST framework collection
    print("\n3. Collecting NIST Cybersecurity Framework...")
    try:
        nist_data = collector.collect_nist_framework()
        print(f"   Collected {len(nist_data.get('controls', []))} controls")
        if nist_data.get('controls'):
            print(f"   Sample control: {nist_data['controls'][0]['title']}")
    except Exception as e:
        print(f"   Error collecting NIST framework: {e}")
    
    # Save results
    print("\n4. Saving results...")
    all_data = {
        'o_ran_specs': specs_data,
        'collection_timestamp': collector.collected_data.get('timestamp', 'N/A')
    }
    
    with open('./output/test_collection_results.json', 'w') as f:
        json.dump(all_data, f, indent=2)
    
    print("   Results saved to ./output/test_collection_results.json")
    print("\nData collection test completed successfully!")
    
    return all_data

if __name__ == "__main__":
    test_data_collection()
