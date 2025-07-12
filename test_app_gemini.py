#!/usr/bin/env python3
"""
Test the specific Gemini functions used in the O-RAN app
"""
import sys
import os
sys.path.append('.')

# Import the app's Gemini function
try:
    from app import call_gemini_with_retry
    import json
    
    print("🧪 Testing O-RAN app's Gemini integration...")
    
    # Test 1: Basic functionality
    system_prompt = "You are a cybersecurity expert."
    user_prompt = "Generate a simple JSON response with format: {\"test\": \"success\"}"
    
    try:
        response = call_gemini_with_retry(system_prompt, user_prompt)
        print("✅ Basic Gemini call successful")
        print(f"Response: {response[:100]}...")
        
        # Test 2: JSON parsing
        try:
            # Try to find JSON in the response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_part = response[start:end]
                parsed = json.loads(json_part)
                print("✅ JSON parsing successful")
                print(f"Parsed JSON: {parsed}")
            else:
                print("⚠️  Response doesn't contain JSON format")
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON parsing failed: {e}")
            
    except Exception as e:
        print(f"❌ Gemini call failed: {e}")
        
except ImportError as e:
    print(f"❌ Import error: {e}")
except Exception as e:
    print(f"❌ Test failed: {e}")

print("\n🏁 Test complete!")
