#!/usr/bin/env python3
"""
Test script to verify Gemini API integration is working
"""
import os
import sys
sys.path.append('.')

# Test the Gemini API integration
try:
    import google.generativeai as genai
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get API key
    api_key = os.getenv('GEMINI_API_KEY')
    
    if not api_key:
        print("❌ ERROR: GEMINI_API_KEY not found in environment variables")
        sys.exit(1)
    
    # Configure Gemini
    genai.configure(api_key=api_key)
    
    # Test API call
    model = genai.GenerativeModel('gemini-2.0-flash-exp')
    
    test_prompt = "Hello, Gemini! Please respond with 'API connection successful' to confirm you're working."
    
    response = model.generate_content(test_prompt)
    
    if response.text:
        print("✅ SUCCESS: Gemini API integration working!")
        print(f"Response: {response.text}")
    else:
        print("❌ ERROR: Empty response from Gemini")
        
except ImportError as e:
    print(f"❌ ERROR: Missing required packages: {e}")
    print("Please run: pip install google-generativeai python-dotenv")
    
except Exception as e:
    print(f"❌ ERROR: API test failed: {e}")
    print("Please check your GEMINI_API_KEY in the .env file")
