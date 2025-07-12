#!/usr/bin/env python3
"""
Setup script for O-RAN Security Test Case Generator with Gemini 2.5 Flash
Author: N. Sachin Deshik
GitHub: sachin-deshik-10
Email: nsachindeshik.ec21@rvce.edu.in
LinkedIn: https://www.linkedin.com/in/sachin-deshik-nayakula-62b93b362
"""
import os
import subprocess
import sys

def run_command(cmd):
    """Run a command and return success status"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def main():
    print("🚀 Setting up O-RAN Security Test Case Generator with Gemini 2.5 Flash")
    print("=" * 70)
    
    # Check Python version
    print("\n1. Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ ERROR: Python 3.8 or higher required")
        sys.exit(1)
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    
    # Install pipenv
    print("\n2. Installing pipenv...")
    success, stdout, stderr = run_command("pip install --user pipenv")
    if success:
        print("✅ pipenv installed successfully")
    else:
        print("⚠️  pipenv installation may have failed, but continuing...")
    
    # Install dependencies directly with pip
    print("\n3. Installing dependencies...")
    packages = [
        "streamlit", "spacy", "nltk", "google-generativeai", 
        "python-dotenv", "ipython"
    ]
    
    all_success = True
    for package in packages:
        success, stdout, stderr = run_command(f"pip install {package}")
        if not success:
            print(f"⚠️  Warning: {package} installation may have failed")
            all_success = False
    
    if all_success:
        print("✅ Dependencies installed successfully")
    else:
        print("⚠️  Some dependencies may need manual installation")
    
    # Check .env file
    print("\n4. Checking .env file...")
    if os.path.exists('.env'):
        print("✅ .env file found")
        with open('.env', 'r') as f:
            content = f.read()
            if 'GEMINI_API_KEY' in content:
                print("✅ GEMINI_API_KEY found in .env file")
            else:
                print("⚠️  GEMINI_API_KEY not found in .env file")
                print("Please add your Gemini API key to the .env file")
    else:
        print("⚠️  .env file not found")
        print("Please create a .env file with your GEMINI_API_KEY")
    
    # Test Gemini API
    print("\n5. Testing Gemini API connection...")
    success, stdout, stderr = run_command("python test_gemini.py")
    if success:
        print("✅ Gemini API test passed")
        print(f"   {stdout.strip()}")
    else:
        print("❌ Gemini API test failed")
        print("Please check your GEMINI_API_KEY in the .env file")
    
    print("\n" + "=" * 70)
    print("🎉 Setup complete!")
    print("\nTo run the application:")
    print("   streamlit run app.py")
    print("\nMake sure you have a valid Gemini API key in your .env file!")
    print("The application will be available at: http://localhost:8501")

if __name__ == "__main__":
    main()
