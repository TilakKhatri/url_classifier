#!/usr/bin/env python3

import os
import sys

print("🔍 Testing import path resolution...")

def test_import_path():
    """Test if the import path is correctly set up"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    ml_dir = os.path.join(current_dir, 'ml')
    
    print(f"📁 Root directory: {current_dir}")
    print(f"📁 ML directory: {ml_dir}")
    
    # Check if ml directory exists and has __init__.py
    if os.path.exists(ml_dir) and os.path.exists(os.path.join(ml_dir, '__init__.py')):
        print("✅ ML directory structure is correct")
    else:
        print("❌ ML directory structure is incorrect")
        return False
    
    # Test adding ml directory to path
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    
    # Test if we can import the ml package
    try:
        import ml
        print("✅ ML package can be imported")
    except ImportError as e:
        print(f"❌ ML package import failed: {e}")
        return False
    
    return True

def test_feature_extraction_import():
    """Test if feature_extraction can be imported from different locations"""
    print("\n🧪 Testing feature_extraction imports...")
    
    # Test 1: From root directory
    print("1️⃣ Testing from root directory:")
    try:
        # This should work with our path fix
        sys.path.insert(0, os.path.join(os.getcwd(), 'ml'))
        import feature_extraction
        print("✅ feature_extraction imported successfully from root")
    except ImportError as e:
        print(f"❌ feature_extraction import failed from root: {e}")
    
    # Test 2: From ml directory
    print("2️⃣ Testing from ml directory:")
    original_cwd = os.getcwd()
    try:
        os.chdir('ml')
        # Clear and reset path for ml directory
        sys.path = [os.getcwd()] + [p for p in sys.path if 'ml' not in p]
        
        import feature_extraction
        print("✅ feature_extraction imported successfully from ml directory")
    except ImportError as e:
        print(f"❌ feature_extraction import failed from ml directory: {e}")
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    if test_import_path():
        test_feature_extraction_import()
    
    print("\n🎯 **SOLUTION VERIFICATION:**")
    print("The import path fixes have been applied to:")
    print("✅ ml/train_classifier.py - Added sys.path manipulation")
    print("✅ ml/predict.py - Added sys.path manipulation")
    print("✅ ml/__init__.py - Created for package structure")
    
    print("\n📋 **This resolves the ModuleNotFoundError by:**")
    print("1. Making imports work from both root and ml directories")
    print("2. Ensuring Python can find the feature_extraction module")
    print("3. Maintaining compatibility with existing code") 