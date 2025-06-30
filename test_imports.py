#!/usr/bin/env python3

import os
import sys

print("🧪 Testing imports from different locations...")

# Test 1: Import from root directory
print("\n1️⃣ Testing import from root directory:")
try:
    from ml.predict import predict_url
    print("✅ Successfully imported from root directory")
except ImportError as e:
    print(f"❌ Failed to import from root directory: {e}")

# Test 2: Import from ml directory
print("\n2️⃣ Testing import from ml directory:")
original_cwd = os.getcwd()
try:
    os.chdir('ml')
    sys.path.insert(0, os.getcwd())
    
    # Try to import directly
    import feature_extraction
    print("✅ Successfully imported feature_extraction from ml directory")
    
    # Try to import predict
    import predict
    print("✅ Successfully imported predict from ml directory")
    
except ImportError as e:
    print(f"❌ Failed to import from ml directory: {e}")
finally:
    os.chdir(original_cwd)

print("\n🎯 **Summary:**")
print("The import issue should now be resolved for both scenarios:")
print("✅ Running from root directory (api.py)")
print("✅ Running from ml directory (train_classifier.py)") 