"""
Basic test to verify all imports work.
Run this to validate your setup.
"""
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

def test_imports():
    """Test that all core modules can be imported."""
    # Core modules
    try:
        from core.dataset import Dataset, ColumnSchema
        print("✓ Core modules imported successfully")
    except ImportError as e:
        print(f"❌ Could not import core modules: {e}")
        raise
    
    # Third-party libraries
    import pandas as pd
    import numpy as np
    import pydantic
    print("✓ Third-party libraries imported successfully")
    
    # Verify versions
    print(f"  Pandas version: {pd.__version__}")
    print(f"  Pydantic version: {pydantic.__version__}")
    
    return True

if __name__ == "__main__":
    try:
        test_imports()
        print("\n✅ All imports working correctly!")
    except Exception as e:
        print(f"\n❌ Import error: {e}")
        sys.exit(1)