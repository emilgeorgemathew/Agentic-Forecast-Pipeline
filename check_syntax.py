
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    # Just try to import Interface to check for syntax errors
    # Note: Streamlit apps might run code on import, so we wrap in try/except
    # but we are mainly looking for SyntaxError
    import Interface
    print("✅ Interface.py imported successfully (Syntax OK)")
except SyntaxError as e:
    print(f"❌ Syntax Error in Interface.py: {e}")
    sys.exit(1)
except Exception as e:
    # Other errors (like missing streamlit context) are expected on import
    print(f"✅ Interface.py syntax check passed (Runtime error expected on import: {e})")
