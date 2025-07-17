#!/usr/bin/env python3
"""
Simple test to check if web search tools can work
"""

def test_import():
    """Test if we can import duckduckgo_search"""
    try:
        import duckduckgo_search
        print("✅ duckduckgo_search module found")
        print(f"   Module path: {duckduckgo_search.__file__}")
        print(f"   Module contents: {dir(duckduckgo_search)}")
        return True
    except ImportError as e:
        print(f"❌ Cannot import duckduckgo_search: {e}")
        return False

def test_ddgs_import():
    """Test if we can import DDGS specifically"""
    try:
        from duckduckgo_search import DDGS
        print("✅ DDGS class imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Cannot import DDGS: {e}")
        return False
    except Exception as e:
        print(f"❌ Error importing DDGS: {e}")
        return False

def test_alternative_import():
    """Test alternative import methods"""
    try:
        # Try different import patterns
        import sys
        print(f"Python version: {sys.version}")
        
        # Try importing as ddg
        try:
            import ddg
            print("✅ ddg module found")
        except ImportError:
            print("❌ ddg module not found")
        
        # Try importing as duckduckgo
        try:
            import duckduckgo
            print("✅ duckduckgo module found")
        except ImportError:
            print("❌ duckduckgo module not found")
        
        # List installed packages
        import pkg_resources
        installed_packages = [d.project_name for d in pkg_resources.working_set]
        duck_packages = [p for p in installed_packages if 'duck' in p.lower()]
        print(f"Duck-related packages: {duck_packages}")
        
    except Exception as e:
        print(f"❌ Error in alternative import test: {e}")

def test_manual_search():
    """Test manual web search without duckduckgo_search"""
    import requests
    
    try:
        # Try a simple web request
        response = requests.get("https://httpbin.org/get", timeout=10)
        if response.status_code == 200:
            print("✅ Internet connection working")
            return True
        else:
            print(f"❌ Internet connection issue: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Internet connection error: {e}")
        return False

def main():
    print("🧪 Simple Web Search Tools Test")
    print("=" * 40)
    
    print("\n1. Testing duckduckgo_search import...")
    import_ok = test_import()
    
    if not import_ok:
        print("\n2. Testing DDGS import...")
        ddgs_ok = test_ddgs_import()
        
        if not ddgs_ok:
            print("\n3. Testing alternative imports...")
            test_alternative_import()
            
            print("\n4. Testing basic internet connectivity...")
            test_manual_search()
    else:
        print("\n2. Testing DDGS import...")
        test_ddgs_import()
    
    print("\n📊 Conclusion:")
    if import_ok:
        print("   Web search tools should work - dependency is available")
    else:
        print("   Web search tools may not work - dependency issue detected")
        print("   Suggestions:")
        print("   - Try: pip install duckduckgo-search --user")
        print("   - Try: pip install duckduckgo-search --force-reinstall")
        print("   - Check if you're using the correct Python environment")

if __name__ == "__main__":
    main()