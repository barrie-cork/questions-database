#!/usr/bin/env python3
"""Test setup script to verify all dependencies are installed correctly"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            importlib.import_module(module_name, package_name)
        else:
            importlib.import_module(module_name)
        print(f"✓ {module_name}")
        return True
    except ImportError as e:
        print(f"✗ {module_name} - {e}")
        return False

def main():
    print("Testing Python dependencies...\n")
    
    modules = [
        # Core
        ("fastapi", None),
        ("uvicorn", None),
        ("httpx", None),
        
        # API clients
        ("mistralai", None),
        ("google.genai", None),
        
        # Processing
        ("langchain_text_splitters", None),
        ("pydantic", None),
        ("tenacity", None),
        
        # Database
        ("sqlalchemy", None),
        ("asyncpg", None),
        ("psycopg2", None),
        ("pgvector", None),
        ("alembic", None),
        
        # Utilities
        ("dotenv", None),
        ("aiofiles", None),
        
        # Development
        ("pytest", None),
        ("pytest_asyncio", None),
    ]
    
    failed = []
    for module, package in modules:
        if not test_import(module, package):
            failed.append(module)
    
    print("\n" + "="*50)
    if failed:
        print(f"❌ Failed to import {len(failed)} modules: {', '.join(failed)}")
        print("\nPlease run: pip install -r requirements.txt")
        sys.exit(1)
    else:
        print("✅ All dependencies installed successfully!")
        
    # Test environment variables
    print("\nChecking environment variables...")
    from dotenv import load_dotenv
    import os
    
    load_dotenv()
    
    env_vars = {
        "MISTRAL_API_KEY": os.getenv("MISTRAL_API_KEY"),
        "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
        "POSTGRES_USER": os.getenv("POSTGRES_USER"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD"),
    }
    
    missing_vars = [k for k, v in env_vars.items() if not v]
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease copy .env.example to .env and fill in your API keys")
    else:
        print("✅ All required environment variables are set")
    
    print("\nSetup test complete!")

if __name__ == "__main__":
    main()