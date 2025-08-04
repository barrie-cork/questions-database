#!/usr/bin/env python3
"""Test script to verify project setup"""

print("🔍 Testing PDF Question Extractor Setup...")

# Test imports
try:
    import flask
    print("✅ Flask installed:", flask.__version__)
except ImportError:
    print("❌ Flask not installed")

try:
    import sqlalchemy
    print("✅ SQLAlchemy installed:", sqlalchemy.__version__)
except ImportError:
    print("❌ SQLAlchemy not installed")

try:
    import psycopg2
    print("✅ psycopg2 installed")
except ImportError:
    print("❌ psycopg2 not installed")

try:
    import pgvector
    print("✅ pgvector installed")
except ImportError:
    print("❌ pgvector not installed")

try:
    from dotenv import load_dotenv
    import os
    load_dotenv()
    print("✅ python-dotenv installed")
    
    # Check environment variables
    has_mistral = bool(os.getenv('MISTRAL_API_KEY'))
    has_google = bool(os.getenv('GOOGLE_API_KEY'))
    
    print(f"{'✅' if has_mistral else '❌'} MISTRAL_API_KEY {'set' if has_mistral else 'not set'}")
    print(f"{'✅' if has_google else '❌'} GOOGLE_API_KEY {'set' if has_google else 'not set'}")
    
except ImportError:
    print("❌ python-dotenv not installed")

# Test config import
try:
    from config import Config
    print("✅ Config module loads successfully")
except Exception as e:
    print(f"❌ Config module error: {e}")

print("\n📋 Next steps:")
print("1. Edit .env file with your API keys")
print("2. Set up PostgreSQL database")
print("3. Run database initialization")
print("4. Start building services!")