"""Database initialization script for PDF Question Extractor"""
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import Config


def init_postgresql():
    """Initialize PostgreSQL with pgvector extension"""
    print("Initializing PostgreSQL database...")
    
    try:
        # Connect to PostgreSQL server (not to a specific database)
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD
        )
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Check if database exists
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{Config.POSTGRES_DB}'")
        if not cur.fetchone():
            print(f"Creating database '{Config.POSTGRES_DB}'...")
            cur.execute(f"CREATE DATABASE {Config.POSTGRES_DB}")
            print(f"Database '{Config.POSTGRES_DB}' created successfully.")
        else:
            print(f"Database '{Config.POSTGRES_DB}' already exists.")
        
        cur.close()
        conn.close()
        
        # Connect to the specific database and create extensions
        print("Creating extensions and schema...")
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            database=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD
        )
        cur = conn.cursor()
        
        # Create extensions
        print("Creating pgvector extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        print("Creating pg_trgm extension...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        
        conn.commit()
        
        # Read and execute schema file
        schema_path = Path(__file__).parent / 'schema.sql'
        if schema_path.exists():
            print("Executing schema.sql...")
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            cur.execute(schema_sql)
            conn.commit()
            print("Schema created successfully.")
        else:
            print(f"Warning: schema.sql not found at {schema_path}")
        
        cur.close()
        conn.close()
        
        print("Database initialization completed successfully!")
        return True
        
    except psycopg2.Error as e:
        print(f"PostgreSQL error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def verify_database():
    """Verify database setup"""
    try:
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            database=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD
        )
        cur = conn.cursor()
        
        # Check extensions
        print("\nVerifying extensions...")
        cur.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')")
        extensions = [row[0] for row in cur.fetchall()]
        print(f"Installed extensions: {extensions}")
        
        # Check tables
        print("\nVerifying tables...")
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = [row[0] for row in cur.fetchall()]
        print(f"Created tables: {tables}")
        
        # Check indexes
        print("\nVerifying indexes...")
        cur.execute("""
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public'
            ORDER BY indexname
        """)
        indexes = [row[0] for row in cur.fetchall()]
        print(f"Created indexes: {len(indexes)} indexes")
        
        cur.close()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"Verification error: {e}")
        return False


def create_readonly_user():
    """Create read-only user for MCP access (requires superuser privileges)"""
    try:
        conn = psycopg2.connect(
            host=Config.POSTGRES_HOST,
            port=Config.POSTGRES_PORT,
            database=Config.POSTGRES_DB,
            user=Config.POSTGRES_USER,
            password=Config.POSTGRES_PASSWORD
        )
        cur = conn.cursor()
        
        readonly_password = os.getenv('READONLY_PASSWORD', 'readonly_password')
        
        # Create read-only user
        cur.execute("SELECT 1 FROM pg_user WHERE usename = 'readonly_user'")
        if not cur.fetchone():
            print("\nCreating read-only user...")
            cur.execute(f"CREATE ROLE readonly_user WITH LOGIN PASSWORD '{readonly_password}'")
            cur.execute("GRANT CONNECT ON DATABASE question_bank TO readonly_user")
            cur.execute("GRANT USAGE ON SCHEMA public TO readonly_user")
            cur.execute("GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user")
            cur.execute("ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO readonly_user")
            conn.commit()
            print("Read-only user created successfully.")
        else:
            print("Read-only user already exists.")
        
        cur.close()
        conn.close()
        
    except psycopg2.Error as e:
        print(f"Note: Could not create read-only user (may require superuser privileges): {e}")


if __name__ == "__main__":
    # Initialize database
    if init_postgresql():
        # Verify setup
        verify_database()
        
        # Try to create read-only user
        create_readonly_user()
    else:
        print("Database initialization failed!")
        sys.exit(1)