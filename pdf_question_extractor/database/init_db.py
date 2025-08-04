import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config

def init_database():
    """Initialize PostgreSQL with pgvector extension"""
    
    # Connection parameters
    conn_params = {
        'host': Config.POSTGRES_HOST,
        'port': Config.POSTGRES_PORT,
        'user': Config.POSTGRES_USER,
        'password': Config.POSTGRES_PASSWORD
    }
    
    try:
        # Connect to PostgreSQL server (not to a specific database)
        print("Connecting to PostgreSQL server...")
        conn = psycopg2.connect(**conn_params, database='postgres')
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Create database if not exists
        cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{Config.POSTGRES_DB}'")
        if not cur.fetchone():
            cur.execute(f"CREATE DATABASE {Config.POSTGRES_DB}")
            print(f"Database '{Config.POSTGRES_DB}' created successfully")
        else:
            print(f"Database '{Config.POSTGRES_DB}' already exists")
        
        cur.close()
        conn.close()
        
        # Connect to the application database
        print(f"\nConnecting to database '{Config.POSTGRES_DB}'...")
        conn = psycopg2.connect(**conn_params, database=Config.POSTGRES_DB)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cur = conn.cursor()
        
        # Create extensions
        print("Creating extensions...")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        print("Extensions created successfully")
        
        # Check if extensions were created
        cur.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm')")
        extensions = cur.fetchall()
        print(f"Active extensions: {[ext[0] for ext in extensions]}")
        
        # Execute schema.sql
        schema_path = os.path.join(os.path.dirname(__file__), 'schema.sql')
        print(f"\nExecuting schema from: {schema_path}")
        
        with open(schema_path, 'r') as f:
            cur.execute(f.read())
        
        print("Schema created successfully")
        
        # Verify tables were created
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            AND table_type = 'BASE TABLE'
        """)
        tables = cur.fetchall()
        print(f"\nCreated tables: {[table[0] for table in tables]}")
        
        cur.close()
        conn.close()
        
        print("\nDatabase initialization completed successfully!")
        
    except psycopg2.Error as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    init_database()