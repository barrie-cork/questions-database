import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    # PostgreSQL connection
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'questionuser')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'securepassword123')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'question_bank')
    
    # Async database URL for SQLAlchemy
    SQLALCHEMY_DATABASE_URI = (
        f'postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@'
        f'{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
    )
    
    # Sync database URL for initialization
    SYNC_DATABASE_URI = (
        f'postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@'
        f'{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}'
    )
    
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Pool configuration for better performance
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_size': 10,
        'pool_recycle': 3600,
        'pool_pre_ping': True,
        'max_overflow': 20
    }
    
    # API Keys
    MISTRAL_API_KEY = os.getenv('MISTRAL_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    # Application settings
    APP_ENV = os.getenv('APP_ENV', 'development')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_SIZE', 52428800))  # 50MB default
    
    # Paths
    UPLOAD_DIR = 'uploads'
    CACHE_DIR = 'cache'
    LOG_DIR = 'logs'
    
    # Service configurations
    MISTRAL_MODEL = os.getenv('MISTRAL_MODEL', 'mistral-ocr-latest')
    GEMINI_MODEL = os.getenv('GEMINI_MODEL', 'gemini-2.5-flash')
    GEMINI_EMBEDDING_MODEL = os.getenv('GEMINI_EMBEDDING_MODEL', 'models/embedding-001')
    
    # Processing settings
    TEXT_CHUNK_SIZE = int(os.getenv('TEXT_CHUNK_SIZE', 50000))
    TEXT_CHUNK_OVERLAP = int(os.getenv('TEXT_CHUNK_OVERLAP', 200))
    EMBEDDING_DIMENSION = int(os.getenv('EMBEDDING_DIMENSION', 768))
    EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', 10))
    
    # Rate limiting
    MISTRAL_RATE_LIMIT = int(os.getenv('MISTRAL_RATE_LIMIT', 60))  # calls per minute
    GEMINI_RATE_LIMIT = int(os.getenv('GEMINI_RATE_LIMIT', 60))
    EMBEDDING_RATE_LIMIT = int(os.getenv('EMBEDDING_RATE_LIMIT', 100))