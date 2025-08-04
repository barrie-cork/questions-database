"""Configuration settings for PDF Question Extractor"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.absolute()


class Config:
    """Base configuration"""
    # Flask settings
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    DEBUG = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    # PostgreSQL configuration
    POSTGRES_USER = os.getenv('POSTGRES_USER', 'postgres')
    POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD', 'password')
    POSTGRES_HOST = os.getenv('POSTGRES_HOST', 'localhost')
    POSTGRES_PORT = os.getenv('POSTGRES_PORT', '5432')
    POSTGRES_DB = os.getenv('POSTGRES_DB', 'question_bank')
    
    SQLALCHEMY_DATABASE_URI = (
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
    
    # Upload settings
    UPLOAD_FOLDER = os.path.join(BASE_DIR, os.getenv('UPLOAD_FOLDER', 'uploads'))
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 52428800))  # 50MB
    ALLOWED_EXTENSIONS = {'pdf'}
    
    # Pagination
    QUESTIONS_PER_PAGE = 20
    
    # Processing settings
    MAX_PAGES_PER_PDF = 1000
    MAX_PDF_SIZE_MB = 50
    PROCESSING_TIMEOUT = 600  # 10 minutes
    
    # Embedding settings
    EMBEDDING_MODEL = 'gemini-embedding-001'
    EMBEDDING_DIMENSION = 768
    EMBEDDING_BATCH_SIZE = 100
    
    # Retry settings
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds
    
    @staticmethod
    def init_app(app):
        """Initialize application with configuration"""
        # Create upload folder if it doesn't exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        
    @classmethod
    def validate(cls):
        """Validate required configuration"""
        errors = []
        
        if not cls.MISTRAL_API_KEY:
            errors.append("MISTRAL_API_KEY is required")
        
        if not cls.GOOGLE_API_KEY:
            errors.append("GOOGLE_API_KEY is required")
        
        if not cls.SECRET_KEY or cls.SECRET_KEY == 'dev-secret-key-change-in-production':
            errors.append("Please set a secure SECRET_KEY")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    FLASK_ENV = 'production'


class TestingConfig(Config):
    """Testing configuration"""
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'


# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}