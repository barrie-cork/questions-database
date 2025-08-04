from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from config import Config

# Create async engine
engine = create_async_engine(
    Config.SQLALCHEMY_DATABASE_URI,
    echo=Config.APP_ENV == 'development',
    pool_size=Config.SQLALCHEMY_ENGINE_OPTIONS['pool_size'],
    pool_recycle=Config.SQLALCHEMY_ENGINE_OPTIONS['pool_recycle'],
    pool_pre_ping=Config.SQLALCHEMY_ENGINE_OPTIONS['pool_pre_ping'],
    max_overflow=Config.SQLALCHEMY_ENGINE_OPTIONS['max_overflow']
)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False
)

# Dependency to get DB session
async def get_db():
    """Provide database session for FastAPI dependency injection"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()