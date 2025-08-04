"""
Test configuration and fixtures for PDF Question Extractor
"""
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock, patch
import os
import tempfile
from pathlib import Path

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.pool import NullPool
from fastapi.testclient import TestClient
import httpx

# Import models and database
from database.models import Base
from database.session import get_db
from app import app
from config import Config


# Set test environment
os.environ['APP_ENV'] = 'test'


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine"""
    # Use SQLite for testing
    test_db_url = "sqlite+aiosqlite:///:memory:"
    engine = create_async_engine(
        test_db_url,
        echo=False,
        poolclass=NullPool,
    )
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session"""
    async_session_maker = async_sessionmaker(
        test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    
    async with async_session_maker() as session:
        yield session
        await session.rollback()


@pytest.fixture
def test_client(test_session) -> Generator[TestClient, None, None]:
    """Create test client with overridden database dependency"""
    
    async def override_get_db():
        yield test_session
    
    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    app.dependency_overrides.clear()


@pytest.fixture
def mock_mistral_client():
    """Mock Mistral OCR client"""
    with patch('services.ocr_service.Mistral') as mock:
        client = Mock()
        mock.return_value = client
        
        # Mock the chat completions
        chat = Mock()
        client.chat = chat
        completions = Mock()
        chat.completions = completions
        
        # Create async complete method
        async_complete = AsyncMock()
        completions.create = async_complete
        
        yield {
            'client': client,
            'complete': async_complete
        }


@pytest.fixture
def mock_gemini_client():
    """Mock Gemini client for LLM and embeddings"""
    with patch('google.genai.Client') as mock:
        client = Mock()
        mock.return_value = client
        
        # Mock models
        models = Mock()
        client.models = models
        
        # Mock generate_content for LLM
        generate_content = AsyncMock()
        models.generate_content = generate_content
        
        # Mock embed for embeddings
        embed = AsyncMock()
        models.embed = embed
        
        yield {
            'client': client,
            'generate_content': generate_content,
            'embed': embed
        }


@pytest.fixture
def sample_pdf_path(tmp_path):
    """Create a sample PDF file for testing"""
    pdf_path = tmp_path / "sample.pdf"
    # Create a minimal PDF-like file
    pdf_path.write_bytes(b"%PDF-1.4\n%Sample PDF content\n%%EOF")
    return str(pdf_path)


@pytest.fixture
def sample_ocr_text():
    """Sample OCR output text"""
    return """
    Mathematics Paper 1
    Year: 2024
    Level: AS Level
    
    Q1. [5 marks] Calculate the derivative of f(x) = 3x² + 2x - 1
    
    Q2. [10 marks] Solve the following integral:
    ∫(x³ + 2x)dx
    
    Q3. [15 marks] A particle moves along a straight line with velocity 
    v(t) = 2t - 3. Find:
    a) The acceleration at t = 2
    b) The displacement from t = 0 to t = 4
    """


@pytest.fixture
def sample_questions():
    """Sample extracted questions"""
    return {
        "exam_info": {
            "year": "2024",
            "level": "AS Level",
            "subject": "Mathematics",
            "paper": "Paper 1"
        },
        "questions": [
            {
                "question_number": "Q1",
                "marks": 5,
                "question_type": "calculation",
                "topics": ["calculus", "derivatives"],
                "question_text": "Calculate the derivative of f(x) = 3x² + 2x - 1"
            },
            {
                "question_number": "Q2", 
                "marks": 10,
                "question_type": "calculation",
                "topics": ["calculus", "integration"],
                "question_text": "Solve the following integral: ∫(x³ + 2x)dx"
            },
            {
                "question_number": "Q3",
                "marks": 15,
                "question_type": "multi_part",
                "topics": ["mechanics", "kinematics"],
                "question_text": "A particle moves along a straight line with velocity v(t) = 2t - 3. Find:\na) The acceleration at t = 2\nb) The displacement from t = 0 to t = 4"
            }
        ]
    }


@pytest.fixture
def sample_embedding():
    """Sample embedding vector"""
    import numpy as np
    return np.random.rand(768).tolist()


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing"""
    ws = AsyncMock()
    ws.accept = AsyncMock()
    ws.send_json = AsyncMock()
    ws.receive_json = AsyncMock()
    ws.close = AsyncMock()
    return ws


# Environment variable fixtures
@pytest.fixture(autouse=True)
def setup_test_env(monkeypatch):
    """Set up test environment variables"""
    monkeypatch.setenv("MISTRAL_API_KEY", "test-mistral-key")
    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    monkeypatch.setenv("POSTGRES_HOST", "localhost")
    monkeypatch.setenv("POSTGRES_PORT", "5432")
    monkeypatch.setenv("POSTGRES_USER", "testuser")
    monkeypatch.setenv("POSTGRES_PASSWORD", "testpass")
    monkeypatch.setenv("POSTGRES_DB", "test_db")