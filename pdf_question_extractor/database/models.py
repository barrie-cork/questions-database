from sqlalchemy import Column, BigInteger, Integer, String, Text, Boolean, DateTime, ForeignKey, ARRAY
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

Base = declarative_base()

class ExtractedQuestion(Base):
    """Temporary storage for extracted questions pending review"""
    __tablename__ = 'extracted_questions'
    
    id = Column(BigInteger, primary_key=True, index=True)
    question_number = Column(String)
    marks = Column(Integer)
    year = Column(String)
    level = Column(String)
    topics = Column(ARRAY(Text))
    question_type = Column(String)
    question_text = Column(Text, nullable=False)
    source_pdf = Column(String, nullable=False)
    status = Column(String, default='pending')  # pending/approved/rejected
    modified = Column(Boolean, default=False)
    extraction_date = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSONB)

class Question(Base):
    """Permanent storage for approved questions"""
    __tablename__ = 'questions'
    
    id = Column(BigInteger, primary_key=True, index=True)
    question_number = Column(String)
    marks = Column(Integer)
    year = Column(String, index=True)
    level = Column(String)
    topics = Column(ARRAY(Text))
    question_type = Column(String, index=True)
    question_text = Column(Text, nullable=False)
    source_pdf = Column(String, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata = Column(JSONB)

class QuestionEmbedding(Base):
    """Vector embeddings for semantic search"""
    __tablename__ = 'question_embeddings'
    
    id = Column(BigInteger, primary_key=True, index=True)
    question_id = Column(BigInteger, ForeignKey('questions.id', ondelete='CASCADE'), nullable=False)
    embedding = Column(Vector(768))  # 768-dimension vector
    model_name = Column(String(100), nullable=False, default='gemini-embedding-001')
    model_version = Column(String(50), nullable=False, default='1.0')
    embedding_generated_at = Column(DateTime(timezone=True), server_default=func.now())