"""SQLAlchemy models for PDF Question Extractor"""
from sqlalchemy import Column, BigInteger, Integer, String, Text, Boolean, DateTime, ForeignKey, ARRAY, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from datetime import datetime

Base = declarative_base()


class ExtractedQuestion(Base):
    """Temporary storage for extracted questions pending review"""
    __tablename__ = 'extracted_questions'
    
    id = Column(BigInteger, primary_key=True)
    question_number = Column(Text)
    marks = Column(Integer)
    year = Column(Text)
    level = Column(Text)
    topics = Column(ARRAY(Text))  # PostgreSQL array
    question_type = Column(Text)
    question_text = Column(Text, nullable=False)
    source_pdf = Column(Text, nullable=False)
    status = Column(Text, default='pending')  # pending/approved/rejected
    modified = Column(Boolean, default=False)
    extraction_date = Column(DateTime(timezone=True), server_default=func.now())
    metadata = Column(JSON)  # JSONB for flexible data
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'question_number': self.question_number,
            'marks': self.marks,
            'year': self.year,
            'level': self.level,
            'topics': self.topics or [],
            'question_type': self.question_type,
            'question_text': self.question_text,
            'source_pdf': self.source_pdf,
            'status': self.status,
            'modified': self.modified,
            'extraction_date': self.extraction_date.isoformat() if self.extraction_date else None,
            'metadata': self.metadata
        }
    
    def __repr__(self):
        return f"<ExtractedQuestion(id={self.id}, source={self.source_pdf}, status={self.status})>"


class Question(Base):
    """Permanent storage for approved questions"""
    __tablename__ = 'questions'
    
    id = Column(BigInteger, primary_key=True)
    question_number = Column(Text)
    marks = Column(Integer)
    year = Column(Text)
    level = Column(Text)
    topics = Column(ARRAY(Text))  # PostgreSQL array
    question_type = Column(Text)
    question_text = Column(Text, nullable=False)
    source_pdf = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    metadata = Column(JSON)  # JSONB for flexible data
    
    # Relationship to embeddings
    embeddings = relationship("QuestionEmbedding", back_populates="question", cascade="all, delete-orphan")
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'question_number': self.question_number,
            'marks': self.marks,
            'year': self.year,
            'level': self.level,
            'topics': self.topics or [],
            'question_type': self.question_type,
            'question_text': self.question_text,
            'source_pdf': self.source_pdf,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_extracted(cls, extracted_question):
        """Create Question from ExtractedQuestion"""
        return cls(
            question_number=extracted_question.question_number,
            marks=extracted_question.marks,
            year=extracted_question.year,
            level=extracted_question.level,
            topics=extracted_question.topics,
            question_type=extracted_question.question_type,
            question_text=extracted_question.question_text,
            source_pdf=extracted_question.source_pdf,
            metadata=extracted_question.metadata
        )
    
    def __repr__(self):
        return f"<Question(id={self.id}, year={self.year}, type={self.question_type})>"


class QuestionEmbedding(Base):
    """Storage for question embeddings with versioning support"""
    __tablename__ = 'question_embeddings'
    
    id = Column(BigInteger, primary_key=True)
    question_id = Column(BigInteger, ForeignKey('questions.id', ondelete='CASCADE'), nullable=False)
    embedding = Column(Vector(768))  # 768-dimensional Gemini embedding
    model_name = Column(String(100), nullable=False, default='gemini-embedding-001')
    model_version = Column(String(50), nullable=False, default='1.0')
    embedding_generated_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationship to question
    question = relationship("Question", back_populates="embeddings")
    
    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            'id': self.id,
            'question_id': self.question_id,
            'model_name': self.model_name,
            'model_version': self.model_version,
            'embedding_generated_at': self.embedding_generated_at.isoformat() if self.embedding_generated_at else None,
            # Note: embedding vector is typically not included in API responses due to size
        }
    
    def __repr__(self):
        return f"<QuestionEmbedding(id={self.id}, question_id={self.question_id}, model={self.model_name})>"