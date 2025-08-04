"""
Vector operations for question embeddings and similarity search
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from sqlalchemy import text, select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from database.models import Question, QuestionEmbedding
from database.session import AsyncSessionLocal
from services.embedding_service import EmbeddingService
from config import Config

logger = logging.getLogger(__name__)

class VectorOperations:
    """Handle vector operations for question embeddings"""
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
        self.embedding_dimension = Config.EMBEDDING_DIMENSION
        self.batch_size = Config.EMBEDDING_BATCH_SIZE
    
    async def store_question_embedding(
        self, 
        session: AsyncSession, 
        question_id: int, 
        text: str,
        model_name: str = 'gemini-embedding-001',
        model_version: str = '1.0'
    ) -> bool:
        """
        Generate and store embedding for a question
        
        Args:
            session: Database session
            question_id: ID of the question
            text: Question text to embed
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            
        Returns:
            bool: Success status
        """
        try:
            # Generate embedding
            embedding_vector = await self.embedding_service.get_embedding(text)
            
            if embedding_vector is None:
                logger.error(f"Failed to generate embedding for question {question_id}")
                return False
            
            # Check if embedding already exists for this question and model
            existing = await session.execute(
                select(QuestionEmbedding).where(
                    and_(
                        QuestionEmbedding.question_id == question_id,
                        QuestionEmbedding.model_name == model_name,
                        QuestionEmbedding.model_version == model_version
                    )
                )
            )
            existing_embedding = existing.scalar_one_or_none()
            
            if existing_embedding:
                # Update existing embedding
                existing_embedding.embedding = embedding_vector
                logger.info(f"Updated embedding for question {question_id}")
            else:
                # Create new embedding
                new_embedding = QuestionEmbedding(
                    question_id=question_id,
                    embedding=embedding_vector,
                    model_name=model_name,
                    model_version=model_version
                )
                session.add(new_embedding)
                logger.info(f"Created new embedding for question {question_id}")
            
            await session.commit()
            return True
            
        except Exception as e:
            logger.error(f"Error storing embedding for question {question_id}: {str(e)}")
            await session.rollback()
            return False
    
    async def batch_store_embeddings(
        self,
        question_texts: List[Tuple[int, str]],
        model_name: str = 'gemini-embedding-001',
        model_version: str = '1.0'
    ) -> Dict[str, int]:
        """
        Store embeddings for multiple questions in batches
        
        Args:
            question_texts: List of (question_id, text) tuples
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            
        Returns:
            Dict with success/failure counts
        """
        results = {"success": 0, "failed": 0}
        
        # Process in batches to avoid overwhelming the embedding service
        for i in range(0, len(question_texts), self.batch_size):
            batch = question_texts[i:i + self.batch_size]
            
            async with AsyncSessionLocal() as session:
                for question_id, text in batch:
                    success = await self.store_question_embedding(
                        session, question_id, text, model_name, model_version
                    )
                    if success:
                        results["success"] += 1
                    else:
                        results["failed"] += 1
                    
                    # Small delay to respect rate limits
                    await asyncio.sleep(0.1)
        
        logger.info(f"Batch embedding completed: {results}")
        return results
    
    async def similarity_search(
        self,
        session: AsyncSession,
        query_text: str,
        limit: int = 10,
        similarity_threshold: float = 0.7,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic similarity search on questions
        
        Args:
            session: Database session
            query_text: Text to search for
            limit: Maximum number of results
            similarity_threshold: Minimum cosine similarity score
            filters: Optional filters for year, level, topics, etc.
            
        Returns:
            List of matching questions with similarity scores
        """
        try:
            # Generate embedding for query
            query_embedding = await self.embedding_service.get_embedding(query_text)
            
            if query_embedding is None:
                logger.error("Failed to generate embedding for query")
                return []
            
            # Build base query with similarity search
            query = """
                SELECT 
                    q.id,
                    q.question_number,
                    q.marks,
                    q.year,
                    q.level,
                    q.topics,
                    q.question_type,
                    q.question_text,
                    q.source_pdf,
                    q.created_at,
                    q.metadata,
                    (qe.embedding <=> :query_embedding) as similarity_score
                FROM questions q
                JOIN question_embeddings qe ON q.id = qe.question_id
                WHERE (qe.embedding <=> :query_embedding) < :distance_threshold
            """
            
            params = {
                'query_embedding': str(query_embedding),
                'distance_threshold': 1 - similarity_threshold  # Convert similarity to distance
            }
            
            # Add filters if provided
            if filters:
                if 'year' in filters:
                    query += " AND q.year = :year"
                    params['year'] = filters['year']
                
                if 'level' in filters:
                    query += " AND q.level = :level"
                    params['level'] = filters['level']
                
                if 'question_type' in filters:
                    query += " AND q.question_type = :question_type"
                    params['question_type'] = filters['question_type']
                
                if 'topics' in filters and filters['topics']:
                    query += " AND q.topics && :topics"
                    params['topics'] = filters['topics']
                
                if 'min_marks' in filters:
                    query += " AND q.marks >= :min_marks"
                    params['min_marks'] = filters['min_marks']
                
                if 'max_marks' in filters:
                    query += " AND q.marks <= :max_marks"
                    params['max_marks'] = filters['max_marks']
            
            # Add ordering and limit
            query += " ORDER BY similarity_score ASC LIMIT :limit"
            params['limit'] = limit
            
            # Execute query
            result = await session.execute(text(query), params)
            rows = result.fetchall()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                results.append({
                    'id': row.id,
                    'question_number': row.question_number,
                    'marks': row.marks,
                    'year': row.year,
                    'level': row.level,
                    'topics': row.topics,
                    'question_type': row.question_type,
                    'question_text': row.question_text,
                    'source_pdf': row.source_pdf,
                    'created_at': row.created_at,
                    'metadata': row.metadata,
                    'similarity_score': float(1 - row.similarity_score)  # Convert back to similarity
                })
            
            logger.info(f"Found {len(results)} similar questions for query")
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []
    
    async def find_duplicate_questions(
        self,
        session: AsyncSession,
        similarity_threshold: float = 0.95,
        batch_size: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Find potential duplicate questions based on vector similarity
        
        Args:
            session: Database session
            similarity_threshold: Minimum similarity score to consider duplicates
            batch_size: Number of questions to process at once
            
        Returns:
            List of potential duplicate pairs
        """
        try:
            duplicates = []
            distance_threshold = 1 - similarity_threshold
            
            # Query to find similar question pairs
            query = """
                SELECT 
                    q1.id as id1,
                    q1.question_text as text1,
                    q1.source_pdf as source1,
                    q2.id as id2,
                    q2.question_text as text2,
                    q2.source_pdf as source2,
                    (qe1.embedding <=> qe2.embedding) as distance
                FROM questions q1
                JOIN question_embeddings qe1 ON q1.id = qe1.question_id
                JOIN question_embeddings qe2 ON qe1.question_id < qe2.question_id
                JOIN questions q2 ON q2.id = qe2.question_id
                WHERE (qe1.embedding <=> qe2.embedding) < :distance_threshold
                ORDER BY distance ASC
                LIMIT :batch_size
            """
            
            result = await session.execute(
                text(query), 
                {
                    'distance_threshold': distance_threshold,
                    'batch_size': batch_size
                }
            )
            rows = result.fetchall()
            
            for row in rows:
                duplicates.append({
                    'question1_id': row.id1,
                    'question1_text': row.text1,
                    'question1_source': row.source1,
                    'question2_id': row.id2,
                    'question2_text': row.text2,
                    'question2_source': row.source2,
                    'similarity_score': float(1 - row.distance)
                })
            
            logger.info(f"Found {len(duplicates)} potential duplicate pairs")
            return duplicates
            
        except Exception as e:
            logger.error(f"Error finding duplicates: {str(e)}")
            return []
    
    async def get_embedding_stats(self, session: AsyncSession) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings
        
        Args:
            session: Database session
            
        Returns:
            Dictionary with embedding statistics
        """
        try:
            # Count total embeddings
            total_result = await session.execute(
                text("SELECT COUNT(*) FROM question_embeddings")
            )
            total_embeddings = total_result.scalar()
            
            # Count questions without embeddings
            missing_result = await session.execute(
                text("""
                    SELECT COUNT(*) 
                    FROM questions q 
                    LEFT JOIN question_embeddings qe ON q.id = qe.question_id 
                    WHERE qe.id IS NULL
                """)
            )
            missing_embeddings = missing_result.scalar()
            
            # Get model distribution
            model_result = await session.execute(
                text("""
                    SELECT model_name, model_version, COUNT(*) as count
                    FROM question_embeddings 
                    GROUP BY model_name, model_version
                    ORDER BY count DESC
                """)
            )
            model_stats = [
                {
                    'model_name': row.model_name,
                    'model_version': row.model_version,
                    'count': row.count
                }
                for row in model_result.fetchall()
            ]
            
            return {
                'total_embeddings': total_embeddings,
                'missing_embeddings': missing_embeddings,
                'embedding_coverage': (total_embeddings / (total_embeddings + missing_embeddings)) * 100 if (total_embeddings + missing_embeddings) > 0 else 0,
                'model_distribution': model_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting embedding stats: {str(e)}")
            return {}
    
    async def rebuild_embeddings(
        self,
        model_name: str = 'gemini-embedding-001',
        model_version: str = '1.0',
        force: bool = False
    ) -> Dict[str, int]:
        """
        Rebuild embeddings for all questions
        
        Args:
            model_name: Name of the embedding model
            model_version: Version of the embedding model
            force: Whether to overwrite existing embeddings
            
        Returns:
            Dictionary with processing results
        """
        results = {"processed": 0, "success": 0, "failed": 0, "skipped": 0}
        
        async with AsyncSessionLocal() as session:
            try:
                # Get all questions
                if force:
                    # Get all questions regardless of existing embeddings
                    query = select(Question.id, Question.question_text)
                else:
                    # Only get questions without embeddings for this model
                    query = select(Question.id, Question.question_text).where(
                        ~Question.id.in_(
                            select(QuestionEmbedding.question_id).where(
                                and_(
                                    QuestionEmbedding.model_name == model_name,
                                    QuestionEmbedding.model_version == model_version
                                )
                            )
                        )
                    )
                
                result = await session.execute(query)
                questions = result.fetchall()
                
                logger.info(f"Rebuilding embeddings for {len(questions)} questions")
                
                # Process questions in batches
                for i in range(0, len(questions), self.batch_size):
                    batch = questions[i:i + self.batch_size]
                    
                    for question_id, question_text in batch:
                        results["processed"] += 1
                        
                        if not force:
                            # Check if embedding already exists
                            existing = await session.execute(
                                select(QuestionEmbedding).where(
                                    and_(
                                        QuestionEmbedding.question_id == question_id,
                                        QuestionEmbedding.model_name == model_name,
                                        QuestionEmbedding.model_version == model_version
                                    )
                                )
                            )
                            if existing.scalar_one_or_none():
                                results["skipped"] += 1
                                continue
                        
                        # Generate and store embedding
                        success = await self.store_question_embedding(
                            session, question_id, question_text, model_name, model_version
                        )
                        
                        if success:
                            results["success"] += 1
                        else:
                            results["failed"] += 1
                        
                        # Rate limiting
                        await asyncio.sleep(0.1)
                
                logger.info(f"Embedding rebuild completed: {results}")
                return results
                
            except Exception as e:
                logger.error(f"Error rebuilding embeddings: {str(e)}")
                results["failed"] = results["processed"] - results["success"]
                return results

# Create global instance
vector_ops = VectorOperations()

# Convenience functions for common operations
async def search_similar_questions(
    query_text: str,
    limit: int = 10,
    similarity_threshold: float = 0.7,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """Convenience function for similarity search"""
    async with AsyncSessionLocal() as session:
        return await vector_ops.similarity_search(
            session, query_text, limit, similarity_threshold, filters
        )

async def store_question_embedding(
    question_id: int,
    text: str,
    model_name: str = 'gemini-embedding-001',
    model_version: str = '1.0'
) -> bool:
    """Convenience function for storing single embedding"""
    async with AsyncSessionLocal() as session:
        return await vector_ops.store_question_embedding(
            session, question_id, text, model_name, model_version
        )

async def get_embedding_statistics() -> Dict[str, Any]:
    """Convenience function for getting embedding stats"""
    async with AsyncSessionLocal() as session:
        return await vector_ops.get_embedding_stats(session)