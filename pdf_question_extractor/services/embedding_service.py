import asyncio
import logging
from typing import List, Dict, Optional
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import httpx
import numpy as np

logger = logging.getLogger(__name__)

# Import shared RateLimiter
from .utils import RateLimiter

class GeminiEmbeddingService:
    """Service for generating vector embeddings using Gemini"""
    
    def __init__(self, api_key: str):
        """Initialize Gemini client for embeddings"""
        self.client = genai.Client(api_key=api_key)
        self.model = "models/embedding-001"  # Note: models/ prefix required
        self.dimension = 768  # Using 768 dimensions as specified in schema
        self.rate_limiter = RateLimiter(calls_per_minute=100)  # Higher limit for embeddings
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type((httpx.HTTPError, httpx.TimeoutError))
    )
    async def generate_embedding(
        self, 
        text: str, 
        task_type: str = "RETRIEVAL_DOCUMENT"
    ) -> List[float]:
        """
        Generate embedding for a single text
        
        Args:
            text: Text to embed
            task_type: Type of task (RETRIEVAL_DOCUMENT, RETRIEVAL_QUERY, etc.)
            
        Returns:
            List of float values representing the embedding
        """
        try:
            # Rate limiting
            await self.rate_limiter.acquire()
            
            logger.debug(f"Generating embedding for text ({len(text)} chars)")
            
            # Configure embedding parameters
            config = types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=self.dimension
            )
            
            # Generate embedding
            response = await self.client.models.embed_content_async(
                model=self.model,
                contents=text,
                config=config
            )
            
            # Extract embedding values
            if response.embeddings and len(response.embeddings) > 0:
                embedding = response.embeddings[0].values
                logger.debug(f"Generated embedding with {len(embedding)} dimensions")
                return embedding
            else:
                raise ValueError("No embedding returned from API")
                
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    async def generate_batch_embeddings(
        self, 
        questions: List[Dict],
        batch_size: int = 10
    ) -> Dict[int, List[float]]:
        """
        Generate embeddings for multiple questions in batches
        
        Args:
            questions: List of question dictionaries with 'id' and question data
            batch_size: Number of embeddings to generate concurrently
            
        Returns:
            Dictionary mapping question ID to embedding
        """
        embeddings = {}
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            logger.info(f"Processing embedding batch {i//batch_size + 1}/{(len(questions) + batch_size - 1)//batch_size}")
            
            # Create tasks for concurrent processing
            tasks = []
            for question in batch:
                # Create rich text representation for better embeddings
                text_to_embed = self._create_embedding_text(question)
                task = self.generate_embedding(text_to_embed)
                tasks.append((question['id'], task))
            
            # Execute batch concurrently
            for q_id, task in tasks:
                try:
                    embedding = await task
                    embeddings[q_id] = embedding
                except Exception as e:
                    logger.error(f"Failed to generate embedding for question {q_id}: {str(e)}")
                    # Continue with other embeddings even if one fails
        
        logger.info(f"Generated {len(embeddings)} embeddings successfully")
        return embeddings
    
    def _create_embedding_text(self, question: Dict) -> str:
        """
        Create enriched text representation for embedding
        
        Args:
            question: Question dictionary
            
        Returns:
            Enriched text for embedding
        """
        # Combine multiple fields for richer semantic representation
        parts = []
        
        # Add question text (most important)
        parts.append(f"Question: {question.get('question_text', '')}")
        
        # Add metadata for context
        if question.get('question_type'):
            parts.append(f"Type: {question['question_type']}")
        
        if question.get('topics'):
            topics = question['topics'] if isinstance(question['topics'], list) else [question['topics']]
            parts.append(f"Topics: {', '.join(topics)}")
        
        if question.get('level'):
            parts.append(f"Level: {question['level']}")
        
        if question.get('marks'):
            parts.append(f"Marks: {question['marks']}")
        
        if question.get('year'):
            parts.append(f"Year: {question['year']}")
        
        # Join all parts
        return "\n".join(parts)
    
    async def search_embedding(
        self, 
        query: str, 
        task_type: str = "RETRIEVAL_QUERY"
    ) -> List[float]:
        """
        Generate embedding for a search query
        
        Args:
            query: Search query text
            task_type: Should be RETRIEVAL_QUERY for search
            
        Returns:
            List of float values representing the query embedding
        """
        # Use RETRIEVAL_QUERY for search queries (different from document embeddings)
        return await self.generate_embedding(query, task_type)
    
    def calculate_similarity(
        self, 
        embedding1: List[float], 
        embedding2: List[float]
    ) -> float:
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        # Convert to numpy arrays
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return float(similarity)