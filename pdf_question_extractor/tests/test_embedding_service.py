"""
Unit tests for Embedding Service (Gemini)
"""
import pytest
from unittest.mock import Mock, AsyncMock, patch
import httpx
import numpy as np
from google import genai
from services.embedding_service import GeminiEmbeddingService


@pytest.mark.unit
class TestGeminiEmbeddingService:
    """Test Gemini Embedding Service functionality"""
    
    @pytest.fixture
    def embedding_service(self):
        """Create embedding service instance"""
        return GeminiEmbeddingService(api_key="test-api-key")
    
    @pytest.fixture
    def mock_embedding_response(self):
        """Mock successful embedding response"""
        # Create a 768-dimensional embedding
        embedding_vector = np.random.rand(768).tolist()
        
        mock_response = Mock()
        mock_response.embeddings = [Mock(values=embedding_vector)]
        return mock_response
    
    @pytest.fixture
    def mock_batch_embedding_response(self):
        """Mock batch embedding response"""
        # Create multiple embeddings
        embeddings = [
            Mock(values=np.random.rand(768).tolist()),
            Mock(values=np.random.rand(768).tolist()),
            Mock(values=np.random.rand(768).tolist())
        ]
        
        mock_response = Mock()
        mock_response.embeddings = embeddings
        return mock_response
    
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, embedding_service, mock_embedding_response):
        """Test successful single embedding generation"""
        test_text = "This is a test question about calculus."
        
        with patch.object(embedding_service.client.models, 'embed') as mock_embed:
            async def async_return(*args, **kwargs):
                return mock_embedding_response
            
            mock_embed.return_value = async_return()
            
            with patch.object(embedding_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                result = await embedding_service.generate_embedding(test_text)
        
        # Verify result
        assert isinstance(result, list)
        assert len(result) == 768
        assert all(isinstance(x, float) for x in result)
        
        # Verify API call
        mock_embed.assert_called_once()
        call_args = mock_embed.call_args
        assert call_args[1]['model'] == 'models/embedding-001'
        assert call_args[1]['content'] == test_text
    
    @pytest.mark.asyncio
    async def test_generate_embedding_with_task_type(self, embedding_service, mock_embedding_response):
        """Test embedding generation with different task types"""
        test_text = "Search query about mathematics"
        
        with patch.object(embedding_service.client.models, 'embed') as mock_embed:
            async def async_return(*args, **kwargs):
                return mock_embedding_response
            
            mock_embed.return_value = async_return()
            
            with patch.object(embedding_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                # Test with RETRIEVAL_QUERY task type
                result = await embedding_service.generate_embedding(
                    test_text, 
                    task_type="RETRIEVAL_QUERY"
                )
        
        assert len(result) == 768
        
        # Verify task type was passed
        call_args = mock_embed.call_args
        assert call_args[1]['task_type'] == 'RETRIEVAL_QUERY'
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings_success(self, embedding_service, mock_batch_embedding_response):
        """Test successful batch embedding generation"""
        test_texts = [
            "Question 1 about physics",
            "Question 2 about chemistry",
            "Question 3 about biology"
        ]
        
        with patch.object(embedding_service.client.models, 'embed') as mock_embed:
            async def async_return(*args, **kwargs):
                return mock_batch_embedding_response
            
            mock_embed.return_value = async_return()
            
            with patch.object(embedding_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                results = await embedding_service.generate_batch_embeddings(test_texts)
        
        # Verify results
        assert len(results) == 3
        for embedding in results:
            assert isinstance(embedding, list)
            assert len(embedding) == 768
            assert all(isinstance(x, float) for x in embedding)
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings_empty_list(self, embedding_service):
        """Test batch embedding with empty list"""
        results = await embedding_service.generate_batch_embeddings([])
        assert results == []
    
    @pytest.mark.asyncio
    async def test_generate_batch_embeddings_large_batch(self, embedding_service):
        """Test batch embedding with chunking for large batches"""
        # Create 150 texts (should be split into 2 batches)
        large_texts = [f"Question {i}" for i in range(150)]
        
        # Mock responses for two batches
        batch1_response = Mock()
        batch1_response.embeddings = [Mock(values=np.random.rand(768).tolist()) for _ in range(100)]
        
        batch2_response = Mock()
        batch2_response.embeddings = [Mock(values=np.random.rand(768).tolist()) for _ in range(50)]
        
        with patch.object(embedding_service.client.models, 'embed') as mock_embed:
            call_count = 0
            
            async def async_side_effect(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                return batch1_response if call_count == 1 else batch2_response
            
            mock_embed.side_effect = async_side_effect
            
            with patch.object(embedding_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                results = await embedding_service.generate_batch_embeddings(large_texts)
        
        # Verify results
        assert len(results) == 150
        assert mock_embed.call_count == 2  # Should have made 2 API calls
    
    @pytest.mark.asyncio
    async def test_enrich_question_for_embedding(self, embedding_service):
        """Test question enrichment for embedding"""
        question_data = {
            "question_number": "Q1",
            "marks": 10,
            "year": "2024",
            "level": "AS Level",
            "topics": ["calculus", "derivatives"],
            "question_type": "calculation",
            "question_text": "Calculate the derivative of f(x) = 3x²"
        }
        
        enriched = embedding_service.enrich_question_for_embedding(question_data)
        
        # Verify enrichment
        assert "Q1" in enriched
        assert "10 marks" in enriched
        assert "2024" in enriched
        assert "AS Level" in enriched
        assert "calculus" in enriched
        assert "derivatives" in enriched
        assert "calculation" in enriched
        assert "Calculate the derivative" in enriched
    
    @pytest.mark.asyncio
    async def test_calculate_similarity(self, embedding_service):
        """Test similarity calculation between embeddings"""
        # Create two similar embeddings
        embedding1 = [1.0, 0.0, 0.0] + [0.0] * 765  # 768 dimensions
        embedding2 = [0.9, 0.1, 0.0] + [0.0] * 765
        
        similarity = embedding_service.calculate_similarity(embedding1, embedding2)
        
        # Should be close to 0.9 (cosine similarity)
        assert 0.85 < similarity < 0.95
        
        # Test with identical embeddings
        similarity_same = embedding_service.calculate_similarity(embedding1, embedding1)
        assert similarity_same == pytest.approx(1.0, abs=0.001)
        
        # Test with orthogonal embeddings
        embedding3 = [0.0, 1.0, 0.0] + [0.0] * 765
        similarity_orthogonal = embedding_service.calculate_similarity(embedding1, embedding3)
        assert similarity_orthogonal == pytest.approx(0.0, abs=0.001)
    
    @pytest.mark.asyncio
    async def test_retry_on_http_error(self, embedding_service, mock_embedding_response):
        """Test retry mechanism on HTTP errors"""
        # Setup to fail twice then succeed
        side_effects = [
            httpx.HTTPError("Server error"),
            httpx.HTTPError("Server error"),
            mock_embedding_response
        ]
        
        with patch.object(embedding_service.client.models, 'embed') as mock_embed:
            async def async_side_effect(*args, **kwargs):
                effect = side_effects.pop(0)
                if isinstance(effect, Exception):
                    raise effect
                return effect
            
            mock_embed.side_effect = async_side_effect
            
            with patch.object(embedding_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                result = await embedding_service.generate_embedding("Test text")
        
        assert len(result) == 768
        assert mock_embed.call_count == 3
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, embedding_service, mock_embedding_response):
        """Test rate limiting functionality"""
        with patch.object(embedding_service.client.models, 'embed') as mock_embed:
            async def async_return(*args, **kwargs):
                return mock_embedding_response
            
            mock_embed.return_value = async_return()
            
            # Mock rate limiter to track calls
            with patch.object(embedding_service.rate_limiter, 'acquire', new_callable=AsyncMock) as mock_acquire:
                await embedding_service.generate_embedding("Test text")
        
        # Verify rate limiter was called
        mock_acquire.assert_called()
    
    @pytest.mark.asyncio
    async def test_error_handling_no_embeddings(self, embedding_service):
        """Test handling when API returns no embeddings"""
        mock_response = Mock()
        mock_response.embeddings = []  # Empty list
        
        with patch.object(embedding_service.client.models, 'embed') as mock_embed:
            async def async_return(*args, **kwargs):
                return mock_response
            
            mock_embed.return_value = async_return()
            
            with patch.object(embedding_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                with pytest.raises(ValueError, match="No embedding returned"):
                    await embedding_service.generate_embedding("Test text")
    
    @pytest.mark.asyncio
    async def test_embedding_dimension_validation(self, embedding_service):
        """Test that embeddings have correct dimensions"""
        # Mock response with wrong dimension
        wrong_dim_response = Mock()
        wrong_dim_response.embeddings = [Mock(values=np.random.rand(512).tolist())]  # Wrong dimension
        
        with patch.object(embedding_service.client.models, 'embed') as mock_embed:
            async def async_return(*args, **kwargs):
                return wrong_dim_response
            
            mock_embed.return_value = async_return()
            
            with patch.object(embedding_service.rate_limiter, 'acquire', new_callable=AsyncMock):
                # Should log warning but still return the embedding
                result = await embedding_service.generate_embedding("Test text")
        
        # Result should still be returned even with wrong dimension
        assert len(result) == 512  # The actual dimension returned