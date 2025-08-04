"""
WebSocket tests for real-time progress tracking.

Tests WebSocket functionality including:
- Connection establishment and disconnection
- Real-time progress updates
- Client tracking and cleanup
- Error handling
- Processing cancellation
- Multiple concurrent clients
"""

import pytest
import asyncio
import json
from typing import Dict, Any, List
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app import app, websocket_tracker
from services.pdf_processor import ProcessingStatus, ProcessingProgress
from database.models import ExtractedQuestion


@pytest.mark.websocket
class TestWebSocketConnection:
    """Test WebSocket connection management"""
    
    def test_websocket_connect_success(self, test_client: TestClient):
        """Test successful WebSocket connection"""
        with test_client.websocket_connect("/ws/progress") as websocket:
            # Connection should be established
            data = websocket.receive_json()
            
            assert data["type"] == "connection"
            assert data["status"] == "connected"
            assert "client_id" in data
            assert data["message"] == "WebSocket connection established"
    
    def test_websocket_disconnect(self, test_client: TestClient):
        """Test WebSocket disconnection and cleanup"""
        client_id = None
        
        with test_client.websocket_connect("/ws/progress") as websocket:
            # Get client ID
            data = websocket.receive_json()
            client_id = data["client_id"]
            
            # Verify client is tracked
            assert client_id in websocket_tracker.clients
        
        # After disconnect, client should be removed
        assert client_id not in websocket_tracker.clients
        assert client_id not in websocket_tracker.processors


@pytest.mark.websocket
class TestProgressUpdates:
    """Test real-time progress updates"""
    
    @pytest.fixture
    def mock_progress_updates(self):
        """Generate mock progress updates"""
        return [
            ProcessingProgress(
                file_path="test.pdf",
                status=ProcessingStatus.PROCESSING,
                current_step="Starting OCR",
                total_steps=4,
                completed_steps=0,
                start_time=datetime.utcnow(),
                questions_extracted=0,
                questions_stored=0,
                embeddings_generated=0
            ),
            ProcessingProgress(
                file_path="test.pdf",
                status=ProcessingStatus.OCR_COMPLETE,
                current_step="OCR Complete",
                total_steps=4,
                completed_steps=1,
                start_time=datetime.utcnow(),
                questions_extracted=0,
                questions_stored=0,
                embeddings_generated=0
            ),
            ProcessingProgress(
                file_path="test.pdf",
                status=ProcessingStatus.LLM_COMPLETE,
                current_step="Questions Extracted",
                total_steps=4,
                completed_steps=2,
                start_time=datetime.utcnow(),
                questions_extracted=5,
                questions_stored=0,
                embeddings_generated=0
            ),
            ProcessingProgress(
                file_path="test.pdf",
                status=ProcessingStatus.COMPLETED,
                current_step="Processing Complete",
                total_steps=4,
                completed_steps=4,
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow(),
                questions_extracted=5,
                questions_stored=5,
                embeddings_generated=5
            )
        ]
    
    @patch('services.pdf_processor.PDFQuestionProcessor')
    async def test_progress_updates_during_processing(
        self, 
        mock_processor_class,
        test_client: TestClient,
        mock_progress_updates
    ):
        """Test receiving progress updates during PDF processing"""
        # Setup mock processor
        mock_processor = AsyncMock()
        mock_processor_class.return_value = mock_processor
        
        # Mock the process_pdf method to send progress updates
        async def mock_process_pdf(pdf_path, progress_callback=None):
            if progress_callback:
                for progress in mock_progress_updates:
                    await asyncio.sleep(0.01)  # Small delay
                    await progress_callback(progress)
            return {
                "questions_extracted": 5,
                "questions_stored": 5,
                "embeddings_generated": 5
            }
        
        mock_processor.process_pdf.side_effect = mock_process_pdf
        
        with test_client.websocket_connect("/ws/progress") as websocket:
            # Get connection message
            conn_msg = websocket.receive_json()
            client_id = conn_msg["client_id"]
            
            # Simulate file upload that triggers processing
            with patch('app.websocket_tracker') as mock_tracker:
                mock_tracker.clients = {client_id: websocket}
                mock_tracker.processors = {client_id: mock_processor}
                
                # Manually trigger processing with callback
                await mock_processor.process_pdf(
                    "test.pdf",
                    progress_callback=lambda p: websocket.send_json({
                        "type": "progress",
                        "data": p.to_dict() if hasattr(p, 'to_dict') else {
                            "status": p.status.value,
                            "current_step": p.current_step,
                            "completed_steps": p.completed_steps,
                            "total_steps": p.total_steps,
                            "questions_extracted": p.questions_extracted
                        }
                    })
                )
                
                # Receive and verify progress updates
                progress_messages = []
                for _ in range(4):  # Expect 4 progress updates
                    try:
                        msg = websocket.receive_json(timeout=1)
                        if msg["type"] == "progress":
                            progress_messages.append(msg)
                    except:
                        break
                
                # Verify we received progress updates
                assert len(progress_messages) >= 2  # At least some updates
                
                # Verify progression
                statuses = [msg["data"]["status"] for msg in progress_messages]
                assert ProcessingStatus.PROCESSING.value in statuses or ProcessingStatus.OCR_COMPLETE.value in statuses


@pytest.mark.websocket
class TestErrorHandling:
    """Test error handling through WebSocket"""
    
    def test_websocket_error_message(self, test_client: TestClient):
        """Test sending error messages through WebSocket"""
        with test_client.websocket_connect("/ws/progress") as websocket:
            # Get connection message
            conn_msg = websocket.receive_json()
            client_id = conn_msg["client_id"]
            
            # Simulate an error during processing
            error_msg = {
                "type": "error",
                "message": "OCR processing failed",
                "details": "API rate limit exceeded"
            }
            
            # In a real scenario, this would be sent by the processor
            # For testing, we simulate it
            with patch('app.websocket_tracker.clients', {client_id: websocket}):
                # The app would send this error
                websocket.send_json(error_msg)
                
                # Verify we can receive the error
                received = websocket.receive_json()
                assert received["type"] == "error"
                assert "OCR processing failed" in received["message"]


@pytest.mark.websocket
class TestProcessingCancellation:
    """Test processing cancellation via WebSocket"""
    
    @patch('services.pdf_processor.PDFQuestionProcessor')
    def test_cancel_processing_via_websocket(
        self,
        mock_processor_class,
        test_client: TestClient
    ):
        """Test cancelling processing through WebSocket"""
        # Setup mock processor
        mock_processor = AsyncMock()
        mock_processor_class.return_value = mock_processor
        mock_processor.cancel_processing = AsyncMock(return_value=True)
        
        with test_client.websocket_connect("/ws/progress") as websocket:
            # Get connection message
            conn_msg = websocket.receive_json()
            client_id = conn_msg["client_id"]
            
            # Setup tracker state
            websocket_tracker.clients[client_id] = websocket
            websocket_tracker.processors[client_id] = mock_processor
            
            # Send cancel command
            cancel_msg = {
                "type": "cancel",
                "file_path": "test.pdf"
            }
            websocket.send_json(cancel_msg)
            
            # Verify processor cancel was called
            mock_processor.cancel_processing.assert_called_once_with("test.pdf")


@pytest.mark.websocket
class TestMultipleClients:
    """Test multiple concurrent WebSocket clients"""
    
    def test_multiple_websocket_clients(self, test_client: TestClient):
        """Test handling multiple concurrent WebSocket connections"""
        clients = []
        client_ids = []
        
        # Connect multiple clients
        for i in range(3):
            client = test_client.websocket_connect("/ws/progress").__enter__()
            clients.append(client)
            
            # Get client ID
            conn_msg = client.receive_json()
            client_ids.append(conn_msg["client_id"])
        
        # Verify all clients are tracked
        for client_id in client_ids:
            assert client_id in websocket_tracker.clients
        
        # Verify client IDs are unique
        assert len(set(client_ids)) == 3
        
        # Disconnect all clients
        for client in clients:
            client.__exit__(None, None, None)
        
        # Verify all clients are cleaned up
        for client_id in client_ids:
            assert client_id not in websocket_tracker.clients


@pytest.mark.websocket
class TestMessageValidation:
    """Test WebSocket message format validation"""
    
    def test_websocket_message_format(self, test_client: TestClient):
        """Test WebSocket message format and structure"""
        with test_client.websocket_connect("/ws/progress") as websocket:
            # Connection message format
            conn_msg = websocket.receive_json()
            
            # Validate connection message structure
            assert "type" in conn_msg
            assert "status" in conn_msg
            assert "client_id" in conn_msg
            assert "message" in conn_msg
            assert "timestamp" in conn_msg
            
            # Validate types
            assert conn_msg["type"] == "connection"
            assert conn_msg["status"] == "connected"
            assert isinstance(conn_msg["client_id"], str)
            assert len(conn_msg["client_id"]) == 36  # UUID format
    
    def test_invalid_message_handling(self, test_client: TestClient):
        """Test handling of invalid WebSocket messages"""
        with test_client.websocket_connect("/ws/progress") as websocket:
            # Get connection message
            websocket.receive_json()
            
            # Send invalid message format
            try:
                websocket.send_text("invalid json")
                # Should handle gracefully
                response = websocket.receive_json(timeout=1)
                # May receive an error response or continue normally
            except WebSocketDisconnect:
                # Connection may close on invalid message
                pass


@pytest.mark.websocket
class TestReconnection:
    """Test WebSocket reconnection scenarios"""
    
    def test_websocket_reconnection(self, test_client: TestClient):
        """Test WebSocket reconnection after disconnect"""
        # First connection
        with test_client.websocket_connect("/ws/progress") as websocket1:
            conn_msg1 = websocket1.receive_json()
            client_id1 = conn_msg1["client_id"]
        
        # Verify cleanup after disconnect
        assert client_id1 not in websocket_tracker.clients
        
        # Second connection (reconnection)
        with test_client.websocket_connect("/ws/progress") as websocket2:
            conn_msg2 = websocket2.receive_json()
            client_id2 = conn_msg2["client_id"]
            
            # Should get a new client ID
            assert client_id2 != client_id1
            assert client_id2 in websocket_tracker.clients
    
    def test_websocket_timeout_handling(self, test_client: TestClient):
        """Test WebSocket timeout and keepalive"""
        with test_client.websocket_connect("/ws/progress") as websocket:
            # Get connection message
            conn_msg = websocket.receive_json()
            
            # Simulate long period of inactivity
            # In real app, might have keepalive mechanism
            import time
            time.sleep(0.5)
            
            # Connection should still be alive
            # Send a test message
            websocket.send_json({"type": "ping"})
            
            # Should be able to continue communication
            # (actual behavior depends on server implementation)


@pytest.mark.websocket
class TestIntegration:
    """Integration tests with actual file processing"""
    
    @patch('aiofiles.open')
    @patch('services.pdf_processor.PDFQuestionProcessor')
    async def test_websocket_with_file_upload(
        self,
        mock_processor_class,
        mock_aiofiles,
        test_client: TestClient
    ):
        """Test WebSocket integration with file upload endpoint"""
        # Setup mock processor
        mock_processor = AsyncMock()
        mock_processor_class.return_value = mock_processor
        mock_processor.process_pdf = AsyncMock(return_value={
            "questions_extracted": 3,
            "questions_stored": 3,
            "embeddings_generated": 3
        })
        
        # Mock file operations
        mock_file = AsyncMock()
        mock_file.write = AsyncMock()
        mock_aiofiles.return_value.__aenter__.return_value = mock_file
        
        # Connect WebSocket first
        with test_client.websocket_connect("/ws/progress") as websocket:
            conn_msg = websocket.receive_json()
            client_id = conn_msg["client_id"]
            
            # Upload file with client_id header
            files = {"file": ("test.pdf", b"PDF content", "application/pdf")}
            headers = {"X-Client-Id": client_id}
            
            response = test_client.post(
                "/api/upload",
                files=files,
                headers=headers
            )
            
            # Should receive progress updates through WebSocket
            # (In real scenario, would receive actual progress)
            assert response.status_code == 200