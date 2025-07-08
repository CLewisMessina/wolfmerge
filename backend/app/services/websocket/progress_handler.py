# app/services/websocket/progress_handler.py
import time
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
import structlog

from .manager import websocket_manager

logger = structlog.get_logger()

@dataclass
class ProgressMessage:
    """Structured progress message for WebSocket transmission"""
    message_type: str
    workspace_id: str
    payload: Dict[str, Any]
    priority: str = "normal"  # low, normal, high
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.message_type,
            "priority": self.priority,
            "payload": self.payload,
            "timestamp": time.time()
        }

class ProgressHandler:
    """Handle progress updates and WebSocket message routing"""
    
    def __init__(self):
        self.message_queue = []
        self.rate_limits = {
            "progress_update": 0.5,    # Max one progress update per 0.5 seconds
            "batch_status": 1.0,       # Max one batch status per second
            "error_notification": 0.1, # Max one error per 0.1 seconds (immediate)
            "performance_update": 5.0   # Max one performance update per 5 seconds
        }
        self.last_sent = {}
    
    async def handle_progress_update(self, workspace_id: str, progress_data: Dict[str, Any]):
        """Handle individual job progress update"""
        
        # Rate limit check
        if not self._check_rate_limit("progress_update", workspace_id):
            return
        
        message = ProgressMessage(
            message_type="job_progress",
            workspace_id=workspace_id,
            payload=progress_data,
            priority="normal"
        )
        
        await self._send_message(workspace_id, message)
    
    async def handle_batch_started(self, workspace_id: str, batch_data: Dict[str, Any]):
        """Handle batch processing start notification"""
        
        message = ProgressMessage(
            message_type="batch_started",
            workspace_id=workspace_id,
            payload=batch_data,
            priority="high"
        )
        
        await self._send_message(workspace_id, message)
    
    async def handle_batch_completed(self, workspace_id: str, completion_data: Dict[str, Any]):
        """Handle batch processing completion"""
        
        message = ProgressMessage(
            message_type="batch_completed",
            workspace_id=workspace_id,
            payload=completion_data,
            priority="high"
        )
        
        await self._send_message(workspace_id, message)
    
    async def handle_ui_context_update(self, workspace_id: str, ui_context: Dict[str, Any]):
        """Handle UI context intelligence update"""
        
        message = ProgressMessage(
            message_type="ui_context_update",
            workspace_id=workspace_id,
            payload=ui_context,
            priority="medium"
        )
        
        await self._send_message(workspace_id, message)
    
    async def handle_performance_update(self, workspace_id: str, performance_data: Dict[str, Any]):
        """Handle performance metrics update"""
        
        # Rate limit check
        if not self._check_rate_limit("performance_update", workspace_id):
            return
        
        message = ProgressMessage(
            message_type="performance_update",
            workspace_id=workspace_id,
            payload=performance_data,
            priority="low"
        )
        
        await self._send_message(workspace_id, message)
    
    async def handle_error_notification(self, workspace_id: str, error_data: Dict[str, Any]):
        """Handle error notification (immediate delivery)"""
        
        message = ProgressMessage(
            message_type="error_notification",
            workspace_id=workspace_id,
            payload=error_data,
            priority="high"
        )
        
        await self._send_message(workspace_id, message)
    
    def create_progress_callback(self, workspace_id: str) -> Callable:
        """Create progress callback function for batch processor"""
        
        async def progress_callback(progress_data: Dict[str, Any]):
            await self.handle_progress_update(workspace_id, progress_data)
        
        return progress_callback
    
    def _check_rate_limit(self, message_type: str, workspace_id: str) -> bool:
        """Check if message can be sent based on rate limits"""
        
        current_time = time.time()
        rate_limit = self.rate_limits.get(message_type, 1.0)
        key = f"{workspace_id}:{message_type}"
        
        last_sent_time = self.last_sent.get(key, 0)
        
        if current_time - last_sent_time >= rate_limit:
            self.last_sent[key] = current_time
            return True
        
        return False
    
    async def _send_message(self, workspace_id: str, message: ProgressMessage):
        """Send message via WebSocket manager"""
        
        try:
            sent_count = await websocket_manager.broadcast_to_workspace(
                workspace_id, message.to_dict()
            )
            
            if sent_count > 0:
                logger.debug(
                    "Progress message sent",
                    workspace_id=workspace_id,
                    message_type=message.message_type,
                    connections_reached=sent_count
                )
            
        except Exception as e:
            logger.error(
                "Failed to send progress message",
                workspace_id=workspace_id,
                message_type=message.message_type,
                error=str(e)
            )

# Global progress handler instance
progress_handler = ProgressHandler()