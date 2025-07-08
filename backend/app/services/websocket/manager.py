# app/services/websocket/manager.py
import asyncio
import json
import time
from typing import Dict, List, Optional, Any
from fastapi import WebSocket, WebSocketDisconnect
import structlog

logger = structlog.get_logger()

class ConnectionManager:
    """Manage individual WebSocket connection"""
    
    def __init__(self, websocket: WebSocket, workspace_id: str, connection_id: str):
        self.websocket = websocket
        self.workspace_id = workspace_id
        self.connection_id = connection_id
        self.connected_at = time.time()
        self.last_activity = time.time()
        self.is_active = True
        
        # Message tracking
        self.messages_sent = 0
        self.messages_failed = 0
    
    async def send_message(self, message: Dict[str, Any]) -> bool:
        """Send message to client with error handling"""
        
        if not self.is_active:
            return False
        
        try:
            # Add metadata to message
            message_with_meta = {
                **message,
                "workspace_id": self.workspace_id,
                "connection_id": self.connection_id,
                "timestamp": time.time()
            }
            
            await self.websocket.send_text(json.dumps(message_with_meta))
            self.messages_sent += 1
            self.last_activity = time.time()
            
            logger.debug(
                "WebSocket message sent",
                workspace_id=self.workspace_id,
                connection_id=self.connection_id,
                message_type=message.get("type", "unknown")
            )
            
            return True
            
        except Exception as e:
            self.messages_failed += 1
            self.is_active = False
            
            logger.warning(
                "Failed to send WebSocket message",
                workspace_id=self.workspace_id,
                connection_id=self.connection_id,
                error=str(e)
            )
            
            return False
    
    async def send_ping(self) -> bool:
        """Send ping to keep connection alive"""
        
        return await self.send_message({
            "type": "ping",
            "message": "Connection health check"
        })
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        
        return {
            "connection_id": self.connection_id,
            "workspace_id": self.workspace_id,
            "connected_at": self.connected_at,
            "last_activity": self.last_activity,
            "connection_duration": time.time() - self.connected_at,
            "is_active": self.is_active,
            "messages_sent": self.messages_sent,
            "messages_failed": self.messages_failed,
            "success_rate": self.messages_sent / (self.messages_sent + self.messages_failed) if (self.messages_sent + self.messages_failed) > 0 else 1.0
        }

class WebSocketManager:
    """Centralized WebSocket connection management"""
    
    def __init__(self):
        # Active connections by workspace
        self.connections: Dict[str, List[ConnectionManager]] = {}
        
        # Connection tracking
        self.total_connections = 0
        self.active_connections = 0
        
        # Background tasks
        self.background_tasks = set()
        self.cleanup_interval = 30  # seconds
        self.ping_interval = 60     # seconds
        
        # Start background maintenance
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks"""
        
        # Cleanup task
        cleanup_task = asyncio.create_task(self._cleanup_connections())
        self.background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self.background_tasks.discard)
        
        # Ping task
        ping_task = asyncio.create_task(self._ping_connections())
        self.background_tasks.add(ping_task)
        ping_task.add_done_callback(self.background_tasks.discard)
    
    async def connect(self, websocket: WebSocket, workspace_id: str) -> str:
        """Accept new WebSocket connection"""
        
        await websocket.accept()
        
        # Generate connection ID
        connection_id = f"{workspace_id}_{int(time.time())}_{len(self.connections.get(workspace_id, []))}"
        
        # Create connection manager
        connection = ConnectionManager(websocket, workspace_id, connection_id)
        
        # Add to tracking
        if workspace_id not in self.connections:
            self.connections[workspace_id] = []
        
        self.connections[workspace_id].append(connection)
        self.total_connections += 1
        self.active_connections += 1
        
        # Send welcome message
        await connection.send_message({
            "type": "connection_established",
            "message": "Real-time progress tracking enabled",
            "connection_id": connection_id,
            "features": [
                "real_time_progress",
                "batch_status",
                "performance_metrics",
                "error_notifications"
            ]
        })
        
        logger.info(
            "WebSocket connection established",
            workspace_id=workspace_id,
            connection_id=connection_id,
            total_connections=self.active_connections
        )
        
        return connection_id
    
    def disconnect(self, workspace_id: str, connection_id: str):
        """Handle WebSocket disconnection"""
        
        if workspace_id not in self.connections:
            return
        
        # Find and remove connection
        connections = self.connections[workspace_id]
        for i, conn in enumerate(connections):
            if conn.connection_id == connection_id:
                connections.pop(i)
                self.active_connections -= 1
                
                logger.info(
                    "WebSocket connection closed",
                    workspace_id=workspace_id,
                    connection_id=connection_id,
                    connection_duration=time.time() - conn.connected_at,
                    messages_sent=conn.messages_sent
                )
                break
        
        # Clean up empty workspace
        if not connections:
            del self.connections[workspace_id]
    
    async def broadcast_to_workspace(self, workspace_id: str, message: Dict[str, Any]) -> int:
        """Broadcast message to all connections in workspace"""
        
        if workspace_id not in self.connections:
            return 0
        
        connections = self.connections[workspace_id]
        successful_sends = 0
        failed_connections = []
        
        # Send to all active connections
        for connection in connections:
            if connection.is_active:
                success = await connection.send_message(message)
                if success:
                    successful_sends += 1
                else:
                    failed_connections.append(connection)
        
        # Remove failed connections
        for failed_conn in failed_connections:
            self.disconnect(workspace_id, failed_conn.connection_id)
        
        if successful_sends > 0:
            logger.debug(
                "Message broadcast to workspace",
                workspace_id=workspace_id,
                successful_sends=successful_sends,
                failed_sends=len(failed_connections),
                message_type=message.get("type", "unknown")
            )
        
        return successful_sends
    
    async def send_to_connection(self, workspace_id: str, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to specific connection"""
        
        if workspace_id not in self.connections:
            return False
        
        for connection in self.connections[workspace_id]:
            if connection.connection_id == connection_id and connection.is_active:
                return await connection.send_message(message)
        
        return False
    
    async def _cleanup_connections(self):
        """Background task to clean up stale connections"""
        
        while True:
            try:
                await asyncio.sleep(self.cleanup_interval)
                
                current_time = time.time()
                stale_threshold = 300  # 5 minutes without activity
                
                workspaces_to_cleanup = []
                
                for workspace_id, connections in self.connections.items():
                    stale_connections = []
                    
                    for connection in connections:
                        # Check if connection is stale
                        if (current_time - connection.last_activity > stale_threshold or 
                            not connection.is_active):
                            stale_connections.append(connection)
                    
                    # Remove stale connections
                    for stale_conn in stale_connections:
                        connections.remove(stale_conn)
                        self.active_connections -= 1
                        
                        logger.info(
                            "Cleaned up stale WebSocket connection",
                            workspace_id=workspace_id,
                            connection_id=stale_conn.connection_id,
                            inactive_duration=current_time - stale_conn.last_activity
                        )
                    
                    # Mark empty workspaces for cleanup
                    if not connections:
                        workspaces_to_cleanup.append(workspace_id)
                
                # Clean up empty workspaces
                for workspace_id in workspaces_to_cleanup:
                    del self.connections[workspace_id]
                
            except Exception as e:
                logger.error("Error in connection cleanup", error=str(e))
    
    async def _ping_connections(self):
        """Background task to ping connections"""
        
        while True:
            try:
                await asyncio.sleep(self.ping_interval)
                
                for workspace_id, connections in self.connections.items():
                    for connection in connections[:]:  # Copy to avoid modification during iteration
                        if connection.is_active:
                            success = await connection.send_ping()
                            if not success:
                                # Connection failed, will be cleaned up by cleanup task
                                logger.debug(
                                    "Ping failed for connection",
                                    workspace_id=workspace_id,
                                    connection_id=connection.connection_id
                                )
                
            except Exception as e:
                logger.error("Error in connection ping", error=str(e))
    
    def get_workspace_connections(self, workspace_id: str) -> List[str]:
        """Get list of active connection IDs for workspace"""
        
        if workspace_id not in self.connections:
            return []
        
        return [
            conn.connection_id 
            for conn in self.connections[workspace_id] 
            if conn.is_active
        ]
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """Get WebSocket manager statistics"""
        
        workspace_stats = {}
        for workspace_id, connections in self.connections.items():
            active_conns = [conn for conn in connections if conn.is_active]
            workspace_stats[workspace_id] = {
                "active_connections": len(active_conns),
                "total_messages_sent": sum(conn.messages_sent for conn in active_conns),
                "avg_connection_duration": sum(
                    time.time() - conn.connected_at for conn in active_conns
                ) / len(active_conns) if active_conns else 0
            }
        
        return {
            "total_connections_created": self.total_connections,
            "active_connections": self.active_connections,
            "active_workspaces": len(self.connections),
            "workspace_stats": workspace_stats,
            "background_tasks_running": len(self.background_tasks)
        }
    
    async def shutdown(self):
        """Gracefully shutdown WebSocket manager"""
        
        logger.info("Shutting down WebSocket manager")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close all connections
        for workspace_id, connections in self.connections.items():
            for connection in connections:
                try:
                    await connection.send_message({
                        "type": "server_shutdown",
                        "message": "Server is shutting down"
                    })
                    await connection.websocket.close()
                except:
                    pass  # Ignore errors during shutdown
        
        self.connections.clear()
        self.active_connections = 0

# Global WebSocket manager instance
websocket_manager = WebSocketManager()