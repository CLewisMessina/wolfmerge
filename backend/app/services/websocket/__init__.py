# app/services/websocket/__init__.py
"""
WebSocket Management Module for Real-time Progress Updates

Provides real-time communication between backend processing and frontend UI
for live progress tracking during document analysis.
"""

from .manager import WebSocketManager, ConnectionManager
from .progress_handler import ProgressHandler, ProgressMessage

__all__ = [
    "WebSocketManager",
    "ConnectionManager", 
    "ProgressHandler",
    "ProgressMessage"
]