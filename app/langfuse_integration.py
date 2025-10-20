# -*- coding: utf-8 -*-
"""
Basic Langfuse Integration - Trace Logging Only

This module handles logging all chat interactions to Langfuse for observability.
"""

import os
from datetime import datetime
from typing import Optional, Dict, Any
from langfuse import Langfuse
from config import LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, LANGFUSE_HOST

# Initialize Langfuse client
try:
    # Debug: Print the keys to see if they're loaded
    print("[*] Langfuse keys check:")
    print(f"   Public key: {LANGFUSE_PUBLIC_KEY[:20]}..." if LANGFUSE_PUBLIC_KEY else "   Public key: NOT_FOUND")
    print(f"   Secret key: {LANGFUSE_SECRET_KEY[:20]}..." if LANGFUSE_SECRET_KEY else "   Secret key: NOT_FOUND")
    print(f"   Host: {LANGFUSE_HOST}")
    
    if not LANGFUSE_SECRET_KEY or not LANGFUSE_PUBLIC_KEY:
        print("[!] Langfuse keys not found, skipping initialization")
        langfuse_client = None
    else:
        langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST
        )
        print("[OK] Langfuse client initialized successfully")
        print(f"   Host: {LANGFUSE_HOST}")
except Exception as e:
    print(f"[!] Warning: Failed to initialize Langfuse client: {e}")
    langfuse_client = None


class LangfuseTracker:
    """Handles Langfuse trace logging"""
    
    def __init__(self):
        self.client = langfuse_client
    
    def create_trace(
        self, 
        user_id: str, 
        question: str, 
        answer: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[str]:
        """
        Create a new trace in Langfuse for a chat interaction
        
        Args:
            user_id: User identifier
            question: User's question
            answer: Bot's answer
            session_id: Session identifier
            metadata: Additional metadata
        
        Returns:
            trace_id: Unique identifier for this trace, or None if logging fails
        """
        if not self.client:
            print("[!] Langfuse client not initialized, skipping trace logging")
            return None
        
        try:
            # Create trace with input/output at trace level for UI display
            trace = self.client.trace(
                name="chat_interaction",
                user_id=user_id,
                session_id=session_id or user_id,
                input=question,  # Set input at trace level
                output=answer,   # Set output at trace level
                metadata={
                    **(metadata or {}),
                    "timestamp": datetime.now().isoformat(),
                },
                tags=["chat", "slack2teams"]
            )
            
            # Also add generation span for detailed LLM metrics
            generation = trace.generation(
                name="chat_response",
                model="gpt-4o-mini",
                input=question,
                output=answer,
                metadata={
                    "timestamp": datetime.now().isoformat(),
                    "user_id": user_id
                }
            )
            
            trace_id = trace.id
            print(f"[OK] Trace logged to Langfuse: {trace_id}")
            
            return trace_id
            
        except Exception as e:
            print(f"[ERROR] Error creating Langfuse trace: {e}")
            return None
    
    def add_feedback(
        self,
        trace_id: str,
        rating: str,
        comment: Optional[str] = None
    ) -> bool:
        """
        Add user feedback to a trace in Langfuse
        
        Args:
            trace_id: The trace ID to add feedback to
            rating: "thumbs_up" or "thumbs_down"
            comment: Optional comment from the user
        
        Returns:
            bool: True if feedback was recorded successfully, False otherwise
        """
        if not self.client:
            print("[!] Langfuse client not initialized, skipping feedback logging")
            return False
        
        try:
            # Convert rating to numeric value
            # thumbs_up = 1, thumbs_down = 0
            feedback_value = 1 if rating == "thumbs_up" else 0
            
            # Add feedback to Langfuse
            self.client.score(
                trace_id=trace_id,
                name="user_rating",
                value=feedback_value,
                comment=comment or ""
            )
            
            print(f"[OK] Feedback logged to Langfuse: trace_id={trace_id}, rating={rating}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Error adding feedback to Langfuse: {e}")
            return False


# Global tracker instance
langfuse_tracker = LangfuseTracker()

