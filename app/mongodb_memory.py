from typing import List, Dict, Optional
import asyncio
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ConnectionFailure, DuplicateKeyError
import logging

from config import MONGODB_URL, MONGODB_DATABASE, MONGODB_CHAT_COLLECTION

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MongoDBMemoryManager:
    """MongoDB-based chat history management."""
    
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.database = None
        self.collection = None
        self._connection_lock = asyncio.Lock()
    
    async def connect(self):
        """Initialize MongoDB connection."""
        async with self._connection_lock:
            if self.client is None:
                try:
                    self.client = AsyncIOMotorClient(MONGODB_URL)
                    self.database = self.client[MONGODB_DATABASE]
                    self.collection = self.database[MONGODB_CHAT_COLLECTION]
                    
                    # Test connection
                    await self.client.admin.command('ping')
                    logger.info(f"Connected to MongoDB: {MONGODB_DATABASE}.{MONGODB_CHAT_COLLECTION}")
                    
                    # Create indexes for better performance
                    await self._create_indexes()
                    
                except ConnectionFailure as e:
                    logger.error(f"Failed to connect to MongoDB: {e}")
                    raise e
    
    async def _create_indexes(self):
        """Create database indexes for better performance."""
        try:
            # Index on user_id for fast lookups
            await self.collection.create_index("user_id", unique=True)
            # Index on timestamp for sorting
            await self.collection.create_index("last_updated")
            logger.info("MongoDB indexes created successfully")
        except Exception as e:
            logger.warning(f"Could not create indexes: {e}")
    
    async def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            self.client = None
            self.database = None
            self.collection = None
            logger.info("Disconnected from MongoDB")
    
    async def get_or_create_user_conversation(self, user_id: str) -> List[Dict[str, str]]:
        """Get or create a conversation for a specific user."""
        await self.connect()
        
        try:
            # Try to find existing conversation
            user_doc = await self.collection.find_one({"user_id": user_id})
            
            if user_doc:
                return user_doc.get("messages", [])
            else:
                # Create new conversation
                new_conversation = {
                    "user_id": user_id,
                    "messages": [],
                    "created_at": datetime.utcnow(),
                    "last_updated": datetime.utcnow()
                }
                await self.collection.insert_one(new_conversation)
                return []
                
        except Exception as e:
            logger.error(f"Error getting/creating conversation for user {user_id}: {e}")
            return []
    
    async def add_to_conversation(self, user_id: str, role: str, content: str):
        """Add a message to the user's conversation history."""
        await self.connect()
        
        try:
            # Get current conversation
            conversation = await self.get_or_create_user_conversation(user_id)
            
            # Add new message
            new_message = {
                "role": role,
                "content": content,
                "timestamp": datetime.utcnow()
            }
            conversation.append(new_message)
            
            # Keep only last 20 messages to prevent context overflow
            if len(conversation) > 20:
                conversation = conversation[-20:]
            
            # Update in database
            await self.collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "messages": conversation,
                        "last_updated": datetime.utcnow()
                    }
                },
                upsert=True
            )
            
        except Exception as e:
            logger.error(f"Error adding message to conversation for user {user_id}: {e}")
    
    async def get_conversation_context(self, user_id: str) -> str:
        """Get formatted conversation context for a user."""
        conversation = await self.get_or_create_user_conversation(user_id)
        
        if not conversation:
            return ""
        
        context = "\n\nPrevious conversation:\n"
        # Get last 5 messages for context
        for msg in conversation[-5:]:
            role = "User" if msg["role"] == "user" else "Assistant"
            context += f"{role}: {msg['content']}\n"
        
        return context
    
    async def get_user_chat_history(self, user_id: str) -> List[Dict[str, str]]:
        """Get full chat history for a user."""
        return await self.get_or_create_user_conversation(user_id)
    
    async def clear_user_chat_history(self, user_id: str):
        """Clear chat history for a specific user."""
        await self.connect()
        
        try:
            await self.collection.update_one(
                {"user_id": user_id},
                {
                    "$set": {
                        "messages": [],
                        "last_updated": datetime.utcnow()
                    }
                },
                upsert=True
            )
            logger.info(f"Cleared chat history for user {user_id}")
            
        except Exception as e:
            logger.error(f"Error clearing chat history for user {user_id}: {e}")
    
    async def get_all_users(self) -> List[str]:
        """Get list of all user IDs in the database."""
        await self.connect()
        
        try:
            cursor = self.collection.find({}, {"user_id": 1})
            user_ids = [doc["user_id"] async for doc in cursor]
            return user_ids
        except Exception as e:
            logger.error(f"Error getting all users: {e}")
            return []
    
    async def get_conversation_stats(self) -> Dict:
        """Get statistics about conversations."""
        await self.connect()
        
        try:
            total_users = await self.collection.count_documents({})
            
            # Get total messages across all users
            pipeline = [
                {"$project": {"message_count": {"$size": "$messages"}}},
                {"$group": {"_id": None, "total_messages": {"$sum": "$message_count"}}}
            ]
            
            result = await self.collection.aggregate(pipeline).to_list(1)
            total_messages = result[0]["total_messages"] if result else 0
            
            return {
                "total_users": total_users,
                "total_messages": total_messages,
                "database": MONGODB_DATABASE,
                "collection": MONGODB_CHAT_COLLECTION
            }
            
        except Exception as e:
            logger.error(f"Error getting conversation stats: {e}")
            return {"error": str(e)}

# Global instance
mongodb_memory = MongoDBMemoryManager()

# Async wrapper functions to maintain compatibility with existing code
async def get_or_create_user_conversation(user_id: str) -> List[Dict[str, str]]:
    """Get or create a conversation for a specific user."""
    return await mongodb_memory.get_or_create_user_conversation(user_id)

async def add_to_conversation(user_id: str, role: str, content: str):
    """Add a message to the user's conversation history."""
    await mongodb_memory.add_to_conversation(user_id, role, content)

async def get_conversation_context(user_id: str) -> str:
    """Get formatted conversation context for a user."""
    return await mongodb_memory.get_conversation_context(user_id)

async def get_user_chat_history(user_id: str) -> List[Dict[str, str]]:
    """Get full chat history for a user."""
    return await mongodb_memory.get_user_chat_history(user_id)

async def clear_user_chat_history(user_id: str):
    """Clear chat history for a specific user."""
    await mongodb_memory.clear_user_chat_history(user_id)

# Additional utility functions
async def get_all_users() -> List[str]:
    """Get list of all user IDs in the database."""
    return await mongodb_memory.get_all_users()

async def get_conversation_stats() -> Dict:
    """Get statistics about conversations."""
    return await mongodb_memory.get_conversation_stats()

async def close_mongodb_connection():
    """Close MongoDB connection."""
    await mongodb_memory.disconnect()
