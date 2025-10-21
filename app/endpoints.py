# -*- coding: utf-8 -*-
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
import uuid
import httpx
import os
import json
import asyncio
import re
from datetime import datetime

from app.llm import setup_qa_chain
from app.vectorstore import retriever, vectorstore
from app.mongodb_memory import add_to_conversation, get_conversation_context, get_user_chat_history, clear_user_chat_history
from app.helpers import strip_markdown, preserve_markdown
from app.langfuse_integration import langfuse_tracker
from config import SYSTEM_PROMPT, MICROSOFT_CLIENT_ID, MICROSOFT_CLIENT_SECRET, MICROSOFT_TENANT
from langchain_core.prompts import ChatPromptTemplate


router = APIRouter()

qa_chain = setup_qa_chain(retriever)

# File path for corrected responses
CORRECTED_RESPONSES_FILE = "./data/corrected_responses/corrected_responses.json"

def load_corrected_responses():
    """Load corrected responses from JSON file."""
    try:
        if os.path.exists(CORRECTED_RESPONSES_FILE):
            with open(CORRECTED_RESPONSES_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('corrected_responses', [])
    except Exception as e:
        print(f"Error loading corrected responses: {e}")
    return []

def find_similar_corrected_response(question: str, threshold: float = 0.7):
    """Check if there's a corrected response for a similar question."""
    from difflib import SequenceMatcher
    
    corrected_responses = load_corrected_responses()
    
    if not corrected_responses:
        return None
    
    # Get the original questions from feedback history to match against
    feedback_file = "./data/feedback_history.json"
    if os.path.exists(feedback_file):
        try:
            with open(feedback_file, 'r', encoding='utf-8') as f:
                feedback_history = json.load(f)
                
                best_match = None
                best_score = 0
                
                for feedback in feedback_history:
                    if feedback.get('rating') == 'thumbs_down':
                        original_question = feedback.get('question', '')
                        trace_id = feedback.get('trace_id', '')
                        
                        # Calculate similarity
                        similarity = SequenceMatcher(None, question.lower(), original_question.lower()).ratio()
                        
                        if similarity > best_score and similarity >= threshold:
                            # Find the corrected response for this trace_id
                            for corrected in corrected_responses:
                                if corrected.get('trace_id') == trace_id:
                                    best_score = similarity
                                    best_match = {
                                        'response': corrected.get('corrected_response'),
                                        'similarity': similarity,
                                        'original_question': original_question
                                    }
                                    break
                
                if best_match:
                    print(f"✅ Found corrected response (similarity: {best_match['similarity']:.2%})")
                    print(f"   Original question: {best_match['original_question']}")
                    return best_match['response']
                    
        except Exception as e:
            print(f"Error checking feedback history: {e}")
    
    return None

def is_conversational_query(question: str) -> bool:
    """Determine if a query is conversational/social rather than informational."""
    question_lower = question.lower().strip()
    
    # Common conversational patterns
    conversational_patterns = [
        r'^(hi|hello|hey|hiya|howdy)',
        r'^(how are you|how\'re you|how do you do)',
        r'^(what\'s up|whats up|wassup)',
        r'^(good morning|good afternoon|good evening)',
        r'^(thanks|thank you|thx)',
        r'^(bye|goodbye|see you|farewell)',
        r'^(yes|no|ok|okay|sure|alright)',
        r'^(what|who|where|when|why|how)\s+(are you|is it|was it)',
        r'^(tell me about yourself|who are you)',
        r'^(what can you do|what do you do)',
        r'^(help|can you help)',
        r'^(sorry|excuse me|pardon)',
        r'^(nice|good|great|awesome|cool|wow)',
        r'^(please|pls)',
    ]
    
    # Check if question matches conversational patterns
    for pattern in conversational_patterns:
        if re.match(pattern, question_lower):
            return True
    
    # Check for very short queries (likely conversational)
    if len(question.strip()) < 10 and not any(word in question_lower for word in ['what', 'how', 'why', 'when', 'where', 'who', 'which']):
        return True
    
    # Check if it's a simple greeting or social interaction
    social_words = ['hi', 'hello', 'hey', 'thanks', 'bye', 'good', 'nice', 'great', 'cool', 'awesome']
    if any(word in question_lower for word in social_words) and len(question.split()) <= 3:
        return True
    
    return False

class ChatRequest(BaseModel):
    question: str
    user_id: str = None
    session_id: str = None  # Keep for backward compatibility

class FeedbackRequest(BaseModel):
    trace_id: str
    rating: str  # "thumbs_up" or "thumbs_down"
    comment: str = None

@router.post("/chat")
async def chat(request: Request):
    """Chat endpoint: returns full answer from vectorstore."""
    data = await request.json()
    question = data.get("question", "")
    user_id = data.get("user_id")
    session_id = data.get("session_id", str(uuid.uuid4()))

    # Use user_id if provided, otherwise fall back to session_id for backward compatibility
    conversation_id = user_id if user_id else session_id

    # FIRST: Check if we have a corrected response for this question
    corrected_answer = find_similar_corrected_response(question)
    
    if corrected_answer:
        # Use the corrected response
        answer = corrected_answer
    # Check if this is a conversational query
    elif is_conversational_query(question):
        # Handle conversational queries directly without document retrieval
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
        
        # Simple conversational prompt
        conversational_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly and helpful AI assistant. Respond naturally to conversational queries like greetings, 'how are you', etc. Be warm and engaging."),
            ("human", "{question}")
        ])
        
        # Get conversation context for continuity
        conversation_context = await get_conversation_context(conversation_id)
        enhanced_query = f"{conversation_context}\n\nUser: {question}" if conversation_context else question
        
        chain = conversational_prompt | llm
        result = chain.invoke({"question": enhanced_query})
        answer = result.content
    else:
        # Handle informational queries with document retrieval
        conversation_context = await get_conversation_context(conversation_id)
        enhanced_query = f"{conversation_context}\n\nUser: {question}" if conversation_context else question
        result = qa_chain.invoke({"query": enhanced_query})
        answer = result["result"]

    # Add both user question and bot response to conversation AFTER processing
    await add_to_conversation(conversation_id, "user", question)
    await add_to_conversation(conversation_id, "assistant", answer)

    # Log to Langfuse for observability
    trace_id = langfuse_tracker.create_trace(
        user_id=conversation_id,
        question=question,
        answer=answer,
        session_id=session_id,
        metadata={
            "endpoint": "/chat",
            "conversational_query": is_conversational_query(question)
        }
    )

    clean_answer = preserve_markdown(answer)
    return {"answer": clean_answer, "user_id": user_id, "session_id": session_id, "trace_id": trace_id}

# ---------------- Streaming Chat Endpoint ----------------

@router.post("/chat/stream")
async def chat_stream(request: Request):
    data = await request.json()
    question = data.get("question", "")
    user_id = data.get("user_id")
    session_id = data.get("session_id", str(uuid.uuid4()))

    # Use user_id if provided, otherwise fall back to session_id for backward compatibility
    conversation_id = user_id if user_id else session_id

    async def generate_stream():
        try:
            # FIRST: Check if we have a corrected response for this question
            corrected_answer = find_similar_corrected_response(question)
            
            if corrected_answer:
                # Use the corrected response
                yield f"data: {json.dumps({'type': 'thinking_complete'})}\n\n"
                
                # Stream the corrected answer token by token
                full_response = corrected_answer
                for i, char in enumerate(corrected_answer):
                    yield f"data: {json.dumps({'token': char, 'type': 'token'})}\n\n"
                    if i % 5 == 0:  # Add slight delay every 5 characters
                        await asyncio.sleep(0.01)
                
                # Add to conversation
                await add_to_conversation(conversation_id, "user", question)
                await add_to_conversation(conversation_id, "assistant", full_response)
                
                # Log to Langfuse
                trace_id = None
                try:
                    trace_id = langfuse_tracker.create_trace(
                        user_id=conversation_id,
                        question=question,
                        answer=full_response,
                        session_id=session_id,
                        metadata={
                            "endpoint": "/chat/stream",
                            "used_corrected_response": True
                        }
                    )
                except Exception as e:
                    print(f"Warning: Langfuse logging failed: {e}")
                
                yield f"data: {json.dumps({'type': 'done', 'full_response': full_response, 'trace_id': trace_id})}\n\n"
                return
            
            # Get conversation context BEFORE adding current question
            conversation_context = await get_conversation_context(conversation_id)

            # Combine question with conversation context for better continuity
            enhanced_query = f"{conversation_context}\n\nUser: {question}" if conversation_context else question
            
            # Check if this is a conversational query
            is_conv = is_conversational_query(question)
            
            if is_conv:
                # Handle conversational queries directly without document retrieval
                yield f"data: {json.dumps({'type': 'thinking_complete'})}\n\n"
                
                from langchain_openai import ChatOpenAI
                from langchain_core.prompts import ChatPromptTemplate
                
                llm = ChatOpenAI(
                    model_name="gpt-4o-mini", 
                    streaming=True, 
                    temperature=0.7,
                    max_tokens=500
                )
                
                # Simple conversational prompt
                conversational_prompt = ChatPromptTemplate.from_messages([
                    ("system", "You are a friendly and helpful AI assistant. Respond naturally to conversational queries like greetings, 'how are you', etc. Be warm and engaging."),
                    ("human", "{question}")
                ])
                
                # Stream the response
                full_response = ""
                messages = conversational_prompt.format_messages(question=enhanced_query)
                async for chunk in llm.astream(messages):
                    if hasattr(chunk, 'content'):
                        token = chunk.content
                        full_response += token
                        yield f"data: {json.dumps({'token': token, 'type': 'token'})}\n\n"
                        await asyncio.sleep(0.01)
                
                # Add to conversation
                await add_to_conversation(conversation_id, "user", question)
                await add_to_conversation(conversation_id, "assistant", full_response)
                
                # Log to Langfuse (don't block response if this fails)
                trace_id = None
                try:
                    trace_id = langfuse_tracker.create_trace(
                        user_id=conversation_id,
                        question=question,
                        answer=full_response,
                        session_id=session_id,
                        metadata={
                            "endpoint": "/chat/stream",
                            "conversational_query": True,
                            "streaming": True
                        }
                    )
                except Exception as e:
                    print(f"Langfuse logging failed: {e}")
                
                # Send completion signal with trace_id
                yield f"data: {json.dumps({'type': 'done', 'full_response': full_response, 'trace_id': trace_id})}\n\n"
                return
            
            # PHASE 1: THINKING - Document retrieval and processing
            # This happens while the frontend shows "Thinking..." animation
            
            try:
                # Enhanced semantic search with better coverage
                final_docs = vectorstore.similarity_search(enhanced_query, k=25)
            except Exception as e:
                print(f"Error during document search: {e}")
                final_docs = []
            
            # Format the documents properly
            context_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(final_docs)])
            
            # Send signal that thinking is complete and streaming will start
            yield f"data: {json.dumps({'type': 'thinking_complete'})}\n\n"
            
            # PHASE 2: STREAMING - Generate and stream response
            # This happens after the frontend clears the "Thinking..." animation
            
            # Import ChatOpenAI for document-based queries
            from langchain_openai import ChatOpenAI
            
            # Create streaming LLM
            llm = ChatOpenAI(
                model_name="gpt-4o-mini", 
                streaming=True, 
                temperature=0.3,
                max_tokens=1500
            )
            
            # Create the prompt template
            from langchain_core.prompts import ChatPromptTemplate
            from config import SYSTEM_PROMPT
            
            prompt_template = ChatPromptTemplate.from_messages([
                ("system", SYSTEM_PROMPT),
                ("human", "Context: {context}\n\nQuestion: {question}")
            ])
            
            # Stream the response with real-time streaming
            full_response = ""
            messages = prompt_template.format_messages(context=context_text, question=enhanced_query)
            async for chunk in llm.astream(messages):
                if hasattr(chunk, 'content'):
                    token = chunk.content
                    full_response += token
                    yield f"data: {json.dumps({'token': token, 'type': 'token'})}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for better streaming effect
            
            
            # Add both user question and bot response to conversation AFTER processing
            await add_to_conversation(conversation_id, "user", question)
            await add_to_conversation(conversation_id, "assistant", full_response)
            
            # Log to Langfuse (don't block response if this fails)
            trace_id = None
            try:
                trace_id = langfuse_tracker.create_trace(
                    user_id=conversation_id,
                    question=question,
                    answer=full_response,
                    session_id=session_id,
                    metadata={
                        "endpoint": "/chat/stream",
                        "conversational_query": is_conversational_query(question),
                        "streaming": True
                    }
                )
            except Exception as e:
                print(f"⚠️ Langfuse logging failed: {e}")
            
            # Send completion signal with trace_id
            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response, 'trace_id': trace_id})}\n\n"
            
        except Exception as e:
            print(f"❌ ERROR in generate_stream: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'error': str(e), 'type': 'error'})}\n\n"

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

# ---------------- User Chat History Endpoints ----------------

@router.get("/chat/history/{user_id}")
async def get_chat_history(user_id: str):
    """Get chat history for a specific user."""
    try:
        history = await get_user_chat_history(user_id)
        return {"user_id": user_id, "history": history}
    except Exception as e:
        return {"error": str(e)}

@router.delete("/chat/history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history for a specific user."""
    try:
        await clear_user_chat_history(user_id)
        return {"message": f"Chat history cleared for user {user_id}"}
    except Exception as e:
        return {"error": str(e)}

# ---------------- Feedback Endpoint ----------------

@router.post("/feedback")
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback (👍/👎) for a chat interaction."""
    try:
        # Validate rating
        if request.rating not in ["thumbs_up", "thumbs_down"]:
            return {"error": "Invalid rating. Must be 'thumbs_up' or 'thumbs_down'"}
        
        # Log feedback to Langfuse
        success = langfuse_tracker.add_feedback(
            trace_id=request.trace_id,
            rating=request.rating,
            comment=request.comment
        )
        
        # Track feedback history for smart auto-correction
        await track_feedback_history(request.trace_id, request.rating, request.comment)
        
        # If thumbs down, trigger auto-correction immediately
        if request.rating == "thumbs_down":
            try:
                # Get the original question and response from the trace
                original_data = await get_trace_data(request.trace_id)
                if original_data:
                    # Trigger auto-correction
                    corrected_response = await trigger_auto_correction_workflow(
                        trace_id=request.trace_id,
                        user_query=original_data.get("question", ""),
                        bad_response=original_data.get("response", ""),
                        user_comment=request.comment
                    )
            except Exception as e:
                print(f"Auto-correction failed: {e}")
        
        if success:
            return {
                "status": "success",
                "message": "Feedback recorded successfully",
                "trace_id": request.trace_id,
                "auto_correction_triggered": request.rating == "thumbs_down"
            }
        else:
            return {
                "status": "error",
                "message": "Failed to record feedback"
            }
            
    except Exception as e:
        return {"error": f"Failed to submit feedback: {str(e)}"}

# ---------------- Auto-Correction System ----------------

async def track_feedback_history(trace_id: str, rating: str, comment: str = None):
    """Track feedback history for smart auto-correction decisions."""
    try:
        feedback_file = "./data/feedback_history.json"
        
        # Load existing data
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"feedback_history": {}}
        
        # Initialize trace data if not exists
        if trace_id not in data["feedback_history"]:
            data["feedback_history"][trace_id] = {
                "negative_count": 0,
                "positive_count": 0,
                "total_count": 0,
                "question_asked_before": False,
                "feedback_history": []
            }
        
        # Update counts
        trace_data = data["feedback_history"][trace_id]
        trace_data["total_count"] += 1
        
        if rating == "thumbs_down":
            trace_data["negative_count"] += 1
        else:
            trace_data["positive_count"] += 1
        
        # Add to feedback history
        feedback_entry = {
            "rating": rating,
            "comment": comment,
            "timestamp": datetime.now().isoformat()
        }
        trace_data["feedback_history"].append(feedback_entry)
        
        # Save updated data
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"Error tracking feedback history: {e}")

async def should_trigger_auto_correction(trace_id: str, user_comment: str = None) -> bool:
    """Simplified auto-correction logic - more effective and predictable."""
    try:
        # Load feedback history
        feedback_stats = await get_feedback_stats_for_question(trace_id)
        negative_count = feedback_stats.get('negative_count', 0)
        
        # SIMPLIFIED RULES - More predictable and effective
        
        # Rule 1: Always correct if user provides specific feedback
        if user_comment and len(user_comment.strip()) > 15:
            return True
        
        # Rule 2: Correct if 2+ negative feedbacks (simple threshold)
        if negative_count >= 2:
            return True
        
        # Rule 3: Correct if this is a recurring problem (same question, multiple negatives)
        if negative_count >= 1 and feedback_stats.get('question_asked_before', False):
            return True
        
        # Rule 4: Don't correct for single negative feedback
        return False
            
    except Exception as e:
        print(f"Error in auto-correction logic: {e}")
        return False

async def get_feedback_stats_for_question(trace_id: str) -> dict:
    """Get feedback statistics for a question to make smart decisions."""
    try:
        # In a real implementation, you'd query Langfuse API for feedback history
        # For now, we'll simulate with local data
        
        # Check if we have feedback data for this trace
        feedback_file = "./data/feedback_history.json"
        
        if os.path.exists(feedback_file):
            with open(feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Find feedback for this trace
            trace_feedback = data.get('feedback_history', {}).get(trace_id, {})
            return trace_feedback
        else:
            # No history, treat as first time
            return {
                'negative_count': 0,
                'total_count': 0,
                'question_asked_before': False
            }
            
    except Exception as e:
        print(f"Error getting feedback stats: {e}")
        return {
            'negative_count': 0,
            'total_count': 0,
            'question_asked_before': False
        }

async def trigger_auto_correction(trace_id: str, user_comment: str = None):
    """Trigger auto-correction for a thumbs down feedback."""
    try:
        # Get the original trace data from Langfuse
        # For now, we'll simulate getting the original question and answer
        # In a real implementation, you'd fetch this from Langfuse API
        
        # Create auto-correction prompt
        correction_prompt = f"""
        The user gave negative feedback (👎) on this response. Please provide a better, improved version.
        
        Original question: [QUESTION_PLACEHOLDER]
        Original answer: [ANSWER_PLACEHOLDER]
        User feedback: {user_comment or "User indicated the response was not helpful"}
        
        Please provide an improved response that:
        1. Directly addresses the user's question
        2. Is more helpful and accurate
        3. Provides specific, actionable information
        4. Is clear and well-structured
        
        Improved response:
        """
        
        # Use LLM to generate improved response
        from langchain_openai import ChatOpenAI
        from langchain_core.prompts import ChatPromptTemplate
        
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
        
        # For now, we'll create a generic improved response
        # In a real implementation, you'd fetch the original Q&A from Langfuse
        improved_response = await generate_improved_response(correction_prompt, llm)
        
        # Save to dataset
        save_corrected_response(trace_id, improved_response, user_comment)
        
    except Exception as e:
        print(f"Auto-correction failed: {e}")
        raise e

async def generate_improved_response(prompt: str, llm):
    """Generate an improved response using LLM."""
    try:
        # Create a simple prompt template
        template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert AI assistant. Improve the given response to be more helpful and accurate."),
            ("human", "{prompt}")
        ])
        
        chain = template | llm
        result = chain.invoke({"prompt": prompt})
        return result.content
        
    except Exception as e:
        print(f"Error generating improved response: {e}")
        return "I apologize, but I'm having trouble generating an improved response at the moment."

def save_corrected_response(trace_id: str, corrected_response: str, user_comment: str = None):
    """Save the corrected response to the dataset."""
    try:
        # Create dataset directory if it doesn't exist
        dataset_dir = "./data/corrected_responses"
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create dataset entry
        dataset_entry = {
            "trace_id": trace_id,
            "corrected_response": corrected_response,
            "user_comment": user_comment,
            "timestamp": datetime.now().isoformat(),
            "status": "corrected"
        }
        
        # Save to JSON file
        dataset_file = f"{dataset_dir}/corrected_responses.json"
        
        # Load existing data
        if os.path.exists(dataset_file):
            with open(dataset_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"corrected_responses": []}
        
        # Add new entry
        data["corrected_responses"].append(dataset_entry)
        
        # Save back to file
        with open(dataset_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
    except Exception as e:
        print(f"Error saving corrected response: {e}")

@router.get("/dataset/corrected-responses")
async def get_corrected_responses():
    """Get all corrected responses from the dataset."""
    try:
        dataset_file = "./data/corrected_responses/corrected_responses.json"
        
        if not os.path.exists(dataset_file):
            return {"corrected_responses": [], "count": 0}
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return {
            "corrected_responses": data.get("corrected_responses", []),
            "count": len(data.get("corrected_responses", []))
        }
        
    except Exception as e:
        return {"error": f"Failed to load corrected responses: {str(e)}"}

@router.delete("/dataset/corrected-responses")
async def clear_corrected_responses():
    """Clear all corrected responses from the dataset."""
    try:
        dataset_file = "./data/corrected_responses/corrected_responses.json"
        
        if os.path.exists(dataset_file):
            os.remove(dataset_file)
            return {"message": "Corrected responses dataset cleared"}
        else:
            return {"message": "No dataset found to clear"}
            
    except Exception as e:
        return {"error": f"Failed to clear dataset: {str(e)}"}

# ---------------- Manual Fine-Tuning System ----------------

@router.post("/fine-tuning/trigger")
async def trigger_manual_fine_tuning():
    """Manually trigger fine-tuning when needed."""
    try:
        # Check if we have enough data for fine-tuning
        dataset_status = await check_dataset_quality()
        
        if not dataset_status["ready_for_training"]:
            return {
                "status": "insufficient_data",
                "message": f"Not enough data for fine-tuning. Need at least {dataset_status['min_required']} samples, have {dataset_status['current_count']}",
                "recommendations": dataset_status["recommendations"]
            }
        
        # Start fine-tuning process
        fine_tuning_result = await start_fine_tuning_process()
        
        return {
            "status": "success",
            "message": "Fine-tuning process started",
            "dataset_info": dataset_status,
            "process_id": fine_tuning_result["process_id"]
        }
        
    except Exception as e:
        return {"error": f"Failed to trigger fine-tuning: {str(e)}"}

@router.get("/fine-tuning/status")
async def get_fine_tuning_status():
    """Get the status of fine-tuning process."""
    try:
        # Check dataset quality
        dataset_status = await check_dataset_quality()
        
        # Check if fine-tuning is in progress
        training_status = await get_training_status()
        
        return {
            "dataset_quality": dataset_status,
            "training_status": training_status,
            "recommendations": await get_fine_tuning_recommendations()
        }
        
    except Exception as e:
        return {"error": f"Failed to get fine-tuning status: {str(e)}"}

async def check_dataset_quality():
    """Check if dataset is ready for fine-tuning."""
    try:
        dataset_file = "./data/corrected_responses/corrected_responses.json"
        
        if not os.path.exists(dataset_file):
            return {
                "ready_for_training": False,
                "current_count": 0,
                "min_required": 10,
                "recommendations": ["Collect more negative feedback data"]
            }
        
        with open(dataset_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        corrected_responses = data.get("corrected_responses", [])
        current_count = len(corrected_responses)
        min_required = 10  # Minimum samples needed
        
        # Quality checks
        quality_score = 0
        recommendations = []
        
        if current_count >= min_required:
            quality_score += 40
        else:
            recommendations.append(f"Need {min_required - current_count} more samples")
        
        # Check for diverse feedback
        unique_questions = len(set([item.get("original_question", "") for item in corrected_responses]))
        if unique_questions >= 5:
            quality_score += 30
        else:
            recommendations.append("Need more diverse question types")
        
        # Check for recent data
        recent_count = 0
        for item in corrected_responses:
            timestamp = datetime.fromisoformat(item.get("timestamp", ""))
            if (datetime.now() - timestamp).days <= 7:
                recent_count += 1
        
        if recent_count >= 3:
            quality_score += 30
        else:
            recommendations.append("Need more recent feedback data")
        
        return {
            "ready_for_training": quality_score >= 70,
            "current_count": current_count,
            "min_required": min_required,
            "quality_score": quality_score,
            "unique_questions": unique_questions,
            "recent_samples": recent_count,
            "recommendations": recommendations
        }
        
    except Exception as e:
        return {
            "ready_for_training": False,
            "error": str(e),
            "recommendations": ["Fix dataset issues"]
        }

async def start_fine_tuning_process():
    """Start the fine-tuning process."""
    try:
        # Generate a unique process ID
        process_id = f"ft_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # In a real implementation, you would:
        # 1. Export data from Langfuse
        # 2. Prepare training data
        # 3. Start fine-tuning job
        # 4. Monitor progress
        
        # For now, simulate the process
        training_status = {
            "process_id": process_id,
            "status": "started",
            "start_time": datetime.now().isoformat(),
            "estimated_completion": (datetime.now().timestamp() + 3600),  # 1 hour from now
            "progress": 0
        }
        
        # Save training status
        status_file = "./data/fine_tuning_status.json"
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(training_status, f, indent=2, ensure_ascii=False)
        
        return training_status
        
    except Exception as e:
        print(f"Failed to start fine-tuning: {e}")
        raise e

async def get_training_status():
    """Get current training status."""
    try:
        status_file = "./data/fine_tuning_status.json"
        
        if not os.path.exists(status_file):
            return {"status": "no_training", "message": "No fine-tuning in progress"}
        
        with open(status_file, 'r', encoding='utf-8') as f:
            status = json.load(f)
        
        return status
        
    except Exception as e:
        return {"status": "error", "message": str(e)}

async def get_fine_tuning_recommendations():
    """Get recommendations for when to run fine-tuning."""
    try:
        dataset_status = await check_dataset_quality()
        
        recommendations = []
        
        if not dataset_status["ready_for_training"]:
            recommendations.append("❌ Not ready for fine-tuning - insufficient data")
            recommendations.extend(dataset_status["recommendations"])
        else:
            recommendations.append("✅ Ready for fine-tuning!")
            recommendations.append("💡 Consider running fine-tuning when you have 20+ samples")
            recommendations.append("🔄 Run fine-tuning weekly for best results")
        
        return recommendations
        
    except Exception as e:
        return [f"Error getting recommendations: {str(e)}"]

# ---------------- Auto-Correction Workflow ----------------

async def get_trace_data(trace_id: str):
    """Get original question and response from trace data."""
    try:
        # In a real implementation, you would retrieve this from Langfuse
        # For now, we'll simulate it
        return {
            "question": "Sample question from trace",
            "response": "Sample response from trace"
        }
    except Exception as e:
        print(f"Error retrieving trace data: {e}")
        return None

async def trigger_auto_correction_workflow(trace_id: str, user_query: str, bad_response: str, user_comment: str = None):
    """Complete auto-correction workflow."""
    try:
        # Step 1: Generate improved response using LLM
        improved_response = await generate_improved_response(user_query, bad_response, user_comment)
        
        # Step 2: Save to correction dataset
        await save_correction_to_dataset(user_query, bad_response, improved_response, trace_id, user_comment)
        
        # Step 3: Update Langfuse trace (optional)
        await update_langfuse_trace(trace_id, improved_response)
        
        return improved_response
        
    except Exception as e:
        print(f"Auto-correction workflow failed: {e}")
        raise e

async def generate_improved_response(user_query: str, bad_response: str, user_comment: str = None):
    """Use LLM with RAG to generate an improved response using the knowledge base."""
    try:
        from langchain_openai import ChatOpenAI
        
        # CRITICAL: Retrieve relevant documents from vectorstore for context
        # This ensures the corrected response is based on actual knowledge base
        relevant_docs = vectorstore.similarity_search(user_query, k=25)
        
        # Format the retrieved documents as context
        context_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
        
        # Create LLM for auto-correction
        llm = ChatOpenAI(
            model_name="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1000
        )
        
        # Create auto-correction prompt WITH knowledge base context
        correction_prompt = f"""
You are an expert assistant specializing in Slack to Microsoft Teams migrations via CloudFuze.

KNOWLEDGE BASE CONTEXT:
{context_text}

PREVIOUS INTERACTION:
User's Question: "{user_query}"
Bot's Poor Response: "{bad_response}"
{f"User's Feedback: {user_comment}" if user_comment else ""}

TASK:
Using ONLY the information from the KNOWLEDGE BASE CONTEXT above, provide a much better, more accurate, and helpful response that:
1. Directly answers the user's question about Slack to Teams migration
2. Uses specific information from the knowledge base documents
3. Provides actionable information about CloudFuze's capabilities
4. Is clear, professional, and helpful
5. Follows the same style and formatting as the system prompt

IMPORTANT: Base your answer strictly on the knowledge base context provided above.

Improved response:
"""
        
        # Generate improved response with knowledge base context
        improved_response = llm.invoke(correction_prompt).content
        return improved_response
        
    except Exception as e:
        print(f"Error generating improved response: {e}")
        return f"I apologize for the previous response. Let me provide a better answer to your question: {user_query}. For detailed information about Slack to Teams migration, please contact our CloudFuze support team."

async def save_correction_to_dataset(user_query: str, bad_response: str, improved_response: str, trace_id: str, user_comment: str = None):
    """Save the correction to JSONL dataset."""
    try:
        import os
        from datetime import datetime
        
        # Create dataset directory if it doesn't exist
        dataset_dir = "./data/fine_tuning_dataset"
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create JSONL record
        correction_record = {
            "input": user_query,
            "bad_output": bad_response,
            "corrected_output": improved_response,
            "trace_id": trace_id,
            "user_comment": user_comment,
            "timestamp": datetime.now().isoformat(),
            "status": "auto_corrected"
        }
        
        # Save to single JSONL file (append mode)
        jsonl_file = f"{dataset_dir}/corrections.jsonl"
        
        with open(jsonl_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(correction_record, ensure_ascii=False) + '\n')
        
        # Also save to the existing corrected_responses.json for compatibility
        save_corrected_response(trace_id, improved_response, user_comment)
        
    except Exception as e:
        print(f"Error saving correction to dataset: {e}")

async def update_langfuse_trace(trace_id: str, improved_response: str):
    """Update Langfuse trace with corrected response."""
    try:
        if langfuse_tracker and langfuse_tracker.client:
            # Log the auto-correction as a score/annotation
            langfuse_tracker.client.score(
                trace_id=trace_id,
                name="auto_correction",
                value=1,
                comment=f"Auto-corrected response: {improved_response[:200]}..."
            )
    except Exception as e:
        print(f"Could not update Langfuse trace: {e}")

# ---------------- Microsoft OAuth Endpoints ----------------

class MicrosoftCallbackRequest(BaseModel):
    code: str
    redirect_uri: str
    code_verifier: str

@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify backend connectivity."""
    return {"message": "Backend is working", "status": "success"}

@router.post("/test-post")
async def test_post_endpoint(data: dict):
    """Test POST endpoint to verify CORS and connectivity."""
    return {"message": "POST request received", "data": data, "status": "success"}

@router.post("/auth/microsoft/callback")
async def microsoft_oauth_callback(request: MicrosoftCallbackRequest):
    """Handle Microsoft OAuth callback and exchange code for tokens."""
    try:
        # Get Microsoft OAuth configuration
        client_id = MICROSOFT_CLIENT_ID
        client_secret = MICROSOFT_CLIENT_SECRET
        tenant = MICROSOFT_TENANT
        
        # Exchange authorization code for access token
        token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
        
        token_data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "code": request.code,
            "redirect_uri": request.redirect_uri,
            "code_verifier": request.code_verifier,
            "grant_type": "authorization_code"
        }
        
        async with httpx.AsyncClient() as client:
            token_response = await client.post(token_url, data=token_data)
            
            if token_response.status_code != 200:
                return {"error": "Failed to exchange code for token", "details": token_response.text}
            
            token_info = token_response.json()
            access_token = token_info.get("access_token")
            
            if not access_token:
                return {"error": "No access token received"}
            
            # Get user information from Microsoft Graph
            graph_response = await client.get(
                "https://graph.microsoft.com/v1.0/me",
                headers={"Authorization": f"Bearer {access_token}"}
            )
            
            if graph_response.status_code != 200:
                return {"error": "Failed to get user information", "details": graph_response.text}
            
            user_info = graph_response.json()
            
            # Create user ID from Microsoft user ID
            user_id = user_info.get("id")
            user_name = user_info.get("displayName", "User")
            user_email = user_info.get("mail") or user_info.get("userPrincipalName", "")
            
            result = {
                "user_id": user_id,
                "name": user_name,
                "email": user_email,
                "access_token": access_token,
                "refresh_token": token_info.get("refresh_token", "")
            }
            
            return result
            
    except Exception as e:
        return {"error": f"OAuth callback failed: {str(e)}"}