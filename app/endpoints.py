from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel
import uuid
import httpx
import os
import json
import asyncio
import re

from app.llm import setup_qa_chain
from app.vectorstore import retriever, vectorstore
from app.memory import add_to_conversation, get_conversation_context, get_user_chat_history, clear_user_chat_history
from app.helpers import strip_markdown, preserve_markdown
from config import SYSTEM_PROMPT, MICROSOFT_CLIENT_ID, MICROSOFT_CLIENT_SECRET, MICROSOFT_TENANT


router = APIRouter()

qa_chain = setup_qa_chain(retriever)

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

@router.post("/chat")
async def chat(request: Request):
    """Chat endpoint: returns full answer from vectorstore."""
    data = await request.json()
    question = data.get("question", "")
    user_id = data.get("user_id")
    session_id = data.get("session_id", str(uuid.uuid4()))

    # Use user_id if provided, otherwise fall back to session_id for backward compatibility
    conversation_id = user_id if user_id else session_id

    # DEBUG: Print question and answer
    print(f"\n QUESTION: {question}")

    # Check if this is a conversational query
    if is_conversational_query(question):
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
        conversation_context = get_conversation_context(conversation_id)
        enhanced_query = f"{conversation_context}\n\nUser: {question}" if conversation_context else question
        
        chain = conversational_prompt | llm
        result = chain.invoke({"question": enhanced_query})
        answer = result.content
    else:
        # Handle informational queries with document retrieval
        conversation_context = get_conversation_context(conversation_id)
        enhanced_query = f"{conversation_context}\n\nUser: {question}" if conversation_context else question
        result = qa_chain.invoke({"query": enhanced_query})
        answer = result["result"]

    # DEBUG: Print bot response
    print(f" ANSWER: {answer}\n")

    # Add both user question and bot response to conversation AFTER processing
    add_to_conversation(conversation_id, "user", question)
    add_to_conversation(conversation_id, "assistant", answer)

    clean_answer = preserve_markdown(answer)
    return {"answer": clean_answer, "user_id": user_id, "session_id": session_id}

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
            # Get conversation context BEFORE adding current question
            conversation_context = get_conversation_context(conversation_id)

            # Combine question with conversation context for better continuity
            enhanced_query = f"{conversation_context}\n\nUser: {question}" if conversation_context else question
            
            # Check if this is a conversational query
            if is_conversational_query(question):
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
                
                
                # Send completion signal
                yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"
                
                # Add to conversation
                add_to_conversation(conversation_id, "user", question)
                add_to_conversation(conversation_id, "assistant", full_response)
                return
            
            # PHASE 1: THINKING - Document retrieval and processing
            # This happens while the frontend shows "Thinking..." animation
            
            # HYBRID SEARCH: Combine semantic search with targeted local document search
            local_docs = []
            blog_docs = []
            
            # Step 1: Direct semantic search for all documents with much broader scope
            all_docs = vectorstore.similarity_search(enhanced_query, k=100)  # Increased from 30 to 100
            
            # Step 2: Separate local documents from blog content
            for doc in all_docs:
                source_type = doc.metadata.get('source_type', '')
                if source_type in ['pdf', 'excel', 'doc']:
                    local_docs.append(doc)
                else:
                    blog_docs.append(doc)
            
            
            # Step 3: If no local docs found, try targeted searches for local documents
            if not local_docs:
                
                # Try different search strategies for local documents
                targeted_searches = [
                    enhanced_query,  # Original query
                    enhanced_query.lower(),  # Lowercase
                    enhanced_query.split()[0] if ' ' in enhanced_query else enhanced_query,  # First word
                    enhanced_query.replace(' ', '_'),  # Underscore version
                ]
                
                for search_term in targeted_searches:
                    if not search_term.strip():
                        continue
                    
                    try:
                        # Search with a broader scope for local documents
                        search_results = vectorstore.similarity_search(search_term, k=50)
                        
                        for doc in search_results:
                            source_type = doc.metadata.get('source_type', '')
                            if source_type in ['pdf', 'excel', 'doc'] and doc not in local_docs:
                                local_docs.append(doc)
                                
                        # If we found some local docs, we can stop
                        if len(local_docs) >= 5:
                            break
                            
                    except Exception as e:
                        continue
            
            # Step 4: Prioritize local documents: take up to 10 local docs, then fill with blog docs
            relevant_docs = local_docs[:10]  # Prioritize local documents
            remaining_slots = 15 - len(relevant_docs)
            relevant_docs.extend(blog_docs[:remaining_slots])  # Fill remaining with blog content
            
            
            # Secondary semantic search with query rephrasing for better coverage
            try:
                # Use the LLM to create a semantic variation of the query
                rephrase_prompt = f"""
                Rephrase this question in 2-3 different ways to help find relevant information:
                Original: {enhanced_query}
                
                Provide 2-3 alternative phrasings that mean the same thing but use different words.
                Each rephrasing should be on a new line and be concise.
                """
                
                # Get LLM for rephrasing
                from langchain_openai import ChatOpenAI
                rephrase_llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
                rephrase_result = rephrase_llm.invoke(rephrase_prompt)
                rephrased_queries = [line.strip() for line in rephrase_result.content.split('\n') if line.strip()]
                
                # Search with each rephrased query, but still prioritize local docs
                for rephrased_query in rephrased_queries[:2]:  # Limit to 2 rephrasings
                    additional_all = vectorstore.similarity_search(rephrased_query, k=20)
                    additional_local = [doc for doc in additional_all if doc.metadata.get('source_type', '') in ['pdf', 'excel', 'doc']]
                    additional_blog = [doc for doc in additional_all if doc.metadata.get('source_type', '') not in ['pdf', 'excel', 'doc']]
                    
                    # Add local docs first, then blog docs
                    relevant_docs.extend(additional_local[:5])
                    relevant_docs.extend(additional_blog[:5])
                    
            except Exception as e:
                # If rephrasing fails, continue with original query only
                pass
            
            # Deduplicate documents while preserving relevance order
            seen_ids = set()
            unique_docs = []
            
            for doc in relevant_docs:
                # Create a unique identifier for the document
                doc_id = f"{doc.metadata.get('source', '')}_{doc.page_content[:50]}"
                if doc_id not in seen_ids:
                    seen_ids.add(doc_id)
                    unique_docs.append(doc)
            
            # Limit to reasonable number of documents for processing
            final_docs = unique_docs[:20]
            
            # Format the documents properly
            context_text = "\n\n".join([f"Document {i+1}:\n{doc.page_content}" for i, doc in enumerate(final_docs)])
            
            # Send signal that thinking is complete and streaming will start
            yield f"data: {json.dumps({'type': 'thinking_complete'})}\n\n"
            
            # PHASE 2: STREAMING - Generate and stream response
            # This happens after the frontend clears the "Thinking..." animation
            
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
            
            
            # Send completion signal
            yield f"data: {json.dumps({'type': 'done', 'full_response': full_response})}\n\n"
            
            # Add both user question and bot response to conversation AFTER processing
            add_to_conversation(conversation_id, "user", question)
            add_to_conversation(conversation_id, "assistant", full_response)
            
        except Exception as e:
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
        history = get_user_chat_history(user_id)
        return {"user_id": user_id, "history": history}
    except Exception as e:
        return {"error": str(e)}

@router.delete("/chat/history/{user_id}")
async def clear_chat_history(user_id: str):
    """Clear chat history for a specific user."""
    try:
        clear_user_chat_history(user_id)
        return {"message": f"Chat history cleared for user {user_id}"}
    except Exception as e:
        return {"error": str(e)}

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