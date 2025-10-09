
import asyncio
from langchain.callbacks.base import BaseCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from config import SYSTEM_PROMPT


class AsyncStreamHandler(BaseCallbackHandler):
    def __init__(self):
        self.queue = asyncio.Queue()

    def on_llm_new_token(self, token: str, **kwargs):
        self.queue.put_nowait(token)

def setup_qa_chain(retriever):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain.chains.combine_documents import create_stuff_documents_chain
    
    # Create a custom prompt template with system message
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Context: {context}\n\nQuestion: {question}")
    ])
    
    llm = ChatOpenAI(
        model_name="gpt-4.1-nano", 
        streaming=True, 
        temperature=0.3,  # Balanced temperature for good responses
        max_tokens=1500   # Allow longer responses for comprehensive answers
    )
    
    # Create the document chain using the modern approach
    document_chain = create_stuff_documents_chain(llm, prompt_template)
    
    # Create a wrapper to maintain compatibility with the existing interface
    class SemanticRetrievalQA:
        def __init__(self, document_chain, retriever):
            self.document_chain = document_chain
            self.retriever = retriever
        
        def invoke(self, inputs):
            # Extract the query from the inputs dict
            query = inputs.get("query", "")
            
            # Get relevant documents using pure semantic search
            from app.vectorstore import vectorstore
            
            # PURE SEMANTIC SEARCH - Let the vectorstore handle semantic understanding
            # No predefined keywords, no hardcoded terms, no forced inclusions
            
            # Primary semantic search with the original query
            relevant_docs = vectorstore.similarity_search(query, k=15)
            
            # Secondary semantic search with query rephrasing for better coverage
            # This helps catch semantically similar but differently worded content
            try:
                # Use the LLM to create a semantic variation of the query
                rephrase_prompt = f"""
                Rephrase this question in 2-3 different ways to help find relevant information:
                Original: {query}
                
                Provide 2-3 alternative phrasings that mean the same thing but use different words.
                Each rephrasing should be on a new line and be concise.
                """
                
                rephrase_result = llm.invoke(rephrase_prompt)
                rephrased_queries = [line.strip() for line in rephrase_result.content.split('\n') if line.strip()]
                
                # Search with each rephrased query
                for rephrased_query in rephrased_queries[:2]:  # Limit to 2 rephrasings
                    additional_docs = vectorstore.similarity_search(rephrased_query, k=8)
                    relevant_docs.extend(additional_docs)
                    
            except Exception as e:
                # If rephrasing fails, continue with original query only
                print(f"Query rephrasing failed: {e}")
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
            # Too many documents can overwhelm the LLM and reduce quality
            final_docs = unique_docs[:20]
            
            # Invoke the document chain with the semantically relevant documents
            result = self.document_chain.invoke({
                "context": final_docs,
                "question": query
            })
            
            return {"result": result}
    
    qa_chain = SemanticRetrievalQA(document_chain, retriever)
    return qa_chain
