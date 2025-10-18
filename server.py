
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.endpoints import router as chat_router
import uvicorn

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,  # Enable credentials for OAuth
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Add a simple health check endpoint
@app.get("/")
async def root():
    return {"message": "CF Chatbot API is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "Server is running"}

app.include_router(chat_router)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)

