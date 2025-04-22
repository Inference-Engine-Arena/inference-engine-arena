import os
from fastapi import FastAPI, Request, Depends, HTTPException, status, Form, Response
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict
import httpx
from pydantic import BaseModel
from pathlib import Path
import uuid
from dotenv import load_dotenv
import gradio as gr
import uvicorn
import sys
import logging

# Add parent directory to path to import modules from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db_service import DatabaseService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Load environment variables
load_dotenv()

# Create FastAPI app
app = FastAPI(title="Inference Engine API", description="API for authentication and file uploads")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# GitHub OAuth configuration
GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET")
REDIRECT_URL = os.environ.get("REDIRECT_URL")


# MongoDB service
db_service = DatabaseService()

# Dictionary to store pending uploads (state -> upload data)
pending_uploads: Dict[str, dict] = {}

# Models
class UploadRequest(BaseModel):
    login: bool = False
    key: Optional[str] = None
    data: dict
    filename: Optional[str] = None

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/auth/callback")
async def auth_callback(code: str, state: Optional[str] = None):
    """Handle GitHub OAuth callback and exchange code for token"""
    if not code:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No authorization code provided"
        )
    
    # Exchange code for access token
    token_response = await exchange_code_for_token(code)
    if "error" in token_response:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=token_response.get("error_description", "Failed to get access token")
        )
    
    # Get user info with the access token
    access_token = token_response["access_token"]
    user_info = await get_user_info(access_token)
    
    # Generate client ID if new user, otherwise use existing one
    client_id = None
    
    # Check if user exists already by GitHub ID
    if user_info.get('id'):
        existing_user = db_service.users.find_one({"github_id": str(user_info.get('id'))})
        if existing_user:
            client_id = existing_user.get('client_id')
    if not client_id:
        # Generate a new client ID if no GitHub ID is provided
        client_id = str(uuid.uuid4())
    
    
    # If we have pending upload data (from state parameter)
    if state and state in pending_uploads:
        upload_data = pending_uploads[state]
        # Prepare user data for storage
        user_data = {
            "user": user_info,
            "client_id": client_id,
            "access_token": access_token,
            "key": upload_data["key"]
        }
    
    # Use the store_user method from db_service
        db_service.store_user(user_data)
        
        
        # Get filename from stored data or generate one
        filename = upload_data.get('filename') if isinstance(upload_data, dict) else None
        if not filename:
            filename = f"upload_{uuid.uuid4()}.json"
        
        # Get the actual data to upload
        json_data = upload_data.get('data', upload_data) if isinstance(upload_data, dict) else upload_data
        
        # Upload to database with the newly obtained client_id
        success, message = db_service.upload_json_data(client_id, filename, json_data)
        
        # Clean up the pending upload
        del pending_uploads[state]
        
        if success:
            return JSONResponse(content={
                "success": True,
                "message": message,
                "client_id": client_id,
                "filename": filename
            })
        else:
            return JSONResponse(content={
                "success": False,
                "message": message,
                "client_id": client_id,
                "filename": filename
            })
    
    # If no pending upload, just return client_id for future reference
    return JSONResponse(
        content={"client_id": client_id, "message": "Authentication successful"}
    )

@app.post("/upload")
async def upload_data(request: UploadRequest):
    """Upload data with or without authentication"""
    client_id = None

    # Handle different authentication scenarios
    if request.login:
        # First check if key is provided
        if not request.key:
            # No key provided with login=true, return error
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "success": False,
                    "message": "A key is required when login=true",
                    "auth_required": False
                }
            )
        
        # Key is provided, validate it
        try:
            # Use the client-provided key as client_id directly
            user = db_service.users.find_one({"key": request.key})
            
            if user:
                logger.info(f"User found: {user}")
                # Key is valid, use the client_id
                client_id = user["client_id"]
            else:
                logger.info(f"User not found for key: {request.key}")
                # Key is invalid, start OAuth process
                if not GITHUB_CLIENT_ID:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="GitHub OAuth credentials not configured"
                    )
                
                # Generate a unique state to track this upload after OAuth
                state = str(uuid.uuid4())
                
                # Store the upload data and filename temporarily
                pending_uploads[state] = {
                    'data': request.data,
                    'filename': request.filename,
                    'key': request.key  # Store original key for reference
                }
                
                # Build GitHub authorization URL with state
                github_auth_url = (
                    f"https://github.com/login/oauth/authorize"
                    f"?client_id={GITHUB_CLIENT_ID}"
                    f"&redirect_uri={REDIRECT_URL}"
                    f"&scope=user:email"
                    f"&state={state}"
                )
                logger.info(f"Redirecting to GitHub for OAuth: {github_auth_url}")
                return RedirectResponse(url=github_auth_url)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Database error: {str(e)}"
            )
    
    # Use provided filename or generate a unique one
    filename = request.filename or f"upload_{uuid.uuid4()}.json"
    
    # Create JSON file path (no need to actually create the file)
    json_data = request.data
    
    # Upload to database with the filename
    success, message = db_service.upload_json_data(client_id, filename, json_data)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=message
        )
    
    return {"success": True, "message": message, "filename": filename}

# Utility functions
async def exchange_code_for_token(code: str):
    """Exchange GitHub code for access token"""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://github.com/login/oauth/access_token",
            data={
                "client_id": GITHUB_CLIENT_ID,
                "client_secret": GITHUB_CLIENT_SECRET,
                "code": code,
                "redirect_uri": REDIRECT_URL
            },
            headers={"Accept": "application/json"}
        )
        
        return response.json()

async def get_user_info(access_token: str):
    """Get GitHub user information using access token"""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"https://api.github.com/user",
            headers={
                "Authorization": f"token {access_token}",
                "Accept": "application/json"
            }
        )
        
        return response.json()

# Import the Gradio app creator function
from leaderboard.app import create_interface

# Create Gradio interface
demo = create_interface()

# Set favicon path
favicon_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "public", "images", "logo", "arena.png")

# Mount Gradio app to FastAPI at the root path
app = gr.mount_gradio_app(app, demo, path="", favicon_path=favicon_path, app_kwargs={"title": "Inference Engine Arena"})

@app.get("/api")
async def api_root():
    return {"message": "Inference Engine API is running. API endpoints available at /api/*"}

def run_server():
    """Run the combined FastAPI and Gradio server"""
    host = "0.0.0.0"
    port = int(os.environ.get("PORT", 8080))
    logger.info(f"Starting combined server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    run_server()
