#!/usr/bin/env python3
"""
API method for Inference Engine API
Provides functions to upload data to the inference engine with authentication handling
"""

import requests
import json
import os
import webbrowser
from pathlib import Path
import uuid
from dotenv import load_dotenv

load_dotenv()

# Configuration
API_BASE_URL = os.getenv('API_BASE_URL')

AUTH_DIR = Path(__file__).parent.parent.parent / '.global_auth'
TOKEN_FILE = AUTH_DIR / 'auth_token.json'


def get_stored_token():
    """
    Try to load previously stored access token
    First from file, then fallback to hardcoded token
    """
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, 'r') as file:
            token = file.read().strip()
            if token:
                return token
    else:
        # Create the directory if it doesn't exist
        AUTH_DIR.mkdir(parents=True, exist_ok=True)
        # Generate a new token
        new_token=str(uuid.uuid4())
        with open(TOKEN_FILE, 'w') as file:
            file.write(new_token)
        return new_token



def upload_json_file(filepath,login=True):
    """
    Upload data to the inference engine API
    
    Args:
        filepath (str): Path to the file
        filedata (dict): Data to upload
        login (bool): Whether to do upload anonymously or with authentication
                
    Returns:
        dict: The response from the API
    """
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        return False, f"File not found: {filepath}"
        
    with open(path, 'r') as f:
        json_data = json.load(f)
    
    # Check if we have a stored token
    if login:
        access_token = get_stored_token()
        payload = {
            "data": json_data,
            "filename": path.name,
            "login": True,
            "key": access_token
        }
    else:
        payload = {
            "data": json_data,
            "filename": path.name,
            "login": False
        }
    # Make the API request with redirect handling
    response = requests.post(
        f"{API_BASE_URL}/upload",
        json=payload,
        allow_redirects=False  # Don't follow redirects automatically for OAuth flow
    )
    
    # Check if we got a redirect (OAuth flow) or a successful response
    if response.status_code == 200:
        return True, "JSON upload successful"
    elif response.status_code == 307:  # Temporary redirect for OAuth flow
        try:
            redirect_url = response.headers['Location']
            print(f"\nGitHub authentication required.")
            print(f"GitHub Auth URL: {redirect_url}")
            # Always display the URL for manual copying
            print("Please copy this URL and paste it into your browser if automatic opening fails:")
            print(redirect_url)
            # Try to open browser but handle potential failures
            print("Attempting to open browser...")
            browser_opened = webbrowser.open(redirect_url)
            if not browser_opened:
                print("Failed to open browser automatically. Please use the URL above.")
            else:
                print("Please complete the GitHub authentication in your browser.")
                print("The API will automatically process your upload after successful authentication.")
            return True, "GitHub authentication required, and post uploading will be done automatically after successful authentication"
        except Exception as e:
            print(f"Error during authentication process: {e}")
            if 'redirect_url' in locals():
                print("Please copy this URL and paste it into your browser:")
                print(redirect_url)
            return False, f"Error: {e}"
    else:
        return False, f"Error: Received status code {response.status_code}"