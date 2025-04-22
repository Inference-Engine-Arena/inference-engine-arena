import os
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from pathlib import Path
from dotenv import load_dotenv
import logging
from datetime import datetime
from src.utils.utils import get_project_version

# Load environment variables and set up logging
load_dotenv()
logger = logging.getLogger(__name__)

# MongoDB connection
MONGO_URI = os.environ.get('MONGO_URI')
DB_NAME = os.environ.get('DB_NAME')
PROJECT_VERSION = get_project_version()

class DatabaseService:
    def __init__(self):
        self.client = None
        self.db = None
        self.users = None
        self.json_data = None
        self.connected = False
        
        try:
            self.client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[DB_NAME]
            self.users = self.db.users
            self.json_data = self.db.json_data
            
            # Create unique indexes
            self.users.create_index('github_id', unique=True)
            self.json_data.create_index('filename', unique=True)
            
            self.connected = True
        except Exception as e:
            logger.error(f"MongoDB connection failed: {str(e)}")
        
    def store_user(self, user_data):
        """Store user auth data in MongoDB"""
        if not self.connected:
            return False
            
        try:
            user_id = user_data['user']['id']
            self.users.update_one(
                {'github_id': user_id},
                {'$set': {
                    'github_id': user_id,
                    'key': user_data['key'],
                    'login': user_data['user']['login'],
                    'name': user_data['user']['name'],
                    'email': user_data['user']['email'],
                    'client_id': user_data['client_id'],
                    'access_token': user_data['access_token'],
                    'created_at': datetime.now(),
                    'avatar_url': user_data['user'].get('avatar_url'),
                    'updated_at': datetime.now()
                }},
                upsert=True
            )
            return True
        except Exception:
            return False
        
    def verify_token(self, client_id):
        """Verify token exists in MongoDB"""
        if not self.connected:
            return False
            
        try:
            return self.users.find_one({'client_id': client_id}) is not None
        except Exception:
            return False
    
    def verify_token_match(self, token_data):
        """Verify local token matches remote token"""
        if not self.connected:
            return False
            
        try:
            client_id = token_data.get('client_id')
            access_token = token_data.get('access_token')
            user_id = token_data.get('user', {}).get('id')
            
            if not all([client_id, access_token, user_id]):
                return False
                
            user = self.users.find_one({'client_id': client_id})
            if not user:
                return False
                
            if user.get('access_token') != access_token:
                return False
                
            if str(user.get('github_id')) != str(user_id):
                return False
                
            return True
        except Exception:
            return False
            
    def upload_json_data(self, client_id, filename, json_data):
        """Upload JSON data directly to MongoDB without requiring a file"""
        if not self.connected:
            return False, "MongoDB connection required but not available"
            
        if client_id and not self.verify_token(client_id):
            return False, "Invalid authentication token"
            
        try:
            # Insert the JSON data directly
            self.json_data.insert_one({
                'filename': filename,
                'client_id': client_id,
                'data': json_data,
                'upload_datetime': datetime.now(),
                'version': PROJECT_VERSION
            })
            
            return True, "JSON upload successful"
                
        except Exception as e:
            return False, f"Error: {str(e)}" 