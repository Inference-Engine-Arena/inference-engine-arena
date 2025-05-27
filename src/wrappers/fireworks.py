#!/usr/bin/env python3
"""
Fireworks Client Wrapper with CSV Logging

This module provides a wrapper around the Fireworks client that logs
usage trace (timestamp, input/output lengths, prompts) to a CSV file.
"""

import csv
import os
from datetime import datetime
from typing import Any, Optional
from fireworks.client import Fireworks as OriginalFireworks


class Fireworks:
    """
    Wrapper around the original Fireworks client that logs usage trace to CSV.
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """Initialize the wrapper with the original Fireworks client."""
        self._client = OriginalFireworks(api_key=api_key, **kwargs)
        self._csv_file = "fireworks_usage_trace.csv"
        self._ensure_csv_header()
    
    def _ensure_csv_header(self):
        """Ensure the CSV file exists with proper headers."""
        if not os.path.exists(self._csv_file):
            with open(self._csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['timestamp', 'input_length', 'output_length', 'prompt'])
    
    def _log_usage(self, prompt: str, usage: Any):
        """Log usage data to CSV file."""
        timestamp = datetime.now().isoformat()
        input_length = getattr(usage, 'prompt_tokens', 0)
        output_length = getattr(usage, 'completion_tokens', 0)
        
        with open(self._csv_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, input_length, output_length, prompt])
    
    @property
    def completion(self):
        """Return a wrapped completion object."""
        return CompletionWrapper(self._client.completion, self._log_usage)
    
    def __getattr__(self, name):
        """Delegate any other attributes to the original client."""
        return getattr(self._client, name)


class CompletionWrapper:
    """Wrapper for the completion API that logs usage data."""
    
    def __init__(self, original_completion, log_function):
        self._completion = original_completion
        self._log_usage = log_function
    
    def create(self, prompt: str, stream: bool = False, **kwargs):
        """Create a completion and log usage data."""
        if stream:
            return self._create_streaming(prompt, **kwargs)
        else:
            return self._create_non_streaming(prompt, **kwargs)
    
    def _create_non_streaming(self, prompt: str, **kwargs):
        """Handle non-streaming completion."""
        response = self._completion.create(prompt=prompt, stream=False, **kwargs)
        
        # Log usage data
        if hasattr(response, 'usage') and response.usage:
            self._log_usage(prompt, response.usage)
        
        return response
    
    def _create_streaming(self, prompt: str, **kwargs):
        """Handle streaming completion."""
        response_generator = self._completion.create(prompt=prompt, stream=True, **kwargs)
        
        # Wrap the generator to capture usage data from the final chunk
        def wrapped_generator():
            usage_logged = False
            for chunk in response_generator:
                yield chunk
                
                # Check if this is the final chunk with usage info
                if (chunk.choices[0].finish_reason is not None and 
                    hasattr(chunk, 'usage') and chunk.usage and not usage_logged):
                    self._log_usage(prompt, chunk.usage)
                    usage_logged = True
        
        return wrapped_generator()
    
    def __getattr__(self, name):
        """Delegate any other attributes to the original completion object."""
        return getattr(self._completion, name) 