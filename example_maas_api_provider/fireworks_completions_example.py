#!/usr/bin/env python3
"""
Simple Fireworks Completions API Example

This script demonstrates how to use the Fireworks/Iearena Completions API
to generate text completions using various models and parameters.

The iearena import is a wrapper around the Fireworks API that automatically logs usage trace to a CSV file.
"""

import os
# original Fireworks import
# from fireworks.client import Fireworks

# iearena import functions the same as the original Fireworks import, but also logs usage trace to 
# a CSV file which contains timestamp, input_length, output_length, prompt
from iearena import Fireworks # type: ignore

def main():
    # Initialize the Fireworks client
    api_key = os.getenv("FIREWORKS_API_KEY")
    if not api_key:
        print("Error: Please set your FIREWORKS_API_KEY environment variable")
        print("You can get your API key from: https://fireworks.ai/")
        return
    
    client = Fireworks(api_key=api_key)
    
    # Example 1: Basic completion
    print("=== Basic Completion Example ===")
    try:
        response = client.completion.create(
            model="accounts/fireworks/models/llama4-scout-instruct-basic",
            prompt="Write a short story about a robot learning to paint:",
            max_tokens=50,
            temperature=0.7
        )
        print(f"Response: {response.choices[0].text}")
        print(f"Usage: {response.usage}")
        print()
    except Exception as e:
        print(f"Error in basic completion: {e}")
        return
    
    # Example 2: Streaming completion
    print("=== Streaming Completion Example ===")
    try:
        response_generator = client.completion.create(
            model="accounts/fireworks/models/llama4-scout-instruct-basic",
            prompt="Write a short story about a robot learning to paint:",
            max_tokens=50,
            temperature=0.8,
            stream=True
        )
        usage_info = None
        
        for chunk in response_generator:
            if chunk.choices[0].text:
                print(chunk.choices[0].text, end="", flush=True)
            
            if chunk.choices[0].finish_reason is not None:
                # According to Fireworks docs, usage is in the last chunk
                if hasattr(chunk, 'usage') and chunk.usage:
                    usage_info = chunk.usage
                    print()
                    print(f"Usage: {usage_info}")
        
        if not usage_info:
            print(f"\n\n⚠️  No usage info found in final chunk")
            print("   This might indicate the response was truncated or an error occurred")
            
    except Exception as e:
        print(f"Error in streaming completion: {e}")

if __name__ == "__main__":
    print("Fireworks Completions API Example")
    print("=" * 40)
    print()
    
    main() 