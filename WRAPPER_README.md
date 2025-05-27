# iearena - API Wrappers with CSV Logging

This package provides drop-in replacements for various LLM API clients that automatically log usage data to CSV files.

## Available Wrappers

- **Fireworks**: Wrapper for Fireworks AI client ✅ 
- **Together**: Wrapper for Together AI client (TODO)
- **Cohere**: Wrapper for Cohere AI client (TODO)

## Installation

```bash
uv pip install -e .
```

## Usage - Fireworks

**Set your API key as an environment variable:**
```bash
export FIREWORKS_API_KEY="your-api-key-here"
```

Simply replace your import statement:

**Before:**
```python
from fireworks.client import Fireworks
```

**After:**
```python
from iearena import Fireworks
```

Everything else works exactly the same! The wrapper is completely transparent and maintains the same API.

## Example

```python
import os
from iearena import Fireworks

# Initialize client (same as before)
client = Fireworks(api_key=os.getenv("FIREWORKS_API_KEY"))

# Use the same API
response = client.completion.create(
    model="accounts/fireworks/models/llama4-scout-instruct-basic",
    prompt="What is the capital of France?",
    max_tokens=10,
    temperature=0.1
)

print(f"Response: {response.choices[0].text}")
# Usage data is automatically logged to 'fireworks_usage_log.csv'
```

## Features

- **Transparent**: Works exactly like the original Fireworks client
- **Automatic logging**: No code changes needed beyond the import
- **CSV output**: Logs to `fireworks_usage_log.csv` in the current directory
- **Streaming support**: Works with both streaming and non-streaming completions
- **Append mode**: New usage data is appended to existing CSV files

## CSV Format

The CSV file contains the following columns:

- `timestamp`: ISO format timestamp of the request
- `input_length`: Number of prompt tokens
- `output_length`: Number of completion tokens
- `prompt`: The actual prompt text sent to the model

## Example CSV Output

```csv
timestamp,input_length,output_length,prompt
2025-05-26T22:04:54.251253,12,50,Write a short story about a robot learning to paint:
2025-05-26T22:06:10.776823,8,10,What is the capital of France?
```

## Requirements

- Python 3.10+
- `fireworks-ai` package
- Valid Fireworks API key

The wrapper requires the original `fireworks-ai` package to be installed, as it acts as a transparent proxy that adds logging functionality. 