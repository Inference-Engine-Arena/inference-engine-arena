"""
API Provider Wrappers

This module contains wrappers for various LLM API providers that add
logging and monitoring capabilities while maintaining the original APIs.

Available wrappers:
- Fireworks: Wrapper for Fireworks AI client
- Together: TODO - Wrapper for Together AI client  
- Cohere: TODO - Wrapper for Cohere AI client
"""

from .fireworks import Fireworks

# Future imports (when implemented):
# from .together import Together
# from .cohere import Client as CohereClient

__all__ = ['Fireworks'] 