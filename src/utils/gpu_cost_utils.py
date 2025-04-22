from typing import Optional
import logging

logger = logging.getLogger(__name__)

# GPU hourly prices in USD
GPU_HOURLY_PRICES = {
    "NVIDIA A100-SXM4-40GB": 1.50,  # A100 80GB
    "NVIDIA A100-SXM4-80GB": 2.00,  # A100 80GB
    "NVIDIA H100 80GB HBM3": 4.00,  # H100 80GB
    # "NVIDIA H200-SXM5-141GB": 4.50,  # H200 141GB
}

def calculate_cost_per_million_tokens(
    gpu_type: str,
    num_gpus: int = 1,
    throughput_tokens_per_second: float = 1000,
) -> Optional[float]:
    """
    Calculate the cost per million tokens based on GPU type and throughput.
    
    Args:
        gpu_type: Type of GPU (A100, H100, or H200)
        throughput_tokens_per_second: Number of tokens processed per second
        num_gpus: Number of GPUs being used (default: 1)
        
    Returns:
        float: Cost per million tokens in USD, or None if GPU type is not supported
    """
    if gpu_type not in GPU_HOURLY_PRICES:
        logger.error(f"Unsupported GPU type: {gpu_type}")
        return None
        
    hourly_price = GPU_HOURLY_PRICES[gpu_type]
    total_hourly_cost = hourly_price * num_gpus
    
    # Calculate tokens per hour
    tokens_per_hour = throughput_tokens_per_second * 3600
    
    # Calculate cost per million tokens
    cost_per_million_tokens = (total_hourly_cost / tokens_per_hour) * 1_000_000
    
    return cost_per_million_tokens

def get_gpu_hourly_price(gpu_type: str) -> Optional[float]:
    """
    Get the hourly price for a specific GPU type.
    
    Args:
        gpu_type: Type of GPU (A100, H100, or H200)
        
    Returns:
        float: Hourly price in USD, or None if GPU type is not supported
    """
    return GPU_HOURLY_PRICES.get(gpu_type)

def get_supported_gpu_types() -> list:
    """
    Get a list of supported GPU types.
    
    Returns:
        list: List of supported GPU type names
    """
    return list(GPU_HOURLY_PRICES.keys()) 