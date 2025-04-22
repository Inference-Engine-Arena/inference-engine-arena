from abc import ABC, abstractmethod
from typing import Dict, Any

from src.utils.docker_utils import get_gpu_info, run_docker_command


class BaseEngine(ABC):
    """
    Abstract base class for inference engines.
    All engine implementations should inherit from this class.
    """
    
    # Default engine type, should be overridden by subclasses
    engine_type = "base"
    
    def __init__(self, model: str, **kwargs):
        """
        Initialize a new engine instance.
        
        Args:
            model: Model identifier (e.g., "NousResearch/Meta-Llama-3-8B-Instruct")
            **kwargs: Additional engine-specific configuration options
        """
        self.name = self.engine_type  # Set the name equal to the engine type
        self.model = model
        self.status = "stopped"
        self.container_id = None
        self.port = None
        self.converted_dtype = None
        self.converted_quantization = None
        
    @abstractmethod
    def start(self, stream_logs: bool = True) -> bool:
        """
        Start the engine in a Docker container.
        
        Args:
            stream_logs: Whether to stream container logs to the console
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        pass
       
    def stop(self) -> bool:
        """
        Stop the engine.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
       
        if not self.container_id:
            return True
            
        cmd = ["docker", "stop", self.container_id]
        result = run_docker_command(cmd)
        
        if result["success"]:
            cmd = ["docker", "rm", "-f", self.container_id]
            result = run_docker_command(cmd)
            if result["success"]:
                self.status = "stopped"
                return True
        return False
    
    @abstractmethod
    def get_endpoint(self) -> str:
        """
        Get the endpoint URL for making requests to this engine.
        
        Returns:
            str: Full URL endpoint
        """
        pass
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information for this engine's container.
        
        Returns:
            Dict[str, Any]: GPU information
        """
        if not self.container_id or self.status != "running":
            return {"success": False, "error": "Engine not running"}
        
        return get_gpu_info(container_id=self.container_id, engine_args=self.engine_args)
    
    def sync_status(self) -> Dict[str, Any]:
        """
        ignore this method for now
        """

    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert engine information to a dictionary for serialization.
        
        Returns:
            Dict[str, Any]: Engine information
        """
        
        result = {
            "name": self.name,
            "type": self.engine_type,
            "model": self.model,
            "status": self.status,
            "container_id": self.container_id,
            "converted_dtype": self.converted_dtype,
            "converted_quantization": self.converted_quantization
        }
        
        # Add engine arguments if available
        if hasattr(self, 'engine_args'):
            result["engine_args"] = self.engine_args
            
        # Add environment variables if available
        if hasattr(self, 'env_vars'):
            result["env_vars"] = self.env_vars
        
        # Add endpoint if running
        if self.status == "running":
            # Get all engine args and env vars if method exists
            result["full_engine_args"] = self.full_engine_args
            result["full_env_vars"] = self.full_env_vars
            result["version"] = self.get_version()
            result["endpoint"] = self.get_endpoint()
            
            # Add converted_dtype and converted_quantization if available
            if self.converted_dtype is not None:
                result["converted_dtype"] = self.converted_dtype
            if self.converted_quantization is not None:
                result["converted_quantization"] = self.converted_quantization
            
            # Add GPU info if running
            gpu_info = self.get_gpu_info()
            if gpu_info.get("success", False):
                result["gpu_info"] = gpu_info
        
        return result
    
    def print_status(self) -> None:
        """
        Print the status(to_dict()) of the vLLM engine.
        """
        print(self.to_dict())