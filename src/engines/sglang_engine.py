import os
import time
import threading
from typing import Dict, Any, Optional

import requests

from src.engines.base_engine import BaseEngine
from src.utils.docker_utils import (
    run_docker_command, 
    stream_container_logs,
    get_model_config_json,
    get_container_status,
)
from src.utils.utils import stream_process_output, get_environment_variables_from_command_line


class SGLangEngine(BaseEngine):
    """
    Implementation for SGLang inference engine.
    """
    
    engine_type = "sglang"
    DEFAULT_PORT = 30000
    DEFAULT_STARTUP_TIMEOUT = 12000  # Default maximum wait time in seconds for server to be ready
    
    def __init__(
        self,
        model: str,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        
        # Set default port and environment configuration
        self.port = self.DEFAULT_PORT
        self.hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        self.gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.startup_timeout = self.DEFAULT_STARTUP_TIMEOUT
        
        # Use HF_HOME environment variable (if exists) or default path as cache mount path
        self.hf_cache_dir = os.environ.get("HF_HOME", f"{os.path.expanduser('~')}/.cache/huggingface")
        
        # Store engine arguments separately from other config
        self.engine_args = {}
        self.env_vars = {}
        
        # Process kwargs to separate engine arguments from other config
        for key, value in kwargs.items():
            if key == "port":
                self.port = value
            self.engine_args[key] = value
            
    def start(self, stream_logs: bool = True) -> bool:
        """
        Start the SGLang engine in a Docker container.
        
        Args:
            stream_logs: Whether to stream container logs to the console
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        # Check if the Docker image exists, if not pull it with streaming output
        image_name = "lmsysorg/sglang:latest"
        check_image_cmd = ["docker", "image", "inspect", image_name]
        image_exists = run_docker_command(check_image_cmd)["success"]
        
        if not image_exists:
            print(f"\n📥 Docker image {image_name} not found locally. Starting download...")
            print("This may take several minutes for the first run. Please wait...\n")
            pull_cmd = ["docker", "pull", image_name]
            
            # Stream the pull process to show download progress
            return_code = stream_process_output(pull_cmd)
            
            if return_code != 0:
                print(f"\n❌ Failed to download Docker image {image_name}")
                return False
            
            print(f"\n✅ Successfully downloaded Docker image {image_name}\n")
        
        # Prepare docker command
        docker_cmd = [
            "docker", "run", "-d", "--name", f"{self.name}_arena_{os.urandom(4).hex()}",
        ]
        
        # Configure GPU allocation
        if self.gpu_devices:
            docker_cmd.extend([
                "--env", f"CUDA_VISIBLE_DEVICES={self.gpu_devices}",
                "--gpus", "all"
            ])
        else:
            # Default to all GPUs if not specified
            docker_cmd.extend(["--gpus", "all"])
            
        # Continue with the rest of the docker command
        docker_cmd.extend([
            "-v", f"{self.hf_cache_dir}:/root/.cache/huggingface",
            "-p", f"{self.port}:{self.port}",
            "--ipc=host",
        ])  
       
        
        # Add HF token if available
        if self.hf_token:
            docker_cmd.extend(["--env", f"HF_TOKEN={self.hf_token}"])
        
        # Get and add all SGLANG_ environment variables
        self.env_vars = get_environment_variables_from_command_line("SGL")
        print("env_vars in sglang_engine.py", self.env_vars)
        for env_name, env_value in self.env_vars.items():
            docker_cmd.extend(["--env", f"{env_name}={env_value}"])
        # Add image
        
        docker_cmd.append(image_name)
        
        # Add base command and model
        docker_cmd.extend([
            "python3", "-m", "sglang.launch_server",
            "--model-path", self.model,
            "--host", "0.0.0.0", # TODO: fix this later
        ])
        
        # Add engine arguments from config
        for arg_name, arg_value in self.engine_args.items():
            if arg_name not in ["model-path"]:  # Skip arguments already added
                if isinstance(arg_value, bool):
                    if arg_value:
                        docker_cmd.append(f"--{arg_name}")
                else:
                    docker_cmd.extend([f"--{arg_name}", str(arg_value)])
        
        # Run docker command
        result = run_docker_command(docker_cmd)
        if result["success"]:
            self.container_id = result["output"].strip()
            self.status = "starting"
            
            print(f"Container started with ID: {self.container_id}")
            print(f"Starting {self.name} with model: {self.model}")
            
            if stream_logs:
                # Start a separate thread to stream logs while we wait for the server to be ready
                stop_event = threading.Event()
                log_thread = threading.Thread(
                    target=stream_container_logs,
                    args=(self.container_id, stop_event)
                )
                log_thread.daemon = True
                log_thread.start()
            
            # Wait for the server to be ready
            print(f"Waiting up to {self.startup_timeout} seconds for server to be ready...")
            start_time = time.time()
            ready = False
            error_message = None
            
            try:
                while time.time() - start_time < self.startup_timeout:
                    # Check container status - only do this after container is confirmed to have started
                    container_status = get_container_status(self.container_id)
                    
                    # If the container exists but has changed to 'exited' status, it means startup has failed
                    if container_status == "exited":
                        error_message = f"Container exited unexpectedly. Check logs for details."
                        print(f"\n❌ {error_message}")
                        break
                    
                    if self._check_if_ready():
                        self.status = "running"
                        ready = True
                        break
                    time.sleep(2)
            finally:
                # Stop streaming logs if we were streaming
                if stream_logs:
                    stop_event.set()
                    log_thread.join(timeout=1.0)
            
            if ready:
                print(f"\n✅ {self.name} server is ready at {self.get_endpoint()}")
                return True
            else:
                # If we reached here, server didn't start in time or container exited
                error_message = f"{self.name} server failed to start"
                print(f"\n❌ {error_message}")
                self.stop()
                return False
        
        print("\n❌ Failed to start container")
        return False
    
    def stop(self) -> bool:
        """
        Stop the SGLang engine container.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        return super().stop()

    
    def get_endpoint(self) -> str:
        """
        Get the API endpoint URL for the SGLang engine.
        
        Returns:
            str: Endpoint URL
        """
        return f"http://localhost:{self.port}"
    
    def _check_if_ready(self) -> bool:

        """
        Check if the Sglang server is ready to accept requests.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.is_healthy()
    
    def is_healthy(self) -> bool:

        """Check if the Sglang engine is healthy by making a simple API call."""   
        try:
            response = requests.get(f"{self.get_endpoint()}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def get_version(self) -> str:
        """Get the version of the Sglang engine."""
        try:
            response = requests.get(f"{self.get_endpoint()}/get_server_info")
            return response.json()["version"]
        except requests.RequestException:
            return "Unknown"

    
    def get_full_engine_args(self) ->  Dict[str, Any]:
        """
        Get the model info of the SGLang engine.
        
        Process dtype values according to specified rules:
        1. If "dtype" is "half", convert it to "float16"
        2. If "dtype" is "auto", try to get the value from "torch_dtype" in config.json
        or use alternatives to determine dtype
        
        Returns:
            Dict[str, Any]: Dictionary containing engine arguments
        """
        try:
            response = requests.get(f"{self.get_endpoint()}/get_server_info")
            server_info = response.json()

            # Update converted values
            self.get_converted_dtype(server_info)
            self.get_converted_quantization(server_info)
            
            return server_info
        except requests.RequestException:
            return {
                "success": False
            }

    def get_converted_dtype(self, server_info: Dict[str, Any]) -> Optional[str]:
        """
        Extract and convert dtype from server info.
        
        Args:
            server_info: Dictionary containing server information
        """
        # Look for dtype in different possible locations
        dtype = None
        
        # Direct dtype field
        if "dtype" in server_info:
            dtype = server_info["dtype"]
        
        # Standardize dtype values if found
        if dtype:
            if dtype == "half":
                self.converted_dtype = "float16"
            elif dtype == "bfloat16":
                self.converted_dtype = "bfloat16"
            elif dtype == "float":
                self.converted_dtype = "float32"
            elif dtype == "auto":
                # Use a more specific value if available
                self.converted_dtype = self._determine_model_dtype()
            else:
                self.converted_dtype = dtype
        return self.converted_dtype

    
    def get_converted_quantization(self, server_info: Dict[str, Any]) -> Optional[str]:
        """
        Extract and convert quantization information from server info.
        
        Args:
            server_info: Dictionary containing server information
        """
        # Look for quantization in different possible locations
        quantization = None
        # Direct quantization field
        if "quantization" in server_info:
            quantization = server_info["quantization"]
        self.converted_quantization = quantization
        return self.converted_quantization

    
    def _determine_model_dtype(self) -> Optional[str]:
        """
        Try multiple approaches to determine the model's dtype.
        For SGLang:
        - When dtype="auto", use FP16 precision for FP32 and FP16 models
        - When dtype="auto", use BF16 precision for BF16 models
        
        Returns:
            Optional[str]: The determined dtype or None if unable to determine
        """
        config_json = get_model_config_json(self.container_id, self.model)
        if config_json and "torch_dtype" in config_json:
            # If torch_dtype exists in config.json, use that value
            torch_dtype = config_json["torch_dtype"]
            # If torch_dtype is a qualified name like torch.float16, remove torch. prefix
            if isinstance(torch_dtype, str) and torch_dtype.startswith("torch."):
                torch_dtype = torch_dtype.replace("torch.", "")
            
            # Apply SGLang's auto dtype rules
            if torch_dtype in ["float32", "float16"]:
                return "float16"
            elif torch_dtype == "bfloat16":
                return "bfloat16"
            
            return torch_dtype
       
        return None
    
   