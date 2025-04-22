import os
import json
import time
import threading
from typing import Dict, Any, Optional

import requests

from src.engines.base_engine import BaseEngine
from src.utils.docker_utils import (
    run_docker_command, 
    get_container_logs,
    stream_container_logs,
)
from src.utils.utils import stream_process_output, get_environment_variables_from_command_line



class VLLMEngine(BaseEngine):
    """
    Implementation for vLLM inference engine.
    """
    
    engine_type = "vllm"
    DEFAULT_PORT = 8000
    DEFAULT_STARTUP_TIMEOUT = 12000  # Default maximum wait time in seconds for server to be ready
    
    def __init__(
        self,
        model: str,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        
        self.port = self.DEFAULT_PORT
        self.hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        self.gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES") 
        self.startup_timeout = self.DEFAULT_STARTUP_TIMEOUT
        
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
        Start the vLLM engine in a Docker container.
        
        Args:
            stream_logs: Whether to stream container logs to the console
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        # Check if the Docker image exists, if not pull it with streaming output
        image_name = "vllm/vllm-openai:latest"
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
            "-v", f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
            "-p", f"{self.port}:{self.port}",
            "--ipc=host",
        ])

        # Add HF token if available
        if self.hf_token:
            docker_cmd.extend(["--env", f"HUGGING_FACE_HUB_TOKEN={self.hf_token}"])
        
        # Get and add all VLLM_ environment variables
        self.env_vars = get_environment_variables_from_command_line("VLLM")
        print("env_vars in vllm_engine.py", self.env_vars)
        for env_name, env_value in self.env_vars.items():
            docker_cmd.extend(["--env", f"{env_name}={env_value}"])
            
        # Add image
        docker_cmd.append(image_name)
        
        # Start with the model as the first argument
        docker_cmd.extend(["--model", self.model])
        

        print("engine_args in vllm_engine.py", self.engine_args)
        # Add engine arguments from config
        for arg_name, arg_value in self.engine_args.items():
            if isinstance(arg_value, bool):
                if arg_value:
                    docker_cmd.append(f"--{arg_name}")
            else:
                docker_cmd.extend([f"--{arg_name}", str(arg_value)])
        

        print("docker_cmd in vllm_engine.py", docker_cmd)

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
            
            try:
                while time.time() - start_time < self.startup_timeout:
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
                # If we reached here, server didn't start in time
                print(f"\n❌ {self.name} server failed to start within the timeout period")
                self.stop()
                return False
        
        print("\n❌ Failed to start container")
        return False
    
    def stop(self) -> bool:
        """
        Stop the vLLM engine container.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        return super().stop()

    
    def get_endpoint(self) -> str:
        """
        Get the OpenAI-compatible endpoint URL for the vLLM engine.
        
        Returns:
            str: Endpoint URL
        """
        return f"http://localhost:{self.port}"
    
    def _check_if_ready(self) -> bool:
        """
        Check if the vLLM server is ready to accept requests.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.is_healthy()
    
    def is_healthy(self) -> bool:
        """Check if the VLLM engine is healthy by making a simple API call."""   
        try:
            response = requests.get(f"{self.get_endpoint()}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False
        
    def get_version(self) -> str:
        """Get the version of the vLLM engine."""
        try:
            response = requests.get(f"{self.get_endpoint()}/version")
            return response.json()["version"]
        except requests.RequestException:
            return "Unknown"
        
            
    def get_full_engine_args(self) -> Dict[str, Any]: # TODO: create a pr in vllm repo to add a new api to get the full engine args; move get_converted_dtype out of this function; key information missing, such as max_num_batched_tokens
        """
        Get all engine arguments used by vLLM by parsing the container logs.
        Specifically looks for the configuration line in the vLLM logs.
        
        Returns:
            Dict[str, Any]: Dictionary containing engine arguments and environment variables
        """
        if not self.container_id or self.status != "running":
            return {"success": False, "error": "Engine not running"}
        
        # Get container logs
        logs = get_container_logs(self.container_id, tail=None)  
        if not logs:
            return {"success": False, "error": "Failed to retrieve container logs"}
        
        # Look for the config line in the logs
        full_engine_args = {}
        
        # Find lines containing config information
        for line in logs.splitlines():
            if ("Initializing a V1 LLM engine" in line or "Initializing a V0 LLM engine" in line) and "with config:" in line:
                # Extract the config part
                config_str = line.split("with config:")[1].strip()
                
                # Parse the configuration string
                try:
                    # Parse key-value pairs
                    current_key = None
                    current_value = ""
                    in_nested = 0
                    
                    i = 0
                    while i < len(config_str):
                        # Skip spaces between key-value pairs
                        if config_str[i].isspace() and current_key is None:
                            i += 1
                            continue
                            
                        # Handle key
                        if current_key is None:
                            key_end = config_str.find('=', i)
                            if key_end != -1:
                                current_key = config_str[i:key_end].strip()
                                i = key_end + 1
                            else:
                                # No more key-value pairs
                                break
                        else:
                            # Handle value
                            if in_nested == 0:
                                # Not inside nested structure
                                if config_str[i] == ',' and not (i > 0 and config_str[i-1] == '}'):
                                    # End of value
                                    full_engine_args[current_key] = self._parse_value(current_value.strip())
                                    current_key = None
                                    current_value = ""
                                    i += 1
                                    continue
                                elif config_str[i] in '({[':
                                    # Start of nested structure
                                    in_nested += 1
                                    current_value += config_str[i]
                                else:
                                    current_value += config_str[i]
                            else:
                                # Inside nested structure
                                if config_str[i] in '({[':
                                    in_nested += 1
                                elif config_str[i] in ')}]':
                                    in_nested -= 1
                                current_value += config_str[i]
                            
                            i += 1
                    
                    # Add the last key-value pair if any
                    if current_key is not None:
                        full_engine_args[current_key] = self._parse_value(current_value.strip())
                
                except Exception as e:
                    print(f"Error parsing config: {e}")
                    continue
                
                break  # We found and processed the config line, no need to continue
        
        # Update converted_dtype and converted_quantization values
        self.get_converted_dtype(full_engine_args)
        self.get_converted_quantization(full_engine_args)
        
        self.full_engine_args = full_engine_args
        return full_engine_args

    def get_converted_dtype(self, engine_args: Dict[str, Any]) -> Optional[str]:
        """
        Extract and convert dtype from engine arguments.
        
        Args:
            engine_args: Dictionary containing engine arguments
        """

        dtype = engine_args.get("dtype")
        if dtype:
            self.converted_dtype = dtype.replace("torch.", "")
        return self.converted_dtype

    def get_converted_quantization(self, engine_args: Dict[str, Any]) -> Optional[str]:
        """
        Extract and convert quantization information from engine arguments.
        
        Args:
            engine_args: Dictionary containing engine arguments
        """
      
        # Look for quantization-related parameters
        quantization = None
        # Check for quantization type in various places
        if "quantization" in engine_args:
            quantization = engine_args["quantization"]
            self.converted_quantization = quantization
        return self.converted_quantization


    def _parse_value(self, value_str: str) -> Any:
        """
        Parse a string value into its appropriate Python type.
        
        Args:
            value_str: String representation of a value
            
        Returns:
            The parsed value in its appropriate type
        """
        # Clean input
        value_str = value_str.strip()
        if value_str.endswith(','):
            value_str = value_str[:-1].strip()
            
        # Handle simple cases directly
        if value_str.lower() == 'none':
            return None
        if value_str.lower() == 'true':
            return True
        if value_str.lower() == 'false':
            return False
        
        # Handle numbers
        try:
            if '.' in value_str:
                return float(value_str)
            return int(value_str)
        except ValueError:
            pass
            
        # Handle JSON structures
        if (value_str.startswith('{') and value_str.endswith('}')) or \
        (value_str.startswith('[') and value_str.endswith(']')):
            try:
                return json.loads(value_str)
            except json.JSONDecodeError:
                pass
        
        # Handle configuration objects (DecodingConfig, etc.)
        if '(' in value_str and ')' in value_str and not value_str.startswith('('):
            config_type, rest = value_str.split('(', 1)
            params_str = rest.rstrip(')')
            
            params = {}
            if params_str.strip():
                # Split by commas, handling nested structures
                parts, current, level = [], "", 0
                
                for char in params_str:
                    if char in '({[':
                        level += 1
                    elif char in ')}]':
                        level -= 1
                    
                    if char == ',' and level == 0:
                        parts.append(current.strip())
                        current = ""
                    else:
                        current += char
                
                if current:
                    parts.append(current.strip())
                
                # Process key-value pairs
                for part in parts:
                    if '=' in part:
                        key, val = part.split('=', 1)
                        params[key.strip()] = self._parse_value(val.strip())
            
            return {"type": config_type.strip(), "params": params}
        
        # Handle quoted strings
        if (value_str.startswith('"') and value_str.endswith('"')) or \
        (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        # Return as string for anything else
        return value_str