import os
import re
import time
import threading
from typing import Dict, Any, Optional, List

import requests

from src.engines.base_engine import BaseEngine
from src.utils.docker_utils import (
    run_docker_command, 
    get_container_logs,
    stream_container_logs,
)
from src.utils.utils import stream_process_output, get_environment_variables_from_command_line



class TensorRTEngine(BaseEngine):
    """
    Implementation for TensorRT-LLM inference engine.
    """
    
    engine_type = "trt"
    DEFAULT_PORT = 9000
    DEFAULT_STARTUP_TIMEOUT = 1800  # Default maximum wait time in seconds for server to be ready 
    DEFAULT_MAX_NUM_TOKENS = 64000  # Default value for max_num_tokens
    
    def __init__(
        self,
        model: str,
        startup_timeout: int = DEFAULT_STARTUP_TIMEOUT,
        **kwargs
    ):
        super().__init__(model=model, **kwargs)
        
        # Set default port and environment configuration
        self.port = self.DEFAULT_PORT
        self.hf_token = os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
        self.gpu_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        self.startup_timeout = startup_timeout
        
        # Store engine arguments separately from other config
        self.engine_args = {}
        self.env_vars = {}
        
        # Process kwargs to separate engine arguments from other config
        for key, value in kwargs.items():
            if key == "port":
                self.port = value
            self.engine_args[key] = value
        
        # Set default max_num_tokens if not provided
        if "max-num-tokens" not in self.engine_args:
            self.engine_args["max-num-tokens"] = self.DEFAULT_MAX_NUM_TOKENS
    
    def start(self, stream_logs: bool = True) -> bool:
        """
        Start the TensorRT-LLM engine in a Docker container.
        
        Args:
            stream_logs: Whether to stream container logs to the console
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        # Check if the Docker image exists, if not pull it with streaming output
        image_name = "nvcr.io/nvidia/tritonserver:25.01-trtllm-python-py3"
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
            "docker", "run", "-d", "--name", f"{self.name}_arena",
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
            
        # # Check if model is a local path
        # is_local_model = self.model.startswith('/') or (self.model.startswith('.') and '/' in self.model)
        
        # # If it's a local path, extract the directory and mount it
        # if is_local_model:
        #     model_dir = os.path.dirname(os.path.abspath(self.model))
            
        #     # Mount the model directory
        #     docker_cmd.extend([
        #         "-v", f"{model_dir}:{model_dir}",
        #     ])
            
        #     # Adjust the model path for use inside the container
        #     container_model = os.path.join(model_dir, os.path.basename(self.model))
        #     self.model = container_model
        # else:
        #     # Standard mount for HF cache
        #     docker_cmd.extend([
        #         "-v", f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
        #     ])
            
        # Continue with the rest of the docker command
        docker_cmd.extend([
            "-v", f"{os.path.expanduser('~')}/.cache/huggingface:/root/.cache/huggingface",
            "-p", f"{self.port}:{self.port}",
            "--ipc=host",
        ])
        
        # Add HF token if available
        if self.hf_token:
            docker_cmd.extend(["--env", f"HF_TOKEN={self.hf_token}"])
        
        # Get and add all TENSORRT_ environment variables
        self.env_vars = get_environment_variables_from_command_line("TRT")
        print("env_vars in tensorrt_engine.py", self.env_vars)
        for env_name, env_value in self.env_vars.items():
            docker_cmd.extend(["--env", f"{env_name}={env_value}"])
        
        # Add image
        docker_cmd.append(image_name)
        
        # Add the base command to start TensorRT-LLM server
        docker_cmd.extend([
            "trtllm-serve",
            self.model,
            "--port", str(self.DEFAULT_PORT),
            "--host", "0.0.0.0"
        ])
        
        # Add engine arguments from config
        # TensorRT-LLM uses underscores in arguments, not hyphens
        for arg_name, arg_value in self.engine_args.items():
            # Convert kebab-case back to snake_case for TensorRT-LLM
            trt_arg_name = arg_name.replace("-", "_")
            
            if isinstance(arg_value, bool):
                if arg_value:
                    docker_cmd.append(f"--{trt_arg_name}")
            else:
                docker_cmd.extend([f"--{trt_arg_name}", str(arg_value)])
        
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
        Stop the TensorRT-LLM engine container.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        return super().stop()

    
    def get_endpoint(self) -> str:
        """
        Get the API endpoint URL for the TensorRT-LLM engine.
        
        Returns:
            str: Endpoint URL
        """
        return f"http://localhost:{self.port}"
    
    def _check_if_ready(self) -> bool:
        """
        Check if the TensorRT-LLM server is ready to accept requests.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.is_healthy()
    
    def is_healthy(self) -> bool:
        """Check if the TensorRT-LLM engine is healthy by making a simple API call."""
        try:
            response = requests.get(f"{self.get_endpoint()}/health", timeout=5)

            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def get_version(self) -> str:
        """Get the version of the TensorRT-LLM engine."""
        try:
            response = requests.get(f"{self.get_endpoint()}/version")

            return response.json().get("version", "Unknown")
        except requests.RequestException:
            return "Unknown"
        
    def get_full_engine_args(self) -> Dict[str, Any]:
            """
            Get all engine args about the loaded TensorRT-LLM model.
            
            This implementation extracts model information from the container logs
            and combines it with information from the API if available.
            
            Returns:
                Dict[str, Any]: Model information including configuration parameters
            """
            if not self.container_id or self.status != "running":
                return {"success": False, "error": "Engine not running"}
            
            logs = get_container_logs(self.container_id, tail=None)  
            if not logs:
                return {"success": False, "error": "Failed to retrieve container logs"}
            full_engine_args = self._parse_value(logs)
            return full_engine_args
    
    def _parse_value(self, logs: str) -> Dict[str, Any]:
        """
        Extract TensorRT-LLM configuration from container logs.
        
        This function parses the logs from a TensorRT-LLM container to extract important
        configuration parameters like model settings, batch sizes, sequence lengths, etc.
        
        Args:
            container_id: Docker container ID
            
        Returns:
            Dict[str, Any]: Dictionary containing TensorRT-LLM configuration parameters
        """
    

        engine_args = {}    
        # Extract TRTGptModel parameters using regex patterns
        # Example log lines:
        # [TensorRT-LLM][INFO] TRTGptModel maxNumSequences: 2048
        # [TensorRT-LLM][INFO] TRTGptModel maxBatchSize: 2048
        param_patterns = [
            (r"TRTGptModel maxNumSequences: (\d+)", "max_num_sequences", int),
            (r"TRTGptModel maxBatchSize: (\d+)", "max_batch_size", int),
            (r"TRTGptModel maxBeamWidth: (\d+)", "max_beam_width", int),
            (r"TRTGptModel maxSequenceLen: (\d+)", "max_seq_len", int),
            (r"TRTGptModel maxDraftLen: (\d+)", "max_draft_length", int),
            (r"TRTGptModel mMaxAttentionWindowSize: \((\d+)\) \* (\d+)", "max_attention_window_size", lambda x, y: int(x) * int(y)),
            (r"TRTGptModel enableTrtOverlap: (\d+)", "enable_trt_overlap", lambda x: bool(int(x))),
            (r"TRTGptModel normalizeLogProbs: (\d+)", "normalize_log_probs", lambda x: bool(int(x))),
            (r"TRTGptModel maxNumTokens: (\d+)", "max_num_tokens", int),
            (r"TRTGptModel maxInputLen: (\d+)", "max_input_length", int),
        ]
        
        for pattern, key, converter in param_patterns:
            match = re.search(pattern, logs)
            if match:
                if len(match.groups()) == 1:
                    engine_args[key] = converter(match.group(1))
                elif len(match.groups()) == 2:
                    engine_args[key] = converter(match.group(1), match.group(2))
        
        
        return {"success": True, "config": engine_args}