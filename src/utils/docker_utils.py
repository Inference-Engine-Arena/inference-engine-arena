import subprocess
import threading
from typing import Dict, Any, List, Optional
import json
import logging
from src.utils.utils import stream_process_output, calculate_num_gpus

logger = logging.getLogger(__name__)


def run_docker_command(cmd: List[str]) -> Dict[str, Any]:
    """
    Run a docker command and return the result.
    
    Args:
        cmd: Command list to execute
        
    Returns:
        Dict with keys:
            - success: bool indicating if command succeeded
            - output: stdout if successful
            - error: stderr if failed
            - exit_code: exit code of the command
    """
    try:
        logger.debug(f"Running docker command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            return {
                "success": True,
                "output": result.stdout,
                "exit_code": result.returncode
            }
        else:
            logger.error(f"Docker command failed: {result.stderr}")
            return {
                "success": False,
                "error": result.stderr,
                "exit_code": result.returncode
            }
    except Exception as e:
        logger.exception(f"Exception running docker command: {e}")
        return {
            "success": False,
            "error": str(e),
            "exit_code": -1
        }


def get_container_logs(container_id: str, tail: Optional[int] = 100) -> str:
    """
    Get logs from a container.
    
    Args:
        container_id: Docker container ID
        tail: Number of log lines to return (defaults to last 100)
        
    Returns:
        str: Container logs
    """
    cmd = ["docker", "logs"]
    if tail is not None:
        cmd.extend(["--tail", str(tail)])
    cmd.append(container_id)
    
    result = run_docker_command(cmd)
    if result["success"]:
        return result["output"]
    return ""


def stream_container_logs(container_id: str, stop_event: threading.Event = None):
    """
    Stream logs from a container to the console in real-time.
    
    Args:
        container_id: Docker container ID
        stop_event: Threading event to signal when to stop streaming
    """
    # Create the docker logs command with follow option
    cmd = ["docker", "logs", "--follow", container_id]
    return stream_process_output(cmd, stop_event)


def list_running_containers() -> List[Dict[str, Any]]:
    """
    List all running containers.
    
    Returns:
        List[Dict[str, Any]]: List of container information dictionaries
    """
    cmd = [
        "docker", "ps", "--format", 
        '{"id":"{{.ID}}", "image":"{{.Image}}", "names":"{{.Names}}", "ports":"{{.Ports}}"}'
    ]
    result = run_docker_command(cmd)
    
    if result["success"]:
        containers = []
        for line in result["output"].strip().split("\n"):
            if line:
                try:
                    containers.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse container info: {line}")
        return containers
    return []


def get_gpu_info(container_id: Optional[str] = None, engine_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get GPU information using nvidia-smi.
    
    Args:
        container_id: Optional Docker container ID. If provided, will return
                     GPU information only for this container.
        engine_args: Optional engine arguments to calculate the number of GPUs
                     
    Returns:
        Dict[str, Any]: GPU information
    """
    result = {}
    
    if container_id:
        return get_container_gpu_info(container_id, engine_args)
    else:
        return get_global_gpu_info()    
    

def get_global_gpu_info() -> Dict[str, Any]:
    """
    Get global GPU information using nvidia-smi.
    
    Returns:
        Dict[str, Any]: GPU information
    """
    cmd = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False
        )
        
        if result.returncode == 0:
            gpus = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 5:
                        gpus.append({
                            "index": int(parts[0]),
                            "name": parts[1],
                            "memory_total": float(parts[2]),
                            "memory_used": float(parts[3]),
                            "utilization": float(parts[4])
                        })
            return {
                "success": True,
                "num_gpus": len(gpus),
                "gpus": gpus
            }
        else:
            return {
                "success": False,
                "error": result.stderr,
            }
    except Exception as e:
        logger.exception(f"Exception getting GPU info: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_container_gpu_info(container_id: str, engine_args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Get GPU information for a specific container.
    
    Args:
        container_id: Docker container ID
        engine_args: Optional engine arguments to calculate the number of GPUs
    Returns:
        Dict[str, Any]: GPU information for the container
    """
    # First, check which GPUs the container can access
    cmd = ["docker", "exec", container_id, "bash", "-c", "echo $CUDA_VISIBLE_DEVICES"]

    result = run_docker_command(cmd)

    
    if not result["success"]:
        logger.error(f"Failed to get CUDA_VISIBLE_DEVICES from container: {result.get('error', 'Unknown error')}")
        return {
            "success": False,
            "error": "Failed to get GPU information from container"
        }
    
    # Get the GPU indices the container can access
    cuda_devices = result["output"].strip()

    
    # If CUDA_VISIBLE_DEVICES is not set or empty, the container can access all GPUs
    if not cuda_devices or cuda_devices == "":
        gpu_indices = None  # All GPUs
    else:
        # Parse the CUDA_VISIBLE_DEVICES value
        try:
            gpu_indices = [int(idx) for idx in cuda_devices.split(',')]
        except ValueError:
            logger.error(f"Invalid CUDA_VISIBLE_DEVICES value: {cuda_devices}")
            gpu_indices = None
    
    # Get global GPU info
    global_gpu_info = get_global_gpu_info()

    if not global_gpu_info["success"]:
        return global_gpu_info
    
    # Filter GPUs if specific indices are set
    if gpu_indices is not None:

        container_gpus = [
            gpu for gpu in global_gpu_info["gpus"] 
            if gpu["index"] in gpu_indices
        ]
    else:
        container_gpus = global_gpu_info["gpus"]
    
    if engine_args and container_gpus:
        num_gpus = calculate_num_gpus(engine_args)
        if len(container_gpus) > num_gpus:
            container_gpus = container_gpus[:num_gpus]
    
    # Now get container-specific GPU stats using nvidia-docker
    cmd = ["docker", "stats", "--no-stream", "--format", "{{json .}}", container_id]
    
    result = run_docker_command(cmd)
    
    container_stats = {}
    if result["success"]:
        try:
            stats = json.loads(result["output"])
            container_stats = {
                "cpu_usage": stats.get("CPUPerc", "N/A"),
                "memory_usage": stats.get("MemUsage", "N/A"),
                "memory_percent": stats.get("MemPerc", "N/A")
            }
        except json.JSONDecodeError:
            logger.error(f"Failed to parse container stats: {result['output']}")
    
    return {
        "success": True,
        "container_id": container_id,
        "container_stats": container_stats,
        "num_gpus": len(container_gpus),
        "gpus": container_gpus
    }


def extract_container_environment_variables(container_detail: Dict[str, Any], prefix: Optional[str] = None) -> Dict[str, str]:
    """
    Extract environment variables from container details, optionally filtering by a prefix.
    
    Args:
        container_detail: Container details from Docker inspect command
        prefix: Optional prefix to filter environment variables by
        
    Returns:
        Dict[str, str]: Dictionary of environment variables
    """
    env_vars = {}

     # List of environment variables to exclude
    excluded_env_vars = ["VLLM_USAGE_SOURCE", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"]

    if "Config" in container_detail and "Env" in container_detail["Config"]:
        for env_var in container_detail["Config"]["Env"]:
            if "=" in env_var:
                key, value = env_var.split("=", 1)
                if key in excluded_env_vars:
                    continue
                if prefix is None or key.lower().startswith(prefix.lower()):
                    env_vars[key] = value
    
    return env_vars


def extract_container_engine_args(container_detail: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract model and engine arguments from container command arguments.
    
    Args:
        container_detail: Container details from Docker inspect command
        
    Returns:
        Dict[str, Any]: Dictionary containing:
            - model: Extracted model name/path if found
            - engine_args: Dictionary of engine arguments
    """
    model = None
    engine_args = {}
    
    if "Cmd" in container_detail.get("Config", {}):
        args = container_detail["Config"]["Cmd"]
        # Find the model
        model_flags = ["--model", "--model-path", "trtllm-serve"]
        for i, arg in enumerate(args):
            if arg in model_flags and i + 1 < len(args):
                model = args[i + 1]
                break
        
        # Extract engine arguments
        i = 0
        while i < len(args):
            arg = args[i]

            if arg.startswith("--"):
                arg_name = arg[2:]  # Remove the '--'
                
                # Check if the next arg is a value or another flag
                if i + 1 < len(args) and not args[i + 1].startswith("--") and str(args[i + 1]) not in model_flags:
                    # Convert value to appropriate type if possible
                    value = args[i + 1]
                    if value.isdigit():
                        value = int(value)
                    elif value.lower() in ("true", "false"):
                        value = value.lower() == "true"
                    
                    # Store with kebab-case as key
                    engine_args[arg_name] = value
                    i += 2
                else:
                    # Flag without value (boolean flag)
                    engine_args[arg_name] = True
                    i += 1
            else:
                i += 1

    return model, engine_args

def get_model_config_json(container_id: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Attempt to get a model's config.json from a Docker container.
    
    Args:
        container_id: Docker container ID
        model_name: Name of the model to find config.json for
        
    Returns:
        Optional[Dict[str, Any]]: The config.json as a dictionary, or None if not found/accessible
    """
    try:
        if not container_id:
            return None
            
        # Convert model name for proper path matching
        # Replace "/" with "--" to match HuggingFace's cache structure
        normalized_model_name = model_name.replace("/", "--")
        
        # Try direct path for HuggingFace models in the new path format
        cmd = [
            "docker", "exec", container_id,
            "bash", "-c", f"find /root/.cache/huggingface/hub/models--{normalized_model_name} -name 'config.json' | head -n 1"
        ]
        result = run_docker_command(cmd)
        if result["success"] and result["output"].strip():
            config_path = result["output"].strip()
            
            # Now get the content of the config.json
            cat_cmd = ["docker", "exec", container_id, "cat", config_path]
            cat_result = run_docker_command(cat_cmd)
            if cat_result["success"]:
                return json.loads(cat_result["output"])
        
        
        # If we got here, we couldn't find the config.json
        logger.warning(f"Could not find config.json for model {model_name} in container {container_id}")
        return None
    except Exception as e:
        logger.error(f"Error getting model config: {e}")
        return None


def get_container_status(container_id: str) -> str:
    """
    Get the status of a Docker container.
    
    Args:
        container_id: Docker container ID
        
    Returns:
        str: Container status (e.g., "running", "exited", "created", etc.)
    """
    if not container_id:
        return "unknown"
        
    cmd = ["docker", "inspect", "--format", "{{.State.Status}}", container_id]
    result = run_docker_command(cmd)
    
    if result["success"] and result["output"].strip():
        # Return the status in lowercase
        return result["output"].strip().lower()
    else:
        logger.error(f"Failed to get status for container {container_id}: {result.get('error', 'Unknown error')}")
        return "unknown"
