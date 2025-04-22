import subprocess
import threading
import time
import sys
from typing import Dict, Any, List, Optional, Tuple
import logging
import os

logger = logging.getLogger(__name__)


def get_environment_variables_from_command_line(prefix: str) -> Dict[str, str]:
    """
    Collect all environment variables that start with the given prefix.
    
    Args:
        prefix: The prefix to filter environment variables by (e.g., "VLLM", "SGLANG")
    
    Returns:
        Dict[str, str]: Dictionary of environment variables
    """
    env_vars = {}
    for key, value in os.environ.items():
        if key.startswith(prefix):
            env_vars[key] = value
    
    logger.debug(f"Found {len(env_vars)} {prefix.rstrip('_')} environment variables: {list(env_vars.keys())}")
    return env_vars


def get_final_precision(converted_dtype: Optional[str], converted_quantization: Optional[str]) -> str:
    """
    Determine the final dtype-quantization category based on the converted_dtype and converted_quantization.
    
    Args:
        converted_dtype: The data type used for conversion (e.g., "float16", "bfloat16")
        converted_quantization: The quantization method used (e.g., "awq", "gptq", "int8")
    
    Returns:
        str: The high-level category (only INT4, INT8, FP8, FP16, BF16, FP32, or OTHER)
    """
    # Helper dictionary for mapping dtype to categories
    dtype_to_category = {
        "float16": "FP16",
        "bfloat16": "BF16",
        "float32": "FP32",
        None: "FP32"  # Default if not specified
    }
    
    # INT4 quantization methods
    int4_methods = [
        "awq", "awq_marlin",            # AWQ family
        "gptq", "gptq_marlin", "gptq_marlin_24",  # GPTQ family
        "nvfp4",                         # NVIDIA specific
        "aqlm", "qqq", "hqq", "quark"    # Other INT4 variants
    ]
    
    # INT8 quantization methods
    int8_methods = [
        "bitsandbytes", "tpu_int8", "experts_int8",  # Standard INT8
        "blockwise_int8", "w8a8_int8",              # SGLang specific
        "ipex", "neuron_quant",                     # Hardware optimized
        "moe_wna16"                                 # Mixed INT8
    ]
    
    # FP8 quantization methods
    fp8_methods = [
        "fp8",                  # Standard FP8
        "ptpc_fp8", "fbgemm_fp8",  # Vendor implementations
        "w8a8_fp8"              # SGLang specific
    ]
    
    # If only dtype is specified
    if converted_quantization is None:
        if converted_dtype in dtype_to_category:
            return dtype_to_category[converted_dtype]
        return "OTHER"
    
    # If quantization is specified
    if converted_quantization:
        if converted_quantization.lower() in int4_methods:
            return "INT4"
        elif converted_quantization.lower() in int8_methods:
            return "INT8"
        elif converted_quantization.lower() in fp8_methods:
            return "FP8"
        else:
            return "OTHER"
    
    # Default fallback
    return "FP32"


def stream_process_output(cmd: List[str], stop_event: threading.Event = None) -> int:
    """
    Stream output from a process to the console in real-time.
    
    Args:
        cmd: Command to execute as a list of strings
        stop_event: Threading event to signal when to stop streaming
        
    Returns:
        int: Return code of the process
    """
    process = None
    try:
        logger.debug(f"Running streaming process: {' '.join(cmd)}")
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1  # Line buffered
        )
        
        # Read and print output line by line
        while True:
            # Check if we should stop
            if stop_event and stop_event.is_set():
                process.terminate()
                break
                
            # Read a line
            line = process.stdout.readline()
            
            # If no more output and process has exited, we're done
            if not line and process.poll() is not None:
                break
                
            # If no output but process still running, wait a bit and try again
            if not line:
                time.sleep(0.1)
                continue
                
            # Print the line (strip trailing newline to avoid double spacing)
            sys.stdout.write(line)
            sys.stdout.flush()
        
        # Make sure we get the final exit code
        if process.poll() is None:
            process.wait()
            
        return_code = process.returncode
        logger.debug(f"Process completed with return code: {return_code}")
        return return_code if return_code is not None else -1
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully
        if process and process.poll() is None:
            process.terminate()
        return -1
    except Exception as e:
        logger.exception(f"Error streaming process output: {e}")
        return -1
    finally:
        # Ensure process is terminated
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

from contextlib import contextmanager

@contextmanager
def temporary_env_vars(envs):
    '''
    Example usage:
    print("original hugging face token", os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    with temporary_env_vars({"HUGGING_FACE_HUB_TOKEN": "kkkkk", "CUDA_VISIBLE_DEVICES": "0,1"}):
        print("new hugging face token", os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    print("afterwards hugging face token", os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    '''
    original_values = {}
    for key in envs:
        if key in os.environ:
            original_values[key] = os.environ[key]
        
    for key, value in envs.items():
        os.environ[key] = value
        
    try:
        yield
    finally:
        for key in envs:
            if key in original_values:
                os.environ[key] = original_values[key]
            else:
                del os.environ[key]

def parse_benchmark_yaml(yaml_file: str) -> List[Dict[str, Any]]:
    """
    Parse a YAML configuration file for benchmark runs.
    
    Args:
        yaml_file: Path to the YAML configuration file
        
    Returns:
        List[Dict[str, Any]]: List of engine and benchmark configurations grouped by runs
    """
    import yaml
    
    try:
        # Parse the YAML file
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # Validate the structure
        if not yaml_data or 'runs' not in yaml_data:
            logger.error("Invalid YAML file: missing 'runs' section")
            return []
        
        runs_section = yaml_data['runs']
        
        # Ensure runs is a list
        if not isinstance(runs_section, list):
            logger.error("Invalid YAML format: 'runs' should be a list of dictionaries")
            return []
        
        # Extract engine and benchmarks from each run
        engine_benchmark_groups = []
        for run in runs_section:
            if 'engine' in run and 'benchmarks' in run:
                engine_benchmark_groups.append({
                    'engine': run['engine'],
                    'benchmarks': run['benchmarks']
                })
        
        logger.info(f"Parsed {len(engine_benchmark_groups)} engine-benchmark groups from YAML config")
        
        # Debug output of parsed groups
        for idx, group in enumerate(engine_benchmark_groups):
            engine_info = group['engine'][0] if isinstance(group['engine'], list) and group['engine'] else group['engine']
            engine_type = engine_info.get('type', 'Unknown') if isinstance(engine_info, dict) else 'Unknown'
            
            if isinstance(group['benchmarks'], list):
                benchmarks = [b.get('type', 'Unknown') if isinstance(b, dict) else str(b) for b in group['benchmarks']]
            else:
                benchmarks = [str(group['benchmarks'])]
                
            logger.info(f"Group {idx+1}: Engine={engine_type}, Benchmarks={benchmarks}")
            
        return engine_benchmark_groups
        
    except Exception as e:
        logger.exception(f"Error parsing benchmark YAML config: {e}")
        return []
    
import tomli
from pathlib import Path

def get_project_version():
    try:
        # Get the directory of the current file (utils.py)
        current_dir = Path(__file__).parent
        # Go up two levels to reach the project root (src -> project root)
        project_root = current_dir.parent.parent
        # Construct path to pyproject.toml
        pyproject_path = project_root / "pyproject.toml"
        
        if pyproject_path.exists():
            with open(pyproject_path, "rb") as f:
                pyproject_data = tomli.load(f)
                return pyproject_data.get("project", {}).get("version", "N/A")
        return "N/A"  # Default fallback
    except Exception as e:
        logger.error(f"Error reading project version: {str(e)}")
        return "N/A"  # Default fallback
    
def calculate_num_gpus(engine_args: Dict[str, Any] = None) -> int:
    """Calculate the number of GPUs based on parallelism dimensions.
    
    Args:
        engine_args (dict): Engine arguments containing parallelism settings
        
    Returns:
        int: Number of GPUs calculated as dp * pp * tp
    """
    dp = 1  # data parallelism
    pp = 1  # pipeline parallelism
    tp = 1  # tensor parallelism
    
    if "dp" in engine_args:
        dp = engine_args["dp"]
    elif "data-parallel-size" in engine_args:
        dp = engine_args["data-parallel-size"]
    elif "dp-size" in engine_args:
        dp = engine_args["dp-size"]
    
    if "pp" in engine_args:
        pp = engine_args["pp"]
    elif "pipeline-parallel-size" in engine_args:
        pp = engine_args["pipeline-parallel-size"]
    
    if "tp" in engine_args:
        tp = engine_args["tp"]
    elif "tensor-parallel-size" in engine_args:
        tp = engine_args["tensor-parallel-size"]
    elif "tp-size" in engine_args:
        tp = engine_args["tp-size"]
    
    return dp * pp * tp