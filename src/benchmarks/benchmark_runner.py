import re
import uuid
import json
import yaml
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

from src.engines.base_engine import BaseEngine
from src.utils.docker_utils import get_gpu_info

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runner for executing benchmarks against inference engines.
    """
    
    def __init__(self, results_dir: str = "./results"):
        """
        Initialize a benchmark runner.
        
        Args:
            results_dir: Directory to store benchmark results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load benchmark configurations
        self.benchmark_configs = self._load_benchmark_configs()
    
    def _extract_namespace_params(self, stdout: str) -> Dict[str, Any]:
        """
        Extract Namespace parameters from the stdout.
        
        Args:
            stdout: Standard output from benchmark process
            
        Returns:
            Dict[str, Any]: Extracted Namespace parameters
        """
        namespace_params = {}
        
        # Find the Namespace(...) part in the stdout
        namespace_match = re.search(r'Namespace\((.*?)\)', stdout, re.DOTALL)
        if namespace_match:
            namespace_str = namespace_match.group(1)
            
            # Process each key-value pair
            for param in re.finditer(r'(\w+)=([^,]+?)(?:,|\))', namespace_str):
                key = param.group(1)
                value_str = param.group(2).strip()
                
                # Convert string values to appropriate types
                if value_str == 'None':
                    value = None
                elif value_str == 'True':
                    value = True
                elif value_str == 'False':
                    value = False
                elif value_str.replace('.', '', 1).isdigit():
                    value = float(value_str) if '.' in value_str else int(value_str)
                else:
                    # Remove quotes for string values
                    value = value_str.strip("'")
                
                namespace_params[key] = value
        
        return namespace_params
    
    def _load_benchmark_configs(self) -> Dict[str, Dict[str, Any]]:
        """
        Load benchmark configurations from YAML files.
        
        Returns:
            Dict[str, Dict[str, Any]]: Dictionary of benchmark configs
        """
        configs = {}
        config_dir = Path(__file__).parent / "benchmark_configs"
        
        for config_file in config_dir.glob("*.yaml"):
            benchmark_name = config_file.stem
            try:
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                configs[benchmark_name] = config
            except Exception as e:
                logger.error(f"Failed to load benchmark config {config_file}: {e}")
        
        return configs
    
    def _generate_sub_run_id(self, engine_name: str, model_name: str, benchmark_type: str) -> str:
        """
        Generate a standardized ID for runs and sub-runs.
        
        Args:
            engine_name: Name of the engine
            model_name: Name of the model
            benchmark_type: Type of benchmark
            
        Returns:
            str: Generated ID
        """
        # Extract model short name from the full path
        model_short_name = model_name.split('/')[-1]
        # Clean up strings to be filesystem-friendly
        engine_clean = re.sub(r'[^a-zA-Z0-9]', '-', engine_name)
        model_clean = re.sub(r'[^a-zA-Z0-9]', '-', model_short_name)
        benchmark_clean = re.sub(r'[^a-zA-Z0-9]', '-', benchmark_type)
        
        # Generate timestamp and random suffix
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        random_suffix = str(uuid.uuid4())[:8]
        
        # Combine elements into ID
        id_str = f"sub-run-{timestamp}-{engine_clean}-{model_clean}-{benchmark_clean}-{random_suffix}"
        
        return id_str
    
    def _generate_run_id(self) -> str:
        timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
        random_suffix = str(uuid.uuid4())[:8]
        id_str = f"run-{timestamp}-{random_suffix}"
        return id_str
    
    def _get_current_time(self) -> str:
        """
        Get the current time in ISO format.
        
        Returns:
            str: Current time in ISO format
        """
        return datetime.now().isoformat()
    
    def _parse_timestamp(self, timestamp: str) -> datetime:
        """
        Parse a timestamp string into a datetime object.
        
        Args:
            timestamp: Timestamp string in ISO format
            
        Returns:
            datetime: Parsed datetime object
        """
        return datetime.fromisoformat(timestamp)
    
    def run_benchmark(
        self,
        engines: List[BaseEngine],
        benchmark_types: List[str]
    ) -> Dict[str, Any]:
        """
        Run benchmarks on specified engines with the given model.
        
        Args:
            engines: List of engine instances to benchmark
            benchmark_types: List of benchmark types to run
            
        Returns:
            Dict[str, Any]: Result summary
        """
        run_id = self._generate_run_id()
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Get global GPU info once before starting benchmarks
        gpu_info = get_gpu_info()
        
        # Create run metadata
        run_metadata = {
            "id": run_id,
            "start_time": self._get_current_time(),
            "engines": [engine.name for engine in engines],
            "benchmark_types": benchmark_types,
            "gpu_info": gpu_info,
            "sub_runs": []
        }
        
        # Save initial run metadata before starting any benchmarks
        run_metadata_path = run_dir / f"{run_id}.json"
        with open(run_metadata_path, 'w') as f:
            json.dump(run_metadata, f, indent=2)
        
        # Run each benchmark for each engine
        for benchmark_type in benchmark_types:
            if benchmark_type not in self.benchmark_configs:
                logger.warning(f"Unknown benchmark type: {benchmark_type}, skipping")
                continue
                
            benchmark_config = self.benchmark_configs[benchmark_type]
            
            for engine in engines:
                # Skip if engine is not running
                if engine.status != "running":
                    logger.warning(f"Engine {engine.name} is not running, skipping")
                    continue
                
                # Run sub-benchmark and collect results
                sub_run_id = self.run_sub_run_benchmark(
                    engine=engine,
                    benchmark_type=benchmark_type,
                    benchmark_config=benchmark_config,
                    run_id=run_id,
                    run_dir=run_dir
                )
                
                # Add sub-run ID to run metadata
                run_metadata["sub_runs"].append(sub_run_id)
                
                # Update run metadata file after each sub-run
                with open(run_metadata_path, 'w') as f:
                    json.dump(run_metadata, f, indent=2)
        
        # Update run metadata with end time
        run_metadata["end_time"] = self._get_current_time()
        run_metadata["duration_seconds"] = (
            self._parse_timestamp(run_metadata["end_time"]) - 
            self._parse_timestamp(run_metadata["start_time"])
        ).total_seconds()
        
        # Save final run metadata
        with open(run_metadata_path, 'w') as f:
            json.dump(run_metadata, f, indent=2)
        
        return run_metadata
    
    def run_sub_run_benchmark(
        self,
        engine: BaseEngine,
        benchmark_type: str,
        benchmark_config: Dict[str, Any],
        run_id: str,
        run_dir: Path
    ) -> str:
        """
        Run a single sub-benchmark for a specific engine and benchmark type.
        
        Args:
            engine: Engine to benchmark
            benchmark_type: Type of benchmark to run
            benchmark_config: Configuration for this benchmark
            run_id: ID of the parent run
            run_dir: Directory of the parent run
            
        Returns:
            str: ID of the sub-run
        """
        # Create sub-run with consistent naming
        sub_run_id = self._generate_sub_run_id(engine.name, engine.model, benchmark_type)
        sub_run_dir = run_dir / sub_run_id
        sub_run_dir.mkdir(parents=True)
        
        # Run the benchmark
        logger.info(f"Running benchmark {benchmark_type} on engine {engine.name}")
        
        # Execute the benchmark and get results
        result = self._execute_benchmark(
            engine=engine,
            benchmark_type=benchmark_type,
            benchmark_config=benchmark_config,
            sub_run_dir=sub_run_dir
        )
        
        # Get full engine information
        engine_info = engine.to_dict()
        
        # Prepare comprehensive sub-run result (single source of truth)
        sub_run_result = {
            # Basic identification
            "id": sub_run_id,
            "parent_run_id": run_id,
            
            # Components involved
            "engine": engine_info,
            "model": engine.model,
            "benchmark": {
                "type": benchmark_type,
                "config": benchmark_config,
                "command": result["command"],
                "namespace_params": result.get("namespace_params", {})
            },
            
            # Timing information
            "start_time": result["start_time"],
            "end_time": result["end_time"],
            "duration_seconds": result["duration_seconds"],
            
            # Results
            "success": result["success"],
            "metrics": result.get("metrics", {}),
            
            # Execution details (useful for debugging)
            "exit_code": result.get("exit_code"),
            "stdout": result.get("stdout"),
            "stderr": result.get("stderr")
        }
        
        # Save comprehensive sub-run result
        result_path = sub_run_dir / f"{sub_run_id}.json"
        with open(result_path, 'w') as f:
            json.dump(sub_run_result, f, indent=2)
        
        return sub_run_id
    
    def _execute_benchmark(
        self,
        engine: BaseEngine,
        benchmark_type: str,
        benchmark_config: Dict[str, Any],
        sub_run_dir: Path
    ) -> Dict[str, Any]:
        """
        Execute a single benchmark against an engine.
        
        Args:
            engine: Engine to benchmark
            benchmark_type: Type of benchmark to run
            benchmark_config: Configuration for this benchmark
            sub_run_dir: Directory to store results
            
        Returns:
            Dict[str, Any]: Benchmark result
        """
        start_time = self._get_current_time()


        # Prepare benchmark command
        benchmark_script = Path(__file__).parent / "benchmark_serving.py"
        cmd = ["python", str(benchmark_script)]
        
        # Add required parameters
        cmd.extend(["--base-url", engine.get_endpoint()])
        cmd.extend(["--model", engine.model])
        cmd.extend(["--backend", engine.name])
        cmd.extend(["--trust-remote-code"])
        cmd.extend(["--served-model-name", engine.engine_args.get("served-model-name", "")])
        
        # Add benchmark-specific parameters
        for key, value in benchmark_config.items():
            if key != "description":
                cmd.extend([f"--{key.replace('_', '-')}", str(value)])
        
        # Add result directory and filename
        cmd.extend([
            "--save-result", 
            "--result-dir", str(sub_run_dir),
            "--result-filename", "raw_result.json"
        ])
        
        # Execute benchmark and capture output
        logger.debug(f"Running benchmark command: {' '.join(cmd)}")
        result = {
            "start_time": start_time,
            "command": ' '.join(cmd)
        }
        
        try:
            # Log the command
            logger.info(f"Executing: {' '.join(cmd)}")
            
            print(f"\nRunning benchmark: {benchmark_type} on {engine.name}")
            print(f"Command: {' '.join(cmd)}")
            print("-" * 80)
            
            # Start the process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1  # Line buffered
            )
            
            # Capture output while streaming it
            stdout_lines = []
            stderr_lines = []
            
            # Handle stdout in real time
            while True:
                stdout_line = process.stdout.readline()
                if stdout_line:
                    print(stdout_line, end='')  # Print to console
                    stdout_lines.append(stdout_line)  # Store for later
                
                stderr_line = process.stderr.readline()
                if stderr_line:
                    print(stderr_line, end='')  # Print to console
                    stderr_lines.append(stderr_line)  # Store for later
                
                # Check if process has finished
                if process.poll() is not None:
                    # Read any remaining output
                    for line in process.stdout:
                        print(line, end='')
                        stdout_lines.append(line)
                    for line in process.stderr:
                        print(line, end='')
                        stderr_lines.append(line)
                    break
            
            print("-" * 80)
            
            # Store captured output
            stdout_content = ''.join(stdout_lines)
            result["stdout"] = stdout_content
            result["stderr"] = ''.join(stderr_lines)
            result["exit_code"] = process.returncode
            result["success"] = process.returncode == 0
            
            # Extract namespace parameters from stdout
            namespace_params = self._extract_namespace_params(stdout_content)
            result["namespace_params"] = namespace_params
            
            if not result["success"]:
                logger.error(f"Benchmark failed with exit code {process.returncode}")
                logger.error(f"Error: {result['stderr']}")
            
            # Parse metrics from result file if successful
            if result["success"]:
                metrics = self._parse_benchmark_output(sub_run_dir)
                result["metrics"] = metrics
                logger.info(f"Benchmark completed successfully. Key metrics: "
                           f"input_throughput={metrics.get('input_throughput', 'N/A')}, "
                           f"output_throughput={metrics.get('output_throughput', 'N/A')}, "
                           f"ttft={metrics.get('ttft', 'N/A')}")
            
        except Exception as e:
            logger.exception(f"Error executing benchmark: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        # Record end time and duration
        end_time = self._get_current_time()
        result["end_time"] = end_time
        result["duration_seconds"] = (
            self._parse_timestamp(end_time) - 
            self._parse_timestamp(start_time)
        ).total_seconds()
        
        return result
    
    def _parse_benchmark_output(self, result_dir: Path) -> Dict[str, Any]:
        """
        Parse benchmark output to extract metrics.
        
        Args:
            result_dir: Directory containing benchmark results
            
        Returns:
            Dict[str, Any]: Extracted metrics
        """
        metrics = {}
        try:
            # Look for raw result json file
            result_file = result_dir / "raw_result.json"
            
            if not result_file.exists():
                logger.warning(f"No raw_result.json found in {result_dir}")
                return metrics
                
            logger.debug(f"Parsing benchmark results from {result_file}")
            
            with open(result_file, 'r') as f:
                results_data = json.load(f)
            
            # Extract key metrics
            if isinstance(results_data, dict):
                # Throughput metrics
                metrics["input_throughput"] = results_data.get("total_token_throughput", 0) - results_data.get("output_throughput", 0)
                metrics["output_throughput"] = results_data.get("output_throughput", 0)
                
                # Latency metrics
                metrics["ttft"] = results_data.get("mean_ttft_ms", 0)  # Time to First Token
                metrics["tpot"] = results_data.get("mean_tpot_ms", 0)  # Time Per Output Token
                
                # Additional metrics
                metrics["duration"] = results_data.get("duration", 0)
                metrics["num_prompts"] = results_data.get("num_prompts", 0)
                metrics["total_input_tokens"] = results_data.get("total_input_tokens", 0)
                metrics["total_output_tokens"] = results_data.get("total_output_tokens", 0)
            else:
                logger.warning(f"Unexpected results format in {result_file}")
                
        except Exception as e:
            logger.exception(f"Error parsing benchmark output: {e}")
        
        return metrics
    
    def run_benchmark_from_yaml(self, yaml_file: str, engine_manager) -> Dict[str, Any]:
        """
        Run benchmarks from a YAML configuration file.
        
        This method handles the full lifecycle of engines and benchmarks:
        1. Parse the YAML configuration
        2. For each engine-benchmark group:
           a. Start the engine
           b. Run all benchmarks for that engine
           c. Stop the engine
        
        Args:
            yaml_file: Path to the YAML configuration file
            engine_manager: Instance of EngineManager to handle engine lifecycle
            
        Returns:
            Dict[str, Any]: Run metadata including results and statistics
        """
        from src.utils.utils import parse_benchmark_yaml, temporary_env_vars
        
        # Parse the YAML configuration file
        engine_benchmark_groups = parse_benchmark_yaml(yaml_file)
        
        if not engine_benchmark_groups:
            logger.error("No valid configurations found in the YAML file")
            return {"success": False, "error": "No valid configurations found"}
        
        # Create a run ID for the entire batch
        run_id = self._generate_run_id()
        run_dir = self.results_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        start_time = self._get_current_time()
        all_sub_runs = []
        all_engines = set()
        all_benchmark_types = set()
        
        logger.info(f"Starting benchmark run from YAML: {run_id}")
        logger.info(f"Found {len(engine_benchmark_groups)} engine-benchmark groups to process")
        
        # Create initial run metadata
        run_metadata = {
            "id": run_id,
            "start_time": start_time,
            "engines": list(all_engines),
            "benchmark_types": list(all_benchmark_types),
            "gpu_info": get_gpu_info(),
            "sub_runs": all_sub_runs,
            "success": True
        }
        
        # Save initial run metadata before starting any benchmarks
        run_metadata_path = run_dir / f"{run_id}.json"
        with open(run_metadata_path, 'w') as f:
            json.dump(run_metadata, f, indent=2)
        
        try:
            # Process each engine-benchmark group separately
            for group_idx, group in enumerate(engine_benchmark_groups):
                engine_configs = group.get('engine', [])
                benchmark_configs = group.get('benchmarks', [])
                
                # Make group processing more visible with separators and direct console output
                group_message = f"\n{'=' * 80}\nPROCESSING GROUP {group_idx+1}/{len(engine_benchmark_groups)}\n{'-' * 80}"
                print(group_message)
                logger.info(f"Processing group {group_idx+1}/{len(engine_benchmark_groups)}")
                
                if not engine_configs:
                    logger.warning(f"Skipping group with missing engine configuration")
                    continue
                    
                if not benchmark_configs:
                    logger.warning(f"Skipping group with missing benchmark configurations")
                    continue
                    
                # Log the engines and benchmarks in this group
                engine_names = [e.get('type', 'N/A') for e in engine_configs]
                benchmark_names = [b.get('type', 'N/A') for b in benchmark_configs]
                
                logger.info(f"  Engines: {engine_names}")
                logger.info(f"  Benchmarks: {benchmark_names}")
                
                # Process each engine configuration
                for engine_config in engine_configs:
                    engine_type = engine_config.get('type', '').lower()
                    model = engine_config.get('model', '')
                    
                    if not engine_type or not model:
                        logger.warning(f"Skipping engine config with missing type or model: {engine_config}")
                        continue
                    
                    all_engines.add(engine_type)
                    
                    # Parse engine arguments
                    engine_args = {}
                    engine_args_str = engine_config.get('args', '')
                    if engine_args_str:
                        args_parts = engine_args_str.split()
                        i = 0
                        while i < len(args_parts):
                            if args_parts[i].startswith('--'):
                                key = args_parts[i][2:]  # Remove leading --
                                if i + 1 < len(args_parts) and not args_parts[i + 1].startswith('--'):
                                    value = args_parts[i + 1]
                                    engine_args[key] = value
                                    i += 2
                                else:
                                    engine_args[key] = True
                                    i += 1
                            else:
                                i += 1
                    
                    # Get environment variables
                    env_vars = engine_config.get('env', {})
                    
                    logger.info(f"Starting engine {engine_type} with model {model}")
                    logger.info(f"  Arguments: {engine_args}")
                    logger.info(f"  Environment variables: {env_vars}")
                    
                    # Handle the case if an engine with the same name is already running
                    existing_engine = engine_manager.get_engine_by_name(engine_type)
                    if existing_engine and existing_engine.status == "running":
                        logger.warning(f"An engine with name '{engine_type}' is already running, stopping it")
                        engine_manager.stop_engine(engine_type)
                    
                    # Create and start the engine
                    engine = engine_manager.create_engine(engine_type, model, **engine_args)
                    if not engine:
                        logger.error(f"Failed to create {engine_type} engine with model {model}")
                        continue
                    
                    # Apply environment variables and start the engine
                    with temporary_env_vars(env_vars):
                        success = engine_manager.start_engine(engine.name)
                        
                        if not success:
                            logger.error(f"Failed to start engine: {engine.name}")
                            engine.env_vars = env_vars

                        else:                        
                            logger.info(f"Engine started successfully: {engine.name}")
                             # Refresh engine status to get the latest info
                            engine_manager.refresh_engines()
                            engine = engine_manager.get_engine_by_name(engine.name)
     
                        if not engine or engine.status != "running":
                            logger.error(f"Engine {engine_type} is not running after start")
                        
                        # Run each benchmark for this engine
                        for benchmark_idx, benchmark_config in enumerate(benchmark_configs):
                            # Make a copy to avoid modifying the original
                            benchmark_config_copy = benchmark_config.copy()
                            benchmark_type = benchmark_config_copy.pop('type', '')
                            
                            if not benchmark_type:
                                logger.warning(f"Skipping benchmark config with missing type: {benchmark_config}")
                                continue
                            
                            all_benchmark_types.add(benchmark_type)
                            
                            # Convert remaining benchmark config to a format expected by run_sub_run_benchmark
                            formatted_benchmark_config = {}
                            for key, value in benchmark_config_copy.items():
                                formatted_benchmark_config[key] = value
                            
                            # Print benchmark progress indicator
                            print(f"\n{'*' * 80}\n\n[BENCHMARK {benchmark_idx+1}/{len(benchmark_configs)} in GROUP {group_idx+1}/{len(engine_benchmark_groups)}]\n{'*' * 80}\n\n")
                            logger.info(f"Running benchmark {benchmark_type} on engine {engine.name}")
                            
                            # Run the benchmark
                            sub_run_id = self.run_sub_run_benchmark(
                                engine=engine,
                                benchmark_type=benchmark_type,
                                benchmark_config=formatted_benchmark_config,
                                run_id=run_id,
                                run_dir=run_dir
                            )
                            
                            all_sub_runs.append(sub_run_id)
                            logger.info(f"Completed benchmark {benchmark_type}, sub-run ID: {sub_run_id}")
                            
                            # Update and save run metadata after each sub-run
                            run_metadata["sub_runs"] = all_sub_runs
                            run_metadata["engines"] = list(all_engines)
                            run_metadata["benchmark_types"] = list(all_benchmark_types)
                            with open(run_metadata_path, 'w') as f:
                                json.dump(run_metadata, f, indent=2)
                    
                    # Stop the engine when done with all benchmarks for this engine
                    logger.info(f"Stopping engine {engine.name}")
                    engine_manager.stop_engine(engine.name)
                    
            
            # Update the overall run metadata
            end_time = self._get_current_time()
            
            run_metadata.update({
                "end_time": end_time,
                "duration_seconds": (
                    self._parse_timestamp(end_time) - 
                    self._parse_timestamp(start_time)
                ).total_seconds(),
                "engines": list(all_engines),
                "benchmark_types": list(all_benchmark_types),
                "sub_runs": all_sub_runs,
                "success": True
            })
            
            # Save final run metadata
            with open(run_metadata_path, 'w') as f:
                json.dump(run_metadata, f, indent=2)
            
            return run_metadata
            
        except Exception as e:
            logger.exception(f"Error running benchmarks from YAML: {e}")
            return {
                "id": run_id,
                "start_time": start_time,
                "end_time": self._get_current_time(),
                "engines": list(all_engines),
                "benchmark_types": list(all_benchmark_types),
                "gpu_info": run_metadata.get("gpu_info", {}),
                "success": False,
                "error": str(e),
                "sub_runs": all_sub_runs
            }