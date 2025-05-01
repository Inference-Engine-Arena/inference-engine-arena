import sys
import argparse
import logging
import glob
import os
from typing import List

from src.engines.engine_manager import EngineManager
from src.benchmarks.benchmark_runner import BenchmarkRunner
from src.utils.upload_leaderboard import upload_json_file
from src.utils.docker_utils import get_container_logs, stream_container_logs
import threading

logger = logging.getLogger(__name__)


def setup_logging(level: int = logging.INFO):
    """Set up logging configuration."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )


class ArenaCommand:
    """Base class for all arena commands."""
    
    def __init__(self):
        self.engine_manager = EngineManager()
        self.benchmark_runner = BenchmarkRunner()
    
    def parse_args(self, args: List[str]) -> argparse.Namespace:
        """Parse command line arguments."""
        parser = argparse.ArgumentParser(description="Inference Engine Arena CLI")
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")
        
        # Register commands
        self._register_start_command(subparsers)
        self._register_stop_command(subparsers)
        self._register_list_command(subparsers)
        self._register_run_command(subparsers)
        self._register_runyaml_command(subparsers)
        self._register_logs_command(subparsers)
        self._register_dashboard_command(subparsers)
        self._register_leaderboard_command(subparsers)
        self._register_upload_command(subparsers)
        
        if not args:
            parser.print_help()
            sys.exit(1)
        
        return parser.parse_args(args)
    
    def _register_start_command(self, subparsers):
        """Register the 'start' command."""
        start_parser = subparsers.add_parser("start", help="Start an inference engine")
        start_parser.add_argument("engine", help="Type of engine to start (vllm, sglang, etc.)")
        start_parser.add_argument("model", help="Model to use")
        # Allow for any additional arguments to be passed through
        start_parser.add_argument("args", nargs=argparse.REMAINDER, help="Additional engine-specific arguments")
    
    def _register_stop_command(self, subparsers):
        """Register the 'stop' command."""
        stop_parser = subparsers.add_parser("stop", help="Stop an inference engine")
        stop_parser.add_argument("engine", help="Type of engine to start (vllm, sglang, etc.)")
    
    def _register_list_command(self, subparsers):
        """Register the 'list' command."""
        list_parser = subparsers.add_parser("list", help="List running engines")
    
    def _register_run_command(self, subparsers):
        """Register the 'run' command."""
        run_parser = subparsers.add_parser("run", help="Run benchmarks")
        run_parser.add_argument("--engine", nargs="+", required=True, help="Engines to benchmark")
        run_parser.add_argument("--benchmark", nargs="+", required=True, help="Benchmark types to run")
    
    def _register_runyaml_command(self, subparsers):
        """Register the 'runyaml' command."""
        yaml_parser = subparsers.add_parser("runyaml", help="Run benchmarks from YAML configuration files")
        yaml_parser.add_argument("config", nargs='+', help="Path(s) to the YAML configuration file(s)")
    
    def _register_logs_command(self, subparsers):
        """Register the 'logs' command."""
        logs_parser = subparsers.add_parser("logs", help="Stream logs from an engine container")
        logs_parser.add_argument("engine", help="Type of engine to start (vllm, sglang, etc.)")
        logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")
        logs_parser.add_argument("--tail", type=int, default=100, help="Number of lines to show from the end")
    
    def _register_dashboard_command(self, subparsers):
        """Register the 'dashboard' command."""
        dashboard_parser = subparsers.add_parser("dashboard", help="Launch the leaderboard dashboard")
        dashboard_parser.add_argument("--port", type=int, default=3001, help="Starting port to run the dashboard on")
        dashboard_parser.add_argument("--port-end", type=int, default=3015, help="End of port range to try if starting port is unavailable")
        dashboard_parser.add_argument("--no-share", action="store_true", help="Disable Gradio sharing")
    
    def _register_leaderboard_command(self, subparsers):
        """Register the 'leaderboard' command."""
        leaderboard_parser = subparsers.add_parser("leaderboard", help="Launch the leaderboard dashboard")
        leaderboard_parser.add_argument("--port", type=int, default=3001, help="Starting port to run the leaderboard on")
        leaderboard_parser.add_argument("--port-end", type=int, default=3015, help="End of port range to try if starting port is unavailable")
        leaderboard_parser.add_argument("--no-share", action="store_true", help="Disable Gradio sharing")
    
    def _register_upload_command(self, subparsers):
        """Register the 'upload' command."""
        upload_parser = subparsers.add_parser("upload", help="Upload benchmark results to the leaderboard")
        upload_parser.add_argument("files", nargs="*", help="JSON files to upload (default: all JSON files in results folder)")
        upload_parser.add_argument("--no-login", action="store_true", help="Bypass login requirement")
    
    def execute(self, args: List[str]) -> int:
        """Execute a command with the given arguments."""
        parsed_args = self.parse_args(args)
        
        # Update engine status before every command
        self.engine_manager.refresh_engines()        
        
        # Command handling mapping
        handlers = {
            "start": self._handle_start,
            "stop": self._handle_stop,
            "list": self._handle_list,
            "run": self._handle_run,
            "runyaml": self._handle_runyaml,
            "logs": self._handle_logs,
            "dashboard": self._handle_dashboard,
            "leaderboard": self._handle_leaderboard,
            "upload": self._handle_upload
        }

        # Dispatch to the corresponding handler.
        if parsed_args.command in handlers:
            return handlers[parsed_args.command](parsed_args)
        else:
            logger.error(f"Unknown command: {parsed_args.command}")
            return 1
    
    def _handle_start(self, args: argparse.Namespace) -> int:
        """Handle the 'start' command."""
        engine_type = args.engine.lower()
        model = args.model

        
        # Check if an engine with the same name already exists
        # In engine_manager.py, engines are stored in a dictionary with engine names as keys
        for name, engine in self.engine_manager.engines.items():
            if name == engine_type and engine.status == "running":
                logger.warning(f"An engine with name '{name}' is already running")
                logger.error(f"Cannot start a new engine with the same name. Please stop the existing one first with 'arena stop {name}'")
                return 1
        
        # Process standard parameters
        kwargs = {}
        
        # Process additional arguments in a vLLM-compatible format
        if args.args:
            i = 0
            while i < len(args.args):
                arg = args.args[i]
                
                # Handle args that start with - or --
                if arg.startswith("-"):
                    # Strip leading dashes 
                    key = arg.lstrip("-")
                    
                    # Check if the next argument is a value or another flag
                    if i + 1 < len(args.args) and not args.args[i + 1].startswith("-"):
                        # Next item is a value
                        value = args.args[i + 1]
                        
                        # Try to convert to appropriate type
                        if value.isdigit():
                            value = int(value)
                        elif value.lower() in ("true", "false"):
                            value = value.lower() == "true"
                        
                        kwargs[key] = value
                        i += 2
                    else:
                        # Flag without value - treat as boolean flag
                        kwargs[key] = True
                        i += 1
                else:
                    # Skip non-flag arguments
                    i += 1
        
        # Create and start engine
        engine = self.engine_manager.create_engine(engine_type, model, **kwargs)

        # print("engine in commands.py", engine.to_dict())
        if not engine:
            logger.error(f"Failed to create {engine_type} engine")
            return 1
        
        logger.info(f"Created engine: {engine.name}")
        
        # Start the engine
        success = self.engine_manager.start_engine(engine.name)
        if not success:
            logger.error(f"Failed to start engine: {engine.name}")
            return 1
        
        logger.info(f"Engine started successfully: {engine.name}")
        logger.info(f"Engine endpoint: {engine.get_endpoint()}")
        
        return 0
    
    def _handle_stop(self, args: argparse.Namespace) -> int:
        """Handle the 'stop' command."""
        engine_name = args.engine.lower()
        
        # Try to find by name
        engine = self.engine_manager.get_engine_by_name(engine_name)
        
        if not engine:
            logger.error(f"Unknown engine: {engine_name}")
            return 1
        
        success = self.engine_manager.stop_engine(engine.name)
        if not success:
            logger.error(f"Failed to stop engine: {engine.name}")
            return 1
        
        logger.info(f"Stopped engine: {engine.name}")
        return 0
    
    def _handle_logs(self, args: argparse.Namespace) -> int:
        """Handle the 'logs' command."""
        engine_name = args.engine.lower()
        
        # Try to find by name
        engine = self.engine_manager.get_engine_by_name(engine_name)
        
        if not engine:
            logger.error(f"Unknown engine: {engine_name}")
            return 1
        
        if not engine.container_id:
            logger.error(f"Engine {engine.name} is not running or has no container ID")
            return 1
        
        if args.follow:
            # Stream logs in real-time
            try:
                print(f"Streaming logs for {engine.name} (Ctrl+C to stop)...")
                stop_event = threading.Event()
                stream_container_logs(engine.container_id, stop_event)
            except KeyboardInterrupt:
                print("\nStopped streaming logs")
        else:
            # Get a specific number of log lines
            logs = get_container_logs(engine.container_id, args.tail)
            print(logs)
        
        return 0
    
    def _handle_list(self, args: argparse.Namespace) -> int:
        """Handle the 'list' command."""
        engines = self.engine_manager.list_engines()
        
        if not engines:
            logger.info("No engines found")
            return 0
        
        # Display engines with more information by default
        print(f"\nInference Engines ({len(engines)}):\n")
        print(f"{'Name':<15} {'Container ID':<15} {'Model':<25} {'Status':<10} {'Endpoint':<30} {'GPU Info'}")
        print("-" * 110)
        
        for engine in engines:
            endpoint = engine.get("endpoint", "N/A") if engine.get("status") == "running" else "N/A"
            container_id = engine.get("container_id", "N/A")
            
            gpu_info = "N/A"
            if engine.get("status") == "running" and "gpu_info" in engine and "gpus" in engine["gpu_info"]:
                gpus = engine["gpu_info"]["gpus"]
                if gpus:
                    gpu_name = gpus[0]['name']
                    gpu_count = len(gpus)
                    gpu_info = f"{gpu_name} (x{gpu_count})" if gpu_count > 1 else gpu_name
            
            print(f"{engine['name']:<15} {container_id:<15} {engine['model'][:25]:<25} {engine['status']:<10} {endpoint:<30} {gpu_info}")
        
        return 0
    
    def _handle_run(self, args: argparse.Namespace) -> int:
        """Handle the 'run' command."""
        engine_names = args.engine
        benchmark_types = args.benchmark
        
        # Validate engines
        engines = []
        for engine_name in engine_names:
            engine = self.engine_manager.get_engine_by_name(engine_name.lower())
            if not engine:
                logger.error(f"Unknown engine: {engine_name}")
                return 1
            if engine.status != "running":
                logger.error(f"Engine {engine_name} is not running")
                return 1
            engines.append(engine)
        
        # Validate benchmarks
        valid_benchmarks = []
        for benchmark_type in benchmark_types:
            if benchmark_type not in self.benchmark_runner.benchmark_configs:
                logger.warning(f"Unknown benchmark type: {benchmark_type}, skipping")
            else:
                valid_benchmarks.append(benchmark_type)
        
        if not valid_benchmarks:
            logger.error("No valid benchmark types specified")
            return 1
        
        # Run benchmarks
        logger.info(f"Running {len(valid_benchmarks)} benchmark(s) on {len(engines)} engine(s)")
        result = self.benchmark_runner.run_benchmark(
            engines=engines,
            benchmark_types=valid_benchmarks
        )
        
        # Display result summary
        print("\nBenchmark Run Summary:")
        print(f"  Run ID: {result['id']}")
        print(f"  Start time: {result['start_time']}")
        print(f"  End time: {result['end_time']}")
        print(f"  Duration: {result['duration_seconds']:.1f} seconds")
        print(f"  Engines: {', '.join(result['engines'])}")
        print(f"  Benchmark types: {', '.join(result['benchmark_types'])}")
        print(f"  Sub-runs: {len(result['sub_runs'])}")
        print(f"\nResults saved to: ./results/{result['id']}")
        
        return 0
    
    def _handle_runyaml(self, args: argparse.Namespace) -> int:
        """Handle the 'runyaml' command."""
        yaml_files = args.config
        
        overall_success = True
        run_summaries = []
        
        for yaml_file in yaml_files:
            logger.info(f"Processing YAML file: {yaml_file}")
            
            if not os.path.exists(yaml_file):
                logger.error(f"YAML file not found: {yaml_file}")
                overall_success = False
                continue
                
            # Use the run_benchmark_from_yaml method in BenchmarkRunner
            result = self.benchmark_runner.run_benchmark_from_yaml(
                yaml_file=yaml_file,
                engine_manager=self.engine_manager
            )
            
            if not result.get("success", False):
                logger.error(f"Failed to run benchmarks from YAML: {result.get('error', 'Unknown error')}")
                overall_success = False
            
            run_summaries.append(result)
            
            # Display result summary
            print("\nBenchmark Run Summary:")
            print(f"  Run ID: {result['id']}")
            print(f"  Start time: {result['start_time']}")
            print(f"  End time: {result['end_time']}")
            print(f"  Duration: {result['duration_seconds']:.1f} seconds")
            print(f"  Engines: {', '.join(result['engines'])}")
            print(f"  Benchmark types: {', '.join(result['benchmark_types'])}")
            print(f"  Sub-runs: {len(result['sub_runs'])}")
            print(f"\nResults saved to: ./results/{result['id']}")
        
        # Overall summary
        if len(yaml_files) > 1:
            print("\n" + "=" * 80)
            print(f"OVERALL SUMMARY: Processed {len(yaml_files)} YAML files")
            print(f"  Overall status: {'SUCCESS' if overall_success else 'FAILURE'}")
            print(f"  Total runs: {len(run_summaries)}")
            print("=" * 80 + "\n")
        
        return 0 if overall_success else 1
    
    def _handle_dashboard(self, args: argparse.Namespace) -> int:
        """Handle the 'dashboard' command."""
        import subprocess
        
        # Build command with appropriate arguments
        cmd = [sys.executable, "dashboard/app.py"]
        
        # Add optional arguments if provided
        if hasattr(args, "port") and args.port:
            cmd.extend(["--port", str(args.port)])
        if hasattr(args, "port_end") and args.port_end:
            cmd.extend(["--port-end", str(args.port_end)])
        if hasattr(args, "no_share") and args.no_share:
            cmd.append("--no-share")
        
        # Execute the dashboard command
        logger.info(f"Starting dashboard: {' '.join(cmd)}")
        try:
            # Call directly without creating a new process
            return subprocess.call(cmd)
        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            return 1

    def _handle_leaderboard(self, args: argparse.Namespace) -> int:
        """Handle the 'leaderboard' command."""
        import subprocess
        
        # Build command with appropriate arguments
        cmd = [sys.executable, "leaderboard/app.py"]
        
        # Add optional arguments if provided
        if hasattr(args, "port") and args.port:
            cmd.extend(["--port", str(args.port)])
        if hasattr(args, "port_end") and args.port_end:
            cmd.extend(["--port-end", str(args.port_end)])
        if hasattr(args, "no_share") and args.no_share:
            cmd.append("--no-share")
        
        # Execute the leaderboard command
        logger.info(f"Starting leaderboard: {' '.join(cmd)}")
        try:
            # Call directly without creating a new process
            return subprocess.call(cmd)
        except Exception as e:
            logger.error(f"Failed to start leaderboard: {e}")
            return 1


    def _handle_upload(self, args: argparse.Namespace) -> int:
        """Handle the 'upload' command."""
        # Determine login status
        login_required = not args.no_login
        
        # If no files specified, find subrun files in results directory
        if not args.files:
            results_dir = os.path.join(os.getcwd(), "results")
            if os.path.exists(results_dir):
                # Get all JSON files in results directory
                all_json_files = glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True)
                
                json_files = [f for f in all_json_files if "sub-run" in os.path.basename(f)]
                
                if not json_files:
                    logger.error(f"No subrun JSON files found in the results directory: {results_dir}")
                    return 1
                
                logger.info(f"Found {len(json_files)} subrun files to upload")
            else:
                logger.error(f"Results directory not found: {results_dir}")
                return 1
        else:
            json_files = args.files
            # Validate that all specified files exist
            for file_path in json_files:
                if not os.path.exists(file_path):
                    logger.error(f"File not found: {file_path}")
                    return 1
        
        # Upload each file
        success_count = 0
        for file_path in json_files:
            logger.info(f"Uploading {file_path}...")
            success, message = upload_json_file(file_path, login=login_required)
            if success:
                logger.info(f"Successfully uploaded {file_path}: {message}")
                success_count += 1
            else:
                logger.error(f"Failed to upload {file_path}: {message}")
        
        logger.info(f"Upload complete. {success_count}/{len(json_files)} files uploaded successfully.")
        return 0 if success_count > 0 else 1


def main():
    """Main entry point for the CLI."""
    setup_logging()
    command = ArenaCommand()
    return command.execute(sys.argv[1:])


if __name__ == "__main__":
    sys.exit(main())