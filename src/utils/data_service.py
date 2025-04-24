import os
import json
from pathlib import Path
import yaml
from typing import Dict, List, Optional, Callable, Set, Any
import threading
from datetime import datetime
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from dotenv import load_dotenv
from pymongo import MongoClient

from .utils import get_final_precision
from .gpu_cost_utils import calculate_cost_per_million_tokens

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_service')

load_dotenv()

# Constants
RESULTS_DIR = Path("./results")

# MongoDB Constants
MONGO_URI = os.environ.get('MONGO_URI')
DB_NAME = os.environ.get('DB_NAME')
MONGO_COLLECTION = "json_data"

ANONYMOUS_AVATAR_URL = "https://p3-pc-sign.douyinpic.com/tos-cn-i-0813/7f9184a875b24dad937a1e1f924b600c~tplv-dy-aweme-images:q75.webp?biz_tag=aweme_images&from=327834062&lk3s=138a59ce&s=PackSourceEnum_SEARCH&sc=image&se=false&x-expires=1747274400&x-signature=FN58S31JYmIHGeOZTQS%2FrygGR2c%3D"

# Global MongoDB client
_mongo_client = None
_mongo_db = None
_mongo_collection = None
_mongo_users_collection = None

# Cache for predefined benchmark configs - load only once at module import time
_PREDEFINED_BENCHMARK_CONFIGS = {}

def _load_predefined_benchmark_configs():
    """Load predefined benchmark configurations from YAML files on module import.
    This is called once when the module is loaded to populate _PREDEFINED_BENCHMARK_CONFIGS.
    """
    configs = {}
    
    # Check a few simple paths in priority order
    possible_paths = [
        Path(__file__).parent / "benchmarks" / "benchmark_configs",  # Most common case
        Path(__file__).parent.parent / "benchmarks" / "benchmark_configs",  # When in installed package
    ]
    
    # Find first existing path
    config_dir = None
    for path in possible_paths:
        if path.exists():
            config_dir = path
            break
    
    if not config_dir:
        logger.warning(f"Benchmark config directory not found: {[str(p) for p in possible_paths]}")
        return configs
    
    for config_file in config_dir.glob("*.yaml"):
        benchmark_name = config_file.stem
        try:
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            configs[benchmark_name] = config
        except Exception as e:
            logger.error(f"Failed to load benchmark config {config_file}: {e}")
    
    return configs

# Initialize configs at module import time
_PREDEFINED_BENCHMARK_CONFIGS = _load_predefined_benchmark_configs()
logger.info(f"Loaded {len(_PREDEFINED_BENCHMARK_CONFIGS)} predefined benchmark configs at module initialization")

def get_mongo_connection():
    """Get or create MongoDB connection"""
    global _mongo_client, _mongo_db, _mongo_collection, _mongo_users_collection
    
    if _mongo_client is None and MONGO_URI and DB_NAME:
        try:
            # Initialize MongoDB connection
            _mongo_client = MongoClient(MONGO_URI)
            _mongo_db = _mongo_client[DB_NAME]
            _mongo_collection = _mongo_db[MONGO_COLLECTION]
            _mongo_users_collection = _mongo_db['users']
            
            # Ping to verify connection
            _mongo_client.admin.command('ping')
            
            logger.info("MongoDB connection established")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {e}")
            # Reset variables in case of error
            _mongo_client = None
            _mongo_db = None
            _mongo_collection = None
            _mongo_users_collection = None
    
    return _mongo_client, _mongo_db, _mongo_collection, _mongo_users_collection

# Initialize MongoDB connection at module load time
get_mongo_connection()

class ResultsFileHandler(FileSystemEventHandler):
    """Handler for file system events related to results files"""
    
    def __init__(self, file_watcher):
        self.file_watcher = file_watcher
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            logger.info(f"File created: {event.src_path}")
            self.file_watcher._notify_listeners(str(event.src_path))
            self.file_watcher._notify_listeners("*")  # Global notification
    
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            logger.info(f"File modified: {event.src_path}")
            self.file_watcher._notify_listeners(str(event.src_path))
            self.file_watcher._notify_listeners("*")  # Global notification
    
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.json'):
            logger.info(f"File deleted: {event.src_path}")
            self.file_watcher._notify_listeners(str(event.src_path))
            self.file_watcher._notify_listeners("*")  # Global notification

class FileWatcher:
    """Watches for file changes and notifies listeners without caching"""
    def __init__(self):
        self.file_watchers: Dict[str, Set[Callable]] = {}
        self.lock = threading.RLock()
        self.observer = None
        self.file_handler = ResultsFileHandler(self)
    
    def start_file_watcher(self):
        """Start watching the results directory for changes"""
        if self.observer is not None and self.observer.is_alive():
            return
            
        if not RESULTS_DIR.exists():
            try:
                RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created results directory: {RESULTS_DIR}")
            except Exception as e:
                logger.error(f"Failed to create results directory: {e}")
                return
        
        self.observer = Observer()
        self.observer.schedule(self.file_handler, str(RESULTS_DIR), recursive=True)
        self.observer.start()
        logger.info(f"Started file watcher for directory: {RESULTS_DIR}")
    
    def stop_file_watcher(self):
        """Stop watching for file changes"""
        if self.observer is not None:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logger.info("Stopped file watcher")
    
    def register_listener(self, file_path: str, callback: Callable):
        """Register a callback for when a file changes"""
        with self.lock:
            if file_path not in self.file_watchers:
                self.file_watchers[file_path] = set()
            self.file_watchers[file_path].add(callback)
    
    def unregister_listener(self, file_path: str, callback: Callable):
        """Unregister a file change callback"""
        with self.lock:
            if file_path in self.file_watchers and callback in self.file_watchers[file_path]:
                self.file_watchers[file_path].remove(callback)
                if not self.file_watchers[file_path]:
                    del self.file_watchers[file_path]
    
    def _notify_listeners(self, file_path: str):
        """Notify all listeners for a given file path"""
        callbacks = set()
        with self.lock:
            if file_path in self.file_watchers:
                callbacks = self.file_watchers[file_path].copy()
            # Also notify global listeners
            if "*" in self.file_watchers:
                callbacks.update(self.file_watchers["*"])
        
        for callback in callbacks:
            try:
                callback(file_path)
            except Exception as e:
                logger.error(f"Error in file change listener callback: {e}")

# Singleton instance
_file_watcher_instance = None

def get_file_watcher_instance() -> FileWatcher:
    """Get the singleton file watcher instance"""
    global _file_watcher_instance
    if _file_watcher_instance is None:
        _file_watcher_instance = FileWatcher()
        _file_watcher_instance.start_file_watcher()
    return _file_watcher_instance

def format_timestamp(timestamp):
    """Format timestamp for display"""
    try:
        dt = datetime.fromisoformat(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except:
        return timestamp

def load_runs(source: str = "local") -> List[Dict]:
    """Load all runs from the specified source"""
    if source == "local":
        # Directly load data from files without caching
        runs = []
        run_map = {}  # Map run IDs to their objects for faster lookup
        
        if not RESULTS_DIR.exists():
            logger.info(f"Results directory does not exist: {RESULTS_DIR}")
            return runs
        
        # Look for run directories
        run_dirs = [d for d in RESULTS_DIR.iterdir() if d.is_dir() and d.name.startswith("run-")]
        
        if not run_dirs:
            logger.info(f"No run directories found in {RESULTS_DIR}")
            return runs
        
        # Process each run directory
        for run_dir in run_dirs:
            # Check run file
            run_file = run_dir / f"{run_dir.name}.json"
            if run_file.exists():
                try:
                    with open(run_file, 'r') as f:
                        data = json.load(f)
                    
                    run_id = data.get("id")
                    if run_id:
                        run_data = data.copy()
                        run_data["subruns"] = []
                        runs.append(run_data)
                        run_map[run_id] = run_data
                except Exception as e:
                    logger.error(f"Error loading run file {run_file}: {e}")
            
            # Check subrun files
            for subrun_dir in [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("sub-run-")]:
                subrun_file = subrun_dir / f"{subrun_dir.name}.json"
                if subrun_file.exists():
                    try:
                        with open(subrun_file, 'r') as f:
                            data = json.load(f)
                        
                        parent_id = data.get("parent_run_id")
                        if parent_id in run_map:
                            run_map[parent_id]["subruns"].append(data)
                    except Exception as e:
                        logger.error(f"Error loading subrun file {subrun_file}: {e}")
        
        return sorted(runs, key=lambda r: r.get("start_time", ""), reverse=True)
    else:  # global
        # Get existing MongoDB connection or create a new one
        client, db, collection, users_collection = get_mongo_connection()
        
        if not client:
            logger.error("Could not establish MongoDB connection")
            return []
        
        try:
            # Query MongoDB for all JSON data - get everything directly
            results = list(collection.find({}))
            
            # Get a list of all unique client_ids from the results
            client_ids = set()
            for doc in results:
                client_id = doc.get('client_id')
                if client_id:
                    client_ids.add(client_id)
            
            # Fetch user data for all client_ids
            user_data = {}
            if client_ids:
                user_docs = list(users_collection.find({'client_id': {'$in': list(client_ids)}}))                
                for user in user_docs:
                    user_data[user['client_id']] = {
                        'avatar_url': user.get('avatar_url', ''),
                        'login': user.get('login', '')
                    }
            
            # Process results and extract data from nested 'data' field if present
            processed_results = []
            for doc in results:
                # Extract the actual data from the nested 'data' field if present
                run_data = doc.get('data', doc)  # Use document directly if no 'data' field
                run_data['upload_datetime'] = doc.get('upload_datetime')
                # Add user details to the run data
                client_id = doc.get('client_id')
                if client_id and client_id in user_data:
                    run_data['avatar_url'] = user_data[client_id]['avatar_url']
                    run_data['client_login'] = user_data[client_id]['login']
                else:
                    # Use anonymous fallbacks if client_id is None or not found in user_data
                    run_data['avatar_url'] = ANONYMOUS_AVATAR_URL
                    run_data['client_login'] = "Anonymous"
                
                processed_results.append(run_data)
            
            return sorted(processed_results, key=lambda r: r.get("start_time", ""), reverse=True)
            
        except Exception as e:
            logger.error(f"Error loading global runs from MongoDB: {e}")
            return []

def get_predefined_benchmark_configs() -> Dict[str, Dict[str, Any]]:
    """Get the predefined benchmark configurations.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary of benchmark configs
    """
    # Simply return the already loaded configs
    return _PREDEFINED_BENCHMARK_CONFIGS

def is_predefined_benchmark(benchmark_type: str, benchmark_config: Dict[str, Any]) -> bool:
    """Check if a benchmark is predefined (vs custom).
    
    Args:
        benchmark_type: The type of benchmark
        benchmark_config: The configuration of the benchmark
        
    Returns:
        bool: True if benchmark is predefined, False if custom
    """
    predefined_configs = get_predefined_benchmark_configs()
    
    # First check if the benchmark type exists in predefined configs
    if benchmark_type not in predefined_configs:
        return False
    
    # If the type matches, compare the configurations
    predefined_config = predefined_configs[benchmark_type]
    
    # Check key parameters to determine if this is the same benchmark
    key_params = ['dataset-name', 'random-input-len', 'random-output-len', 'random-prefix-len', 'num-prompts', 'request-rate'] # TODO: fix this later, need to check all the keys, the predefined configs are also not complete
    
    for param in key_params:
        # Skip if parameter doesn't exist in either config
        if param not in predefined_config and param not in benchmark_config:
            continue
            
        # If parameter exists in one but not the other, or values don't match
        if param not in predefined_config or param not in benchmark_config:
            return False
            
        if predefined_config[param] != benchmark_config[param]:
            return False
    
    return True

def _passes_filters(doc, model_filter, engine_filter, benchmark_filter, gpu_filter, show_custom_benchmarks):
    # Apply filters - ignore filters if they're set to "All X"
    if model_filter and model_filter != "All Models" and doc.get("model") != model_filter:
        return False
    
    if engine_filter and engine_filter != "All Engines" and doc.get("engine", {}).get("name") != engine_filter:
        return False

    if benchmark_filter and benchmark_filter != "All Benchmarks" and doc.get("benchmark", {}).get("type") != benchmark_filter:
        return False
    
    # Extract GPU info
    gpus = doc.get("engine", {}).get("gpu_info", {}).get("gpus", [])
    gpu_info = "Unknown"
    if gpus:
        gpu_info = f"{gpus[0].get('name', 'Unknown')} ({len(gpus)}x)"
    
    if gpu_filter and gpu_filter != "All GPUs" and gpu_info != gpu_filter:
        return False
    
    benchmark_type = doc.get("benchmark", {}).get("type", "Unknown")
    benchmark_config = doc.get("benchmark", {}).get("config", {})
    
    # Skip custom benchmarks if not showing them
    if not show_custom_benchmarks and not is_predefined_benchmark(benchmark_type, benchmark_config):
        return False
    
    return True

def _create_leaderboard_entry(doc):
    # Create a leaderboard entry with the important metrics
    precision = get_final_precision(
        doc.get("engine", {}).get("converted_dtype"),
        doc.get("engine", {}).get("converted_quantization")
    )
    
    # Extract GPU info
    gpus = doc.get("engine", {}).get("gpu_info", {}).get("gpus", [])
    num_gpus = len(gpus)
    gpu_name = gpus[0].get("name", "Unknown") if gpus else "Unknown"
    gpu = f"{gpu_name} ({num_gpus}x)"
    
    input_throughput = doc.get("metrics", {}).get("input_throughput", 0)
    output_throughput = doc.get("metrics", {}).get("output_throughput", 0)

    input_cost_per_million = None
    output_cost_per_million = None
    
    if gpu_name and input_throughput > 0:
        input_cost_per_million = calculate_cost_per_million_tokens(
            gpu_type=gpu_name,
            num_gpus=num_gpus,
            throughput_tokens_per_second=input_throughput
        )
    
    if gpu_name and output_throughput > 0:
        output_cost_per_million = calculate_cost_per_million_tokens(
            gpu_type=gpu_name,
            num_gpus=num_gpus,
            throughput_tokens_per_second=output_throughput
        )
    
    # Calculate per request throughput
    tpot = doc.get("metrics", {}).get("tpot", 0)
    per_request_throughput = 1000 / tpot if tpot > 0 else None
    
    entry = {
        "id": doc.get("id"),
        "parent_run_id": doc.get("parent_run_id"),
        "model": doc.get("model", "Unknown"),
        "engine": doc.get("engine", {}).get("name", "Unknown"),
        "engine_converted_dtype": doc.get("engine", {}).get("converted_dtype", "Unknown"),
        "engine_converted_quantization": doc.get("engine", {}).get("converted_quantization", "Unknown"),
        "precision": precision,
        "engine_args": doc.get("engine", {}).get("engine_args", {}),
        "env_vars": doc.get("engine", {}).get("env_vars", {}),
        "full_engine_args": doc.get("engine", {}).get("full_engine_args", {}),
        "full_env_vars": doc.get("engine", {}).get("full_env_vars", {}),
        "gpu": gpu,
        "benchmark_type": doc.get("benchmark", {}).get("type", "Unknown"),
        "benchmark_config": doc.get("benchmark", {}).get("config", {}),
        "is_predefined_benchmark": is_predefined_benchmark(doc.get("benchmark", {}).get("type", "Unknown"), doc.get("benchmark", {}).get("config", {})),
        "input_throughput": input_throughput,
        "output_throughput": output_throughput,
        "input_cost_per_million": input_cost_per_million,
        "output_cost_per_million": output_cost_per_million,
        "ttft": doc.get("metrics", {}).get("ttft", 0),
        "tpot": doc.get("metrics", {}).get("tpot", 0),
        "per_request_throughput": per_request_throughput,
        "success": doc.get("success", False),
        "start_time": doc.get("start_time", ""),
        "timestamp": format_timestamp(doc.get("start_time", "")),
        "upload_datetime": doc.get("upload_datetime", "").strftime("%Y-%m-%d %H:%M") if isinstance(doc.get("upload_datetime", ""), datetime) else "",
        "avatar_url": doc.get("avatar_url", ""),
        "client_login": doc.get("client_login", "")
    }

    return entry

def get_filtered_data(source: str = "local", model_filter: Optional[str] = None, engine_filter: Optional[str] = None, benchmark_filter: Optional[str] = None, gpu_filter: Optional[str] = None, show_custom_benchmarks: bool = False) -> Dict[str, List[str]]:
    """Get all filter data (models, engines, benchmarks, GPUs) in one function call"""
    runs = load_runs(source=source)
        
    # Initialize sets for each filter type
    models = set()
    engines = set()
    benchmark_types = set()
    gpu_types = set()
    leaderboard_entries = []
    
    if source == "local":
        # Process local runs which have a nested structure
        for run in runs:
            # Extract model
            if "model" in run and run["model"]:
                models.add(run["model"])
            
            # Extract engine
            engine_name = run.get("engine", {}).get("name")
            if engine_name:
                engines.add(engine_name)
            
            # Extract benchmark type
            benchmark_type = run.get("benchmark", {}).get("type")
            if benchmark_type:
                benchmark_types.add(benchmark_type)
            
            # Extract GPU info
            gpus = run.get("engine", {}).get("gpu_info", {}).get("gpus", [])
            if gpus:
                gpu_info = f"{gpus[0].get('name', 'Unknown')} ({len(gpus)}x)"
                gpu_types.add(gpu_info)
            
            # Process subruns if any
            for subrun in run.get("subruns", []):
                # Extract model from subrun
                if "model" in subrun and subrun["model"]:
                    models.add(subrun["model"])
                
                # Extract engine from subrun
                engine_name = subrun.get("engine", {}).get("name")
                if engine_name:
                    engines.add(engine_name)
                
                # Extract benchmark type from subrun
                benchmark_type = subrun.get("benchmark", {}).get("type")
                if benchmark_type:
                    benchmark_types.add(benchmark_type)
                
                # Extract GPU info from subrun
                gpus = subrun.get("engine", {}).get("gpu_info", {}).get("gpus", [])
                if gpus:
                    gpu_info = f"{gpus[0].get('name', 'Unknown')} ({len(gpus)}x)"
                    gpu_types.add(gpu_info)

        for run in runs:
            for subrun in run.get("subruns", []):
                # Skip if filtered out
                if not _passes_filters(subrun, model_filter, engine_filter, benchmark_filter, gpu_filter, show_custom_benchmarks):
                    continue
                        
                entry = _create_leaderboard_entry(subrun)
                leaderboard_entries.append(entry)
                
    else:
        # For MongoDB data, each document is processed directly
        for doc in runs:
            # Extract model
            if "model" in doc and doc["model"]:
                models.add(doc["model"])
            
            # Extract engine
            engine_name = doc.get("engine", {}).get("name")
            if engine_name:
                engines.add(engine_name)
            
            # Extract benchmark type
            benchmark_type = doc.get("benchmark", {}).get("type")
            if benchmark_type:
                benchmark_types.add(benchmark_type)
            
            # Extract GPU info
            gpus = doc.get("engine", {}).get("gpu_info", {}).get("gpus", [])
            if gpus:
                gpu_info = f"{gpus[0].get('name', 'Unknown')} ({len(gpus)}x)"
                gpu_types.add(gpu_info)

        for doc in runs:
            if not _passes_filters(doc, model_filter, engine_filter, benchmark_filter, gpu_filter, show_custom_benchmarks):
                continue
                
            entry = _create_leaderboard_entry(doc)
            leaderboard_entries.append(entry)
            
    sorted_entries = sorted(leaderboard_entries, key=lambda e: e["output_throughput"], reverse=True)

    # Return the results as sorted lists
    return {
        "models": sorted(list(models)),
        "engines": sorted(list(engines)),
        "benchmarks": sorted(list(benchmark_types)),
        "gpus": sorted(list(gpu_types)),
        "leaderboard": sorted_entries
    }