import json
import logging
from typing import Dict, List, Any, Optional

from src.engines.base_engine import BaseEngine
from src.engines.vllm_engine import VLLMEngine
from src.engines.sglang_engine import SGLangEngine
from src.engines.tensorrt_engine import TensorRTEngine
from src.utils.docker_utils import (
    run_docker_command, 
    list_running_containers, 
    extract_container_environment_variables,
    extract_container_engine_args
)


logger = logging.getLogger(__name__)


class EngineManager:
    """
    Manages inference engine instances, including creation, starting, stopping, and tracking.
    """
    
    ENGINE_TYPES = {
        "vllm": VLLMEngine,
        "sglang": SGLangEngine,
        # "trt": TensorRTEngine, # TODO: not able to use yet, fix later
        # Add other engine types as they are implemented
    }
    
    def __init__(self):
        """
        Initialize engine manager.
        """
        self.engines: Dict[str, BaseEngine] = {}
    
    def create_engine(self, engine_type: str, model: str, **kwargs) -> Optional[BaseEngine]:
        """
        Create a new engine instance.
        
        Args:
            engine_type: Type of engine to create (e.g., "vllm", "sglang")
            model: Model to use
            **kwargs: Additional engine-specific parameters
            
        Returns:
            Optional[BaseEngine]: Created engine instance or None if creation failed
        """
        engine_type = engine_type.lower()
        if engine_type not in self.ENGINE_TYPES:
            logger.error(f"Unknown engine type: {engine_type}")
            return None
        
        try:
            # Create engine instance
            engine_cls = self.ENGINE_TYPES[engine_type]
            engine = engine_cls(model=model, **kwargs)
            
            # Add to managed engines using name as key
            self.engines[engine.name] = engine
            
            return engine
        except Exception as e:
            logger.exception(f"Failed to create engine: {e}")
            return None
    
    def refresh_engines(self) -> None:
        """
        Efficiently updates the engines dictionary by directly querying Docker.
        Combines the functionality of update_engine_status, _get_engine_from_container,
        and list_engines_from_docker into a single efficient operation.
        """
        # Store existing engines to preserve configs where possible
        existing_engines = self.engines.copy()
        
        # Clear current engines
        self.engines = {}
        
        # Get all running containers in a single Docker call
        containers = list_running_containers()

        for container in containers:
            # Check if container is an inference engine
            is_engine = False
            container_id = container.get("id")
            image = container.get("image", "").lower()
            name = container.get("names", "").lower().replace("/", "")
            
            # Determine engine type from image or name
            engine_type = None

            for key in self.ENGINE_TYPES.keys():
                if (key in image or key in name) and ("_arena" in image or "_arena" in name):
                    engine_type = key
                    is_engine = True
                    break
            
            if not is_engine:
                continue
                
            # Check if we already have this engine (reuse if we do)
            existing_engine = None
            for engine in existing_engines.values():
                if engine.container_id == container_id:
                    existing_engine = engine
                    break
            
            if existing_engine:
                # Update existing engine status and add it back
                existing_engine.status = "running"
                self.engines[existing_engine.name] = existing_engine
                logger.debug(f"Updated existing engine: {existing_engine.name}")
                continue
                
            # Need to create a new engine instance - get more details
            # Get container details in a single Docker call
            cmd = ["docker", "container", "inspect", container_id]
            result = run_docker_command(cmd)
            
            if not result["success"]:
                logger.warning(f"Failed to inspect container {container_id}")
                continue
                
            try:
                container_details = json.loads(result["output"])
                if not container_details or not isinstance(container_details, list):
                    logger.warning(f"Invalid container details for {container_id}")
                    continue
                    
                container_detail = container_details[0]
                
                # Extract model and arguments
                model = "unknown"
                engine_args = {}
                env_vars = {}
                
                # Extract environment variables
                prefix = "SGL" if engine_type.lower() == "sglang" else engine_type
                env_vars = extract_container_environment_variables(container_detail, prefix)
                
                # Try to get model and engine args from command args
                model, engine_args = extract_container_engine_args(container_detail)

                
                # Try to get port mapping
                port = None
                if "NetworkSettings" in container_detail:
                    ports_info = container_detail["NetworkSettings"].get("Ports", {})
                    for container_port, host_bindings in ports_info.items():
                        if host_bindings and isinstance(host_bindings, list):
                            port = host_bindings[0].get("HostPort")
                            break
                            
                # Get engine class for new engine
                engine_cls = self.ENGINE_TYPES[engine_type]
                
                # Create kwargs for engine initialization
                kwargs = {}
                if port:
                    kwargs["port"] = int(port)
                
                # Create the engine instance
                engine = engine_cls(model=model, **kwargs)
                
                # Update engine fields
                engine.name = engine_type
                engine.status = "running"
                engine.container_id = container_id
                
                engine.engine_args = engine_args
                engine.env_vars = env_vars

                # Get full environment variables
                engine.full_env_vars = extract_container_environment_variables(container_detail)
                
                # Get full engine arguments which will also update converted_dtype and converted_quantization
                engine.full_engine_args = engine.get_full_engine_args()
                
                # Add to managed engines
                self.engines[engine.name] = engine
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse container details for {container_id}")
                continue
        
        logger.info(f"Refreshed engines: {len(self.engines)} engines found")
        
    
    def start_engine(self, engine_name: str) -> bool:
        """
        Start an engine.
        
        Args:
            engine_name: Name of the engine to start
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if engine_name not in self.engines:
            logger.error(f"Unknown engine name: {engine_name}")
            return False
        
        engine = self.engines[engine_name]
        
        success = engine.start()
        
        if success:
            logger.info(f"Started engine {engine.name}")
        else:
           logger.error(f"Failed to start engine {engine.name}")
        
        return success
    
    def stop_engine(self, engine_name: str) -> bool:
        """
        Stop an engine.
        
        Args:
            engine_name: Name of the engine to stop
            
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if engine_name not in self.engines:
            logger.error(f"Unknown engine name: {engine_name}")
            return False
        
        engine = self.engines[engine_name]
        success = engine.stop()
        
        if success:
            logger.info(f"Stopped engine {engine.name}")
        else:
            logger.error(f"Failed to stop engine {engine.name}")
        
        return success
    
    def get_engine(self, engine_name: str) -> Optional[BaseEngine]:
        """
        Get an engine by name.
        
        Args:
            engine_name: Name of the engine to get
            
        Returns:
            Optional[BaseEngine]: Engine instance or None if not found
        """
        return self.engines.get(engine_name)
    
    def get_engine_by_name(self, name: str) -> Optional[BaseEngine]:
        """
        Get an engine by name.
        
        Args:
            name: Name of the engine to get
            
        Returns:
            Optional[BaseEngine]: Engine instance or None if not found
        """
        return self.engines.get(name)
    
    def list_engines(self) -> List[Dict[str, Any]]:
        """
        List all managed engines.
        
        Returns:
            List[Dict[str, Any]]: List of engine information dictionaries
        """
        return [engine.to_dict() for engine in self.engines.values()]
