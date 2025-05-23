import os
import sys
import gradio as gr
import time
import pandas as pd
import numpy as np
import argparse
import logging
import json
import socket
import plotly.graph_objects as go

# Configure logging
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now we can import from src
from src.utils.data_service import (
    get_file_watcher_instance, get_filtered_data
)
from src.utils.utils import get_project_version

GITHUB_REPO_URL = "https://github.com/Inference-Engine-Arena/inference-engine-arena"
GLOBAL_LEADERBOARD_URL = "https://iearena.org/"
PROJECT_VERSION = get_project_version()

# CSS for leaderboard UI
CSS = """
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}
img {
    vertical-align: middle;
    object-fit: cover;
}
.title {
    text-align: center;
    font-size: 2.5rem;
    font-weight: 600;
    margin-bottom: 1rem;
}
.subtitle {
    text-align: center;
    font-size: 1.3rem;
    margin-bottom: 2rem;
    color: #666;
}
.timer {
    text-align: right;
    font-size: 0.9rem;
    color: #888;
    margin-top: 0.5rem;
}
.filter-section {
    padding: 1rem;
    background-color: #f9f9f9;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
}
.score-cell-high {
    font-weight: bold;
    color: #28a745;
}
.score-cell-low {
    font-weight: bold;
    color: #dc3545;
}
.toggle-btn {
    padding: 0.4rem 0.8rem;
    background-color: #f0f0f0;
    border: 1px solid #ddd;
    border-radius: 0.3rem;
    margin-right: 0.5rem;
    cursor: pointer;
    font-size: 0.9rem;
}
.toggle-btn.active {
    background-color: #007bff;
    color: white;
    border-color: #0069d9;
}
.leaderboard-container {
    margin-top: 1rem;
    overflow-x: auto;
    max-height: 900px;  
    overflow-y: auto;  
    border: 1px solid #eee;  
}
.plot-container {
    margin-top: 2rem;
    margin-bottom: 2rem;
    background-color: #000;
}
.plot-title {
    color: white;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 600;
    margin-bottom: 1rem;
}
.plot-subtitle {
    color: #aaa;
    text-align: center;
    font-size: 1.1rem;
    margin-bottom: 1.5rem;
}
.plot-axes-labels {
    color: #aaa;
    text-align: center;
    font-size: 0.9rem;
    # margin-top: 0.5rem;
}
table {
    width: 100%;
    border-collapse: collapse;
    min-width: 1200px;
}
th {
    background-color: #f1f1f1;
    padding: 8px;
    text-align: left;
    border-bottom: 1px solid #ddd;
    font-weight: 600;
    position: sticky; 
    top: 0; 
    z-index: 10; 
}
td {
    padding: 8px;
    border-bottom: 1px solid #ddd;
}
tr:hover {
    background-color: #f5f5f5;
}
.footnote {
    font-size: 0.8rem;
    color: #666;
    margin-top: 1rem;
    text-align: center;
}
.app {
  max-width: 100% !important;
  padding-left: 5% !important;
  padding-right: 5% !important;
}
.custom-benchmark {
  color: #007bff;
  font-style: italic;
}
.advanced-filter-section {
    padding: 1rem;
    background-color: #f5f5f5;
    border-radius: 0.5rem;
    margin-bottom: 1rem;
    margin-top: 0.5rem;
    width: 50%;
    float: left;
}
.advanced-filters-container {
    display: flex;
    flex-direction: row;
    background-color: #f5f5f5;
    gap: 1rem;
}
.advanced-filter-box {
    flex: 1;
    padding: 0.8rem;
    background-color: #f5f5f5;
}
.advanced-filter-title {
    font-weight: 600;
    margin-bottom: 0.5rem;
    font-size: 1rem;
    color: #333;
}
.filter-hint {
    font-size: 0.8rem;
    color: #666;
    margin-top: 0.5rem;
    font-style: italic;
}
.search-button-container {
    display: flex;
    justify-content: flex-start;
    margin-top: 1rem;
}
.search-button {
    background-color: var(--primary-500);
    color: white;
    border: none;
    border-radius: 0.25rem;
    font-weight: 500;
    cursor: pointer;
}
.search-button:hover {
    background-color: #0069d9;
}

/* Top 5 Chart Container Styles */
#top5-chart-container {
    display: flex;
    justify-content: center;
    align-items: center;
    width: 100%;
    margin: 0 auto;
    padding: 20px;
    background-color: rgba(248, 249, 250, 0.5);
    border-radius: 8px;
    overflow: hidden !important;
    position: relative;
    box-sizing: border-box;
    border: 1px solid #e9ecef;
}

#top5-chart-container, #top5-chart-container > * {
    background-color: rgba(248, 249, 250, 0.5) !important;
    border: 1px solid #e9ecef !important;
}

#top5-plot {
    width: 100% !important;
    margin: 0 auto;
    transition: all 0.3s ease;
    overflow: hidden !important;
    background-color: #F8F9FA;
}

#top5-plot .svg-container {
    width: 100% !important;
    overflow: hidden !important;
}

/* Ensure chart content doesn't overflow container */
.js-plotly-plot .plotly {
    overflow: hidden !important;
}

/* Beautify scrollbar styles */
.leaderboard-container::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

.leaderboard-container::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 4px;
}

.leaderboard-container::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}

.leaderboard-container::-webkit-scrollbar-thumb:hover {
    background: #555;
}
"""

# Dictionary of metric definitions for reuse across the application
METRIC_DEFINITIONS = {
    "Input Throughput (tokens/s)": '''This measures how many input tokens the system can process per second. It represents the rate at which the model can ingest and process incoming text from users.''',
    
    "Input $/1M tokens": '''
This is the cost to process 1 million input tokens (the text sent to the model). This pricing metric helps users estimate costs for processing their prompts or input data.
Example calculation:

System cost: $2/hour
Input processing speed: 800 tokens/s
Cost per second: $2/hr ÷ 3600s/hr = $0.000556/s
Tokens processed per second: 800
Cost per token: $0.000556/s ÷ 800 tokens/s = $0.000000695/token
Cost per million tokens: $0.000000695 × 1,000,000 = $0.695/1M tokens
''',
    
    "Output Throughput (tokens/s)": '''Total TPS per system represents the total output tokens per seconds throughput, accounting for all the requests happening simultaneously. As the number of requests increases, the total TPS per system increases, until it reaches a saturation point for all the available GPU compute resources, beyond which it might decrease.''',
    
    "Output $/1M tokens": '''
This is the cost to generate 1 million output tokens (the text produced by the model). Output tokens typically cost more than input tokens because generation is more computationally intensive than processing.
Example calculation:

System cost: $2/hour
Output generation speed: 400 tokens/s
Cost per second: $2/hr ÷ 3600s/hr = $0.000556/s
Tokens generated per second: 400
Cost per token: $0.000556/s ÷ 400 tokens/s = $0.00000139/token
Cost per million tokens: $0.00000139 × 1,000,000 = $1.39/1M tokens
''',
    
    "TTFT (ms)": '''This metric shows how long a user needs to wait before seeing the model's output. This is the time it takes from submitting the query to receiving the first token (if the response is not empty).
            Time to first token generally includes both request queuing time, prefill time and network latency. The longer the prompt, the larger the TTFT. This is because the attention mechanism requires the whole input sequence to compute and create the so-called key-value cache (aka.KV-cache), from which point the iterative generation loop can begin. Additionally, a production application can have several requests in progress, therefore one request's prefill phase may overlap with another request's generation phase.''',
    
    "TPOT (ms)": '''Time per output token (TPOT), This is defined as the average time between consecutive tokens and is also known as inter token latency (ITL).
            It is important to note that with longer output sequences, the KV cache grows and hence the memory cost. The cost of attention computation also grows: for each new token, this cost is linear in the length of the input + output sequence so far (but this computation is generally not compute-bound). Consistent inter-token latencies signifies an efficient memory management, better memory bandwidth as well as efficient attention computation.''',
    
    "TPS for 1 User (tokens/s/req)": '''TPS per user represents throughput from a single user perspective, and defined as (Output sequence length)/(e2e_latency) for each user's request, which asymptotically approaches 1/ITL as the output sequence length increases. Note that as the number of concurrent requests increases in the system, the total TPS for the whole system increases, while TPS per user decreases as latency becomes worse.'''
}

class LeaderboardState:
    def __init__(self):
        self.source = os.environ.get("LEADERBOARD_SOURCE", "local")
        self.last_update_time = 0
        self.file_watcher = get_file_watcher_instance()
        self.is_refreshing = False  # Flag to prevent recursive refreshes
        
        # Initialize filters
        self.model_filter = None
        self.engine_filter = None
        self.benchmark_filter = None
        self.gpu_filter = None
        self.precision_filter = None  # New precision filter
        self.subrun_ids_filter = None  # New subrun IDs filter
        self.argnv_pairs_filter = None  # New argName-argValue pairs filter
        self.show_details = False
        self.show_verified_sources = False  # New checkbox
        self.show_all_precision = False  # New checkbox
        self.show_custom_benchmarks = False
        
        # Custom plot metrics
        self.x_metric = "Output Throughput (tokens/s)"  # Default x-axis metric
        self.y_metric = "TTFT (ms)"  # Default y-axis metric
        
        # Cache for filter data
        self.filter_data = None
        
        # Load initial data
        self.refresh_data()
    
    def refresh_data(self):
        """Refresh leaderboard data"""
        # Prevent recursive refreshes
        if self.is_refreshing:
            logger.warning("Already refreshing, skipping redundant refresh")
            return False
            
        try:
            self.is_refreshing = True
            current_time = time.time()
            
            # Load all filter data at once
            self.filter_data = get_filtered_data(source=self.source, model_filter=self.model_filter, engine_filter=self.engine_filter, benchmark_filter=self.benchmark_filter, gpu_filter=self.gpu_filter, show_custom_benchmarks=self.show_custom_benchmarks)

            # Register a listener for file changes
            self.file_watcher.register_listener("*", self.on_file_changed)
            self.last_update_time = current_time
            return True
        finally:
            self.is_refreshing = False
    
    def get_formatted_update_time(self):
        """Get formatted last update time"""
        if self.last_update_time == 0:
            return "Never"
        return time.strftime("%H:%M:%S", time.localtime(self.last_update_time))
    
    def on_file_changed(self, file_path):
        """Callback for when a file changes"""
        # Avoid recursive refreshes
        if self.is_refreshing:
            return
            
        self.last_update_time = 0  # Reset timer
        # Force immediate data refresh
        self.refresh_data()
    
    def get_redirect_link(self):
        """Get the appropriate redirect link based on current source"""
        if self.source == "global":
            return GITHUB_REPO_URL
        else:
            return GLOBAL_LEADERBOARD_URL
    
    def get_button_text(self):
        """Get appropriate button text based on current source"""
        if self.source == "global":
            return "View on GitHub"
        else:
            return "View Global Leaderboard"

    def get_filtered_data(self):
        self.filter_data = get_filtered_data(source=self.source, model_filter=self.model_filter, engine_filter=self.engine_filter, benchmark_filter=self.benchmark_filter, gpu_filter=self.gpu_filter, show_custom_benchmarks=self.show_custom_benchmarks)
        return self.filter_data
    
    def get_models(self):
        """Get list of models"""
        if not self.filter_data:
            self.filter_data = self.get_filtered_data()
        return ["All Models"] + self.filter_data["models"]
    
    def get_engines(self):
        """Get list of engines"""
        if not self.filter_data:
            self.filter_data = self.get_filtered_data()
        return ["All Engines"] + self.filter_data["engines"]
    
    def get_benchmark_types(self):
        """Get list of benchmark types"""
        if not self.filter_data:
            self.filter_data = self.get_filtered_data()
        return ["All Benchmarks"] + self.filter_data["benchmarks"]
    
    def get_gpus(self):
        """Get list of GPU types"""
        if not self.filter_data:
            self.filter_data = self.get_filtered_data()
        return ["All GPUs"] + self.filter_data["gpus"]
    
    def get_precision_types(self):
        """Get list of precision types"""
        # Define the standard precision types
        return ["All Precision", "INT4", "INT8", "FP8", "FP16", "BF16", "FP32", "OTHER"]
    
    def get_leaderboard_data(self):
        """Get filtered leaderboard data"""
        if not self.filter_data:
            self.filter_data = self.get_filtered_data()
        return self.filter_data["leaderboard"]
    
    def update_model_filter(self, value):
        """Update model filter"""
        self.model_filter = value
    
    def update_engine_filter(self, value):
        """Update engine filter"""
        self.engine_filter = value
    
    def update_benchmark_filter(self, value):
        """Update benchmark filter"""
        self.benchmark_filter = value
    
    def update_gpu_filter(self, value):
        """Update GPU filter"""
        self.gpu_filter = value
    
    def update_precision_filter(self, value):
        """Update precision filter"""
        self.precision_filter = value
    
    def update_subrun_ids_filter(self, value):
        """Update subrun IDs filter"""
        self.subrun_ids_filter = value
    
    def update_argnv_pairs_filter(self, value):
        """Update argName-argValue pairs filter"""
        self.argnv_pairs_filter = value
    
    def update_show_details(self, value):
        """Update show details flag"""
        self.show_details = value
        
    def update_show_verified_sources(self, value):
        """Update show verified sources flag"""
        self.show_verified_sources = value
        
    def update_show_all_precision(self, value):
        """Update show precision flag"""
        self.show_all_precision = value
        
    def update_show_custom_benchmarks(self, value):
        """Update show custom benchmarks flag"""
        self.show_custom_benchmarks = value
        
    def update_x_metric(self, value):
        """Update x-axis metric for custom plot"""
        self.x_metric = value
        
    def update_y_metric(self, value):
        """Update y-axis metric for custom plot"""
        self.y_metric = value
        
    def get_available_metrics(self):
        """Get list of available metrics for plotting"""
        return [
            "Input Throughput (tokens/s)",
            "Input $/1M tokens",
            "Output Throughput (tokens/s)",
            "Output $/1M tokens",
            "TTFT (ms)",
            "TPOT (ms)",
            "TPS for 1 User (tokens/s/req)"
        ]

def check_value_match(row_value, search_value):
    """Helper function to compare values with special handling for booleans"""
    # Handle boolean values
    if isinstance(row_value, bool):
        if search_value.lower() == "true":
            return row_value is True
        elif search_value.lower() == "false":
            return row_value is False
        else:
            return str(row_value) == search_value
    # Handle other types
    else:
        return str(row_value) == search_value

def process_leaderboard_data(state):
    """Process leaderboard data for display and plotting"""
    entries = state.get_leaderboard_data()
    
    if not entries:
        return None
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(entries)
    
    # Check and potentially calculate per_request_throughput if it's missing
    if 'per_request_throughput' not in df.columns and 'mean_tpot_ms' in df.columns:
        logger.info("Calculating per_request_throughput from TPOT")
        # Convert ms to seconds and calculate requests per second
        df['per_request_throughput'] = df['mean_tpot_ms'].apply(lambda x: 1000.0 / float(x) if float(x) > 0 else 0)
    
    # Apply precision filter if set
    if state.precision_filter is not None and state.precision_filter != "All Precision":
        df = df[df['precision'] == state.precision_filter]
    
    # Apply verified sources filter if enabled and source is global
    if state.show_verified_sources and state.source == "global":
        # Filter to verified sources (GitHub users) based on client_login
        # This assumes verified sources have a non-Anonymous client_login
        df = df[df['client_login'].notna() & (df['client_login'] != "Anonymous")]
    
    # Add visualization-specific fields
    if not df.empty:
        # Create a unique key for each configuration group
        df['config_group'] = df.apply(
            lambda row: (
                f"{row['model']}_{row.get('gpu', '')}_{row.get('precision', '')}_{row['engine']}_" + (
                    f"{row.get('benchmark_config', {}).get('dataset-name', '')}_"
                    f"{row.get('benchmark_config', {}).get('random-input-len', '')}_"
                    f"{row.get('benchmark_config', {}).get('random-output-len', '')}_"
                    f"{row.get('benchmark_config', {}).get('random-prefix-len', '')}"
                    if state.show_custom_benchmarks
                    else f"{row.get('benchmark_config', '')}"
                )
            ),
            axis=1
        )
        
        # Create better labels for the legend
        df['label'] = df.apply(
            lambda row: f"{row['model']} - {row['engine']} ({row.get('precision', 'Unknown')})",
            axis=1
        )

    if not state.show_details:
        state.subrun_ids_filter = None
        state.argnv_pairs_filter = None

    # Apply Subrun IDs filter if set
    if state.subrun_ids_filter and state.show_details:
        subrun_ids = [s.strip() for s in state.subrun_ids_filter.split(',') if s.strip()]
        if subrun_ids:
            # Get full IDs that end with any of the specified subrun IDs
            mask = df['id'].apply(lambda x: any(x.endswith(subrun_id) for subrun_id in subrun_ids))
            df = df[mask]
            
            # If no entries after filtering
            if df.empty:
                return None
    
    # Apply argName-argValue pairs filter if set
    if state.argnv_pairs_filter and state.show_details:
        # Parse argName-argValue pairs
        try:
            argnv_pairs = []
            for pair in state.argnv_pairs_filter.split(','):
                if ':' in pair:
                    key, value = pair.split(':', 1)
                    argnv_pairs.append((key.strip(), value.strip()))
            
            if argnv_pairs:
                # For each key-value pair, keep only the matching rows
                for key, value in argnv_pairs:
                    matching_rows = []
                    
                    # Iterate through each row of the DataFrame
                    for index, row in df.iterrows():
                        match_found = False
                        
                        # Check engine_args
                        engine_args = row.get('engine_args', {})
                        if isinstance(engine_args, dict) and key in engine_args:
                            if check_value_match(engine_args[key], value):
                                match_found = True
                        
                        # Check env_vars
                        if not match_found:
                            env_vars = row.get('env_vars', {})
                            if isinstance(env_vars, dict) and key in env_vars:
                                if check_value_match(env_vars[key], value):
                                    match_found = True
                        
                        # Check benchmark_config
                        if not match_found:
                            benchmark_config = row.get('benchmark_config', {})
                            if isinstance(benchmark_config, dict) and key in benchmark_config:
                                if check_value_match(benchmark_config[key], value):
                                    match_found = True
                        
                        if match_found:
                            matching_rows.append(row)
                    
                    # If matches found, update DataFrame
                    if matching_rows:
                        df = pd.DataFrame(matching_rows)
                    else:
                        return None
        except Exception as e:
            logger.error(f"Error applying key-value filter: {e}")
            return None
    
    return df

def render_leaderboard(state):
    """Render the leaderboard UI"""
    df = process_leaderboard_data(state)
    
    if df is None or df.empty:
        return "<div class='no-data'>No data available. Try adjusting the filters or refreshing.</div>"
    
    # Create HTML for the leaderboard
    html = "<div class='leaderboard-container'>"
    html += "<table>"
    html += "<thead><tr>"
    
    # Add User column if source is global
    if state.source == "global":
        html += "<th>Contributor</th>"
    
    if state.show_details:
        html += "<th>Subrun ID</th>"

    html += "<th>Model</th>"
    
    # Add precision column if show_all_precision is enabled
    if state.show_all_precision:
        html += "<th>Precision</th>"
        
    html += "<th>Engine</th>"
    
    # Show detailed columns if enabled
    if state.show_details:
        html += "<th>Engine Args</th>"
        html += "<th>Env Vars</th>"
    
    html += "<th>GPU</th>"
    html += "<th>Benchmark</th>"
    
    # Show benchmark config if details enabled
    if state.show_details:
        html += "<th>Benchmark Config</th>"
    
    html += "<th>Input Throughput (tokens/s)</th>"
    html += "<th>Input $/1M tokens</th>"
    html += "<th>Output Throughput (tokens/s)</th>"
    html += "<th>Output $/1M tokens</th>"
    html += "<th>TTFT (ms)</th>"
    html += "<th>TPOT (ms)</th>"
    html += "<th>TPS for 1 User (tokens/s/req)</th>"
    html += "<th>Uploaded Time</th>"
    
    # Show reproducible commands if details enabled
    if state.show_details:
        html += "<th>Reproducible Commands</th>"
    
    html += "</tr></thead>"
    html += "<tbody>"
    
    for _, row in df.iterrows():
        html += "<tr>"
        
        # Add avatar column with image if available when source is global
        if state.source == "global":
            avatar_url = row.get('avatar_url', '')
            client_login = row.get('client_login', 'Unknown')
            
            # Show avatar with client login if show_details is enabled
            if state.show_details:
                if avatar_url:
                    html += f"<td><img src='{avatar_url}' width='32' height='32' style='border-radius: 50%;'><br><span style='font-size: 0.8em;'>{client_login}</span></td>"
                else:
                    html += f"<td><div style='width: 32px; height: 32px; background-color: #ccc; border-radius: 50%;'></div><br><span style='font-size: 0.8em;'>{client_login}</span></td>"
            else:
                # Just show avatar without login when details are not enabled
                if avatar_url:
                    html += f"<td><img src='{avatar_url}' width='32' height='32' style='border-radius: 50%;'></td>"
                else:
                    html += "<td><div style='width: 32px; height: 32px; background-color: #ccc; border-radius: 50%;'></div></td>"
                    
        # Add the Subrun ID column value - extract directly from the id field
        if state.show_details:
            full_id = row.get('id', '')
            html += f"<td>{full_id}</td>"

        
        html += f"<td>{row['model']}</td>"
        
        # Add precision column if show_all_precision is enabled
        if state.show_all_precision:
            html += f"<td>{row['precision']}</td>"
            
        html += f"<td>{row['engine']}</td>"
        
        
        # Show detailed columns if enabled
        if state.show_details:
            # Format engine args as nicely formatted JSON
            engine_args = row.get('engine_args', {})
            if isinstance(engine_args, dict) and engine_args:
                engine_args_str = "<pre style='max-height: 100px; overflow-y: auto;'>" + json.dumps(engine_args, indent=2) + "</pre>"
            else:
                engine_args_str = "{}"
            html += f"<td>{engine_args_str}</td>"
            
            # Format env vars as nicely formatted JSON
            env_vars = row.get('env_vars', {})
            if isinstance(env_vars, dict) and env_vars:
                env_vars_str = "<pre style='max-height: 100px; overflow-y: auto;'>" + json.dumps(env_vars, indent=2) + "</pre>"
            else:
                env_vars_str = "{}"
            html += f"<td>{env_vars_str}</td>"
        
        html += f"<td>{row['gpu']}</td>"
        
        # Show benchmark type with (custom) label if it's not a predefined benchmark
        is_predefined = row.get('is_predefined_benchmark', True)
        if is_predefined:
            html += f"<td>{row['benchmark_type']}</td>"
        else:
            html += f"<td><span class='custom-benchmark'>{row['benchmark_type']} (custom)</span></td>"
        
        # Show benchmark config if details enabled
        if state.show_details:
            benchmark_config = row.get('benchmark_config', {})
            if isinstance(benchmark_config, dict) and benchmark_config:
                config_str = "<pre style='max-height: 200px; overflow-y: auto;'>" + json.dumps(benchmark_config, indent=2) + "</pre>"
            else:
                config_str = "{}"
            html += f"<td>{config_str}</td>"
        
        html += f"<td>{row['input_throughput']:.2f}</td>"
        
        # Display input cost per million tokens
        input_cost_per_million = row.get('input_cost_per_million')
        if input_cost_per_million is not None:
            html += f"<td>${input_cost_per_million:.4f}</td>"
        else:
            html += "<td>N/A</td>"
            
        html += f"<td>{row['output_throughput']:.2f}</td>"
        
        # Display output cost per million tokens
        output_cost_per_million = row.get('output_cost_per_million')
        if output_cost_per_million is not None:
            html += f"<td>${output_cost_per_million:.4f}</td>"
        else:
            html += "<td>N/A</td>"
            
        html += f"<td>{row['mean_ttft_ms']:.2f}</td>"
        html += f"<td>{row['mean_tpot_ms']:.2f}</td>"
        
        # Add per request throughput column
        per_request_throughput = row.get('per_request_throughput')
        if per_request_throughput is not None:
            html += f"<td>{per_request_throughput:.2f}</td>"
        else:
            html += "<td>N/A</td>"
            
        # Use different timestamp field based on source
        if state.source == "global":
            html += f"<td>{row.get('upload_datetime')}</td>"
        else:
            html += f"<td>{row['timestamp']}</td>"
        
        # Show reproducible commands if details enabled
        if state.show_details:
            # Generate reproducible commands
            commands = []
            
            # Add environment variables
            if env_vars:
                for key, value in env_vars.items():
                    commands.append(f"export {key}={value}")
            
            model_flags = ["model", "model-path"]
            # Add engine start command
            engine_start_cmd = f"arena start {row['engine']} {row['model']}"
            if engine_args: 
                for key, value in engine_args.items():
                    if key not in model_flags:
                        if isinstance(value, bool):
                            if value:
                                engine_start_cmd += f" --{key}"
                        else:
                            engine_start_cmd += f" --{key} {value}"
                commands.append(engine_start_cmd)
            
            # Add benchmark run command
            run_cmd = f"arena run --engine {row['engine']} --benchmark {row['benchmark_type']}"
            commands.append(run_cmd)
            
            # Format commands with line breaks
            commands_html = "<pre style='max-height: 200px; overflow-y: auto;'>" + "\n".join(commands) + "</pre>"
            html += f"<td>{commands_html}</td>"
            
            html += "</tr>"
    
    html += "</tbody></table></div>"
    
    return html

def create_performance_scatter_plot(state):
    """Create a scatter plot for the leaderboard data that resembles the GTC keynote style"""
    try:
        # Use the shared data processing
        df = process_leaderboard_data(state)
        
        if df is None or df.empty:
            logger.warning("No data available for plotting")
            return None
        
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame size: {len(df)} rows")
        
        # Only keep rows with valid data for plotting
        df = df[(df['output_throughput'] > 0) & (df['per_request_throughput'] > 0)]
        
        if df.empty:
            logger.warning("No valid data points for plotting after filtering")
            return None
            
        # Create the figure with a black background
        fig = go.Figure()
        
        # Get unique configuration groups
        unique_groups = df['config_group'].unique()
        
        # Predefined color palette with distinct colors
        color_palette = [
            'rgb(31, 119, 180)',   # blue
            'rgb(255, 127, 14)',    # orange
            'rgb(44, 160, 44)',     # green
            'rgb(214, 39, 40)',     # red
            'rgb(148, 103, 189)',   # purple
            'rgb(140, 86, 75)',     # brown
            'rgb(227, 119, 194)',   # pink
            'rgb(127, 127, 127)',   # gray
            'rgb(188, 189, 34)',    # olive
            'rgb(23, 190, 207)'     # cyan
        ]
        
        # Assign colors to groups directly
        colors = {}
        for i, group in enumerate(unique_groups):
            colors[group] = color_palette[i % len(color_palette)]
            
        # Determine common attributes for title
        common_attributes = {}
        
        # Check if any filter is applied
        if state.model_filter and state.model_filter != "All Models":
            common_attributes["Model"] = state.model_filter
        if state.engine_filter and state.engine_filter != "All Engines":
            common_attributes["Engine"] = state.engine_filter
        if state.gpu_filter and state.gpu_filter != "All GPUs":
            common_attributes["GPU"] = state.gpu_filter
        if state.benchmark_filter and state.benchmark_filter != "All Benchmarks":
            common_attributes["Benchmark"] = state.benchmark_filter
        if state.precision_filter and state.precision_filter != "All Precision":
            common_attributes["Precision"] = state.precision_filter
            
        # If all data points have the same value for a column, add it as common
        for col in ['model', 'engine', 'benchmark_type', 'gpu', 'precision']:
            if col in df.columns and len(df[col].unique()) == 1:
                attr_name = col.replace('_', ' ').title()
                common_attributes[attr_name] = df[col].iloc[0]
                
        # Create title based on common attributes
        if common_attributes:
            unique_values = set(common_attributes.values())
            title_parts = [f"{value}" for value in unique_values]
            title = "" + ", ".join(title_parts)
        else:
            title = "Inference Performance Comparison"

        # Identify which columns have varying values
        varying_columns = {}
        for col in ['model', 'engine', 'benchmark_type', 'gpu', 'precision']:
            if col in df.columns and len(df[col].unique()) > 1:
                varying_columns[col] = df[col].unique()

        def format_dict_for_tooltip(data, header):
            """Format dictionary data for tooltip display with a header."""
            result = f"<b>{header}:</b><br>"
            
            if isinstance(data, dict):
                for key, value in data.items():
                    result += f"{key}: {value}<br>"
            else:
                result += f"{str(data)}<br>"
            
            result += "<br>"
            return result
        
        # Plot each group
        for group in unique_groups:
            group_df = df[df['config_group'] == group].copy()
            # Sort by x-value for connected lines
            group_df = group_df.sort_values('per_request_throughput')
            
            # Get an example row for the label
            example = group_df.iloc[0]
            
            # Create a custom label that only includes the varying attributes
            label_parts = []
            for col in ['model', 'engine', 'benchmark_type', 'gpu', 'precision']:
                if col in varying_columns:
                    label_parts.append(f"{example[col]}")
            
            # If everything is the same (which would be strange in this context),
            # at least show something distinctive
            if not label_parts:
                label = example['label']
            else:
                label = ",".join(label_parts)
            
            # Add scatter plot with lines and markers
            fig.add_trace(
                go.Scatter(
                    x=group_df['per_request_throughput'],
                    y=group_df['output_throughput'],
                    mode='lines+markers',
                    name=label,
                    line=dict(color=colors[group], width=2),
                    marker=dict(
                        size=8,
                        color=colors[group],
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    text=[
                        f"Subrun ID: {row.get('id', 'N/A')}<br>" +
                        f"{row['model']} - {row['engine']} ({row.get('precision', 'Unknown')})<br>" +
                        f"TPS for 1 User: {row.get('per_request_throughput', 0):.2f}<br>" +
                        f"Throughput: {row.get('output_throughput', 0):.2f}<br><br>" +
                        f"Engine: {row['engine']}<br>" +
                        format_dict_for_tooltip(row.get('engine_args', {}), "Engine Args") +
                        format_dict_for_tooltip(row.get('env_vars', {}), "Env Vars") +
                        f"Benchmark Type: {row['benchmark_type']}<br>" +
                        format_dict_for_tooltip(row.get('benchmark_config', {}), "Benchmark Config") 
                        for _, row in group_df.iterrows()
                    ]
                )
            )
        
        # Update layout to match GTC keynote style
        fig.update_layout(
            # Dark theme with black background
            template="plotly_dark",
            paper_bgcolor='rgb(0, 0, 0)',
            plot_bgcolor='rgb(0, 0, 0)',
            
            # Title styling
            title=dict(
                text=title,
                font=dict(
                    family="Arial, sans-serif",
                    size=24,
                    color="white"
                ),
                x=0.5,  # Center title
                y=0.95  # Position at top
            ),
            
            # Axis styling
            xaxis=dict(
                title=dict(
                    text="TPS per user (tokens/s/req) - higher is better",
                    font=dict(
                        family="Arial, sans-serif",
                        size=14,
                        color="white"
                    )
                ),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.1)',
                showline=True,
                linecolor='rgba(255,255,255,0.5)',
                tickfont=dict(color='white')
            ),
            
            yaxis=dict(
                title=dict(
                    text="Total TPS per system (tokens/s) - higher is better",
                    font=dict(
                        family="Arial, sans-serif",
                        size=14,
                        color="white"
                    )
                ),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.1)',
                showline=True,
                linecolor='rgba(255,255,255,0.5)',
                tickfont=dict(color='white')
            ),
            
            # Legend styling
            legend=dict(
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color="white"
                ),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1,
                orientation="h",  # Horizontal legend
                y=-0.15,  # Position at bottom
                x=0.5,   # Center
                xanchor="center"
            ),
            
            # Margins, size, hover behavior
            margin=dict(l=60, r=40, t=80, b=60),
            autosize=True,
            height=700,
            hovermode='closest',
        )
        
        logger.info("Created enhanced GTC-style Plotly plot successfully")
        return fig
        
    except Exception as e:
        logger.exception(f"Error creating performance scatter plot: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_custom_scatter_plot(state, x_metric, y_metric):
    """Create a customizable scatter plot where users can select x and y metrics"""
    try:
        # Use the shared data processing
        df = process_leaderboard_data(state)
        
        if df is None or df.empty:
            logger.warning("No data available for plotting")
            return None
        
        logger.info(f"DataFrame columns: {df.columns.tolist()}")
        logger.info(f"DataFrame size: {len(df)} rows")
        
        # Map friendly names to actual column names
        metric_mapping = {
            "Input Throughput (tokens/s)": "input_throughput",
            "Input $/1M tokens": "input_cost_per_million",
            "Output Throughput (tokens/s)": "output_throughput",
            "Output $/1M tokens": "output_cost_per_million",
            "TTFT (ms)": "mean_ttft_ms",
            "TPOT (ms)": "mean_tpot_ms",
            "TPS for 1 User (tokens/s/req)": "per_request_throughput"
        }
        
        # Define which metrics are "higher is better" vs "lower is better"
        higher_is_better = {
            "Input Throughput (tokens/s)": True,
            "Input $/1M tokens": False,
            "Output Throughput (tokens/s)": True,
            "Output $/1M tokens": False,
            "TTFT (ms)": False,
            "TPOT (ms)": False,
            "TPS for 1 User (tokens/s/req)": True
        }
        
        x_column = metric_mapping.get(x_metric)
        y_column = metric_mapping.get(y_metric)
        
        if not x_column or not y_column:
            logger.warning(f"Invalid metrics selected: x={x_metric}, y={y_metric}")
            return None
        
        # Only keep rows with valid data for the selected metrics
        df = df[(df[x_column].notna()) & (df[y_column].notna()) & (df[x_column] > 0) & (df[y_column] > 0)]
        
        if df.empty:
            logger.warning("No valid data points for plotting after filtering")
            return None
            
        # Create the figure with a black background
        fig = go.Figure()
        
        # Get unique configuration groups
        unique_groups = df['config_group'].unique()
        
        # Predefined color palette with distinct colors
        color_palette = [
            'rgb(31, 119, 180)',   # blue
            'rgb(255, 127, 14)',    # orange
            'rgb(44, 160, 44)',     # green
            'rgb(214, 39, 40)',     # red
            'rgb(148, 103, 189)',   # purple
            'rgb(140, 86, 75)',     # brown
            'rgb(227, 119, 194)',   # pink
            'rgb(127, 127, 127)',   # gray
            'rgb(188, 189, 34)',    # olive
            'rgb(23, 190, 207)'     # cyan
        ]
        
        # Assign colors to groups directly
        colors = {}
        for i, group in enumerate(unique_groups):
            colors[group] = color_palette[i % len(color_palette)]
            
        # Determine common attributes for title
        common_attributes = {}
        
        # Check if any filter is applied
        if state.model_filter and state.model_filter != "All Models":
            common_attributes["Model"] = state.model_filter
        if state.engine_filter and state.engine_filter != "All Engines":
            common_attributes["Engine"] = state.engine_filter
        if state.gpu_filter and state.gpu_filter != "All GPUs":
            common_attributes["GPU"] = state.gpu_filter
        if state.benchmark_filter and state.benchmark_filter != "All Benchmarks":
            common_attributes["Benchmark"] = state.benchmark_filter
        if state.precision_filter and state.precision_filter != "All Precision":
            common_attributes["Precision"] = state.precision_filter
            
        # If all data points have the same value for a column, add it as common
        for col in ['model', 'engine', 'benchmark_type', 'gpu', 'precision']:
            if col in df.columns and len(df[col].unique()) == 1:
                attr_name = col.replace('_', ' ').title()
                common_attributes[attr_name] = df[col].iloc[0]
                
        # Create title based on common attributes
        if common_attributes:
            unique_values = set(common_attributes.values())
            title_parts = [f"{value}" for value in unique_values]
            title = "" + ", ".join(title_parts)
        else:
            title = "Custom Inference Performance Comparison"

        # Identify which columns have varying values
        varying_columns = {}
        for col in ['model', 'engine', 'benchmark_type', 'gpu', 'precision']:
            if col in df.columns and len(df[col].unique()) > 1:
                varying_columns[col] = df[col].unique()

        def format_dict_for_tooltip(data, header):
            """Format dictionary data for tooltip display with a header."""
            result = f"<b>{header}:</b><br>"
            
            if isinstance(data, dict):
                for key, value in data.items():
                    result += f"{key}: {value}<br>"
            else:
                result += f"{str(data)}<br>"
            
            result += "<br>"
            return result
        
        # Plot each group
        for group in unique_groups:
            group_df = df[df['config_group'] == group].copy()
            # Sort by x-value for connected lines
            group_df = group_df.sort_values(x_column)
            
            # Get an example row for the label
            example = group_df.iloc[0]
            
            # Create a custom label that only includes the varying attributes
            label_parts = []
            for col in ['model', 'engine', 'benchmark_type', 'gpu', 'precision']:
                if col in varying_columns:
                    label_parts.append(f"{example[col]}")
            
            # If everything is the same (which would be strange in this context),
            # at least show something distinctive
            if not label_parts:
                label = example['label']
            else:
                label = ",".join(label_parts)
            
            # Add scatter plot with lines and markers
            fig.add_trace(
                go.Scatter(
                    x=group_df[x_column],
                    y=group_df[y_column],
                    mode='lines+markers',
                    name=label,
                    line=dict(color=colors[group], width=2),
                    marker=dict(
                        size=8,
                        color=colors[group],
                        line=dict(color='white', width=1)
                    ),
                    hovertemplate='<b>%{text}</b><extra></extra>',
                    text=[
                        f"Subrun ID: {row.get('id', 'N/A')}<br>" +
                        f"{row['model']} - {row['engine']} ({row.get('precision', 'Unknown')})<br>" +
                        f"{x_metric}: {row.get(x_column, 0):.2f}<br>" +
                        f"{y_metric}: {row.get(y_column, 0):.2f}<br><br>" +
                        f"Engine: {row['engine']}<br>" +
                        format_dict_for_tooltip(row.get('engine_args', {}), "Engine Args") +
                        format_dict_for_tooltip(row.get('env_vars', {}), "Env Vars") +
                        f"Benchmark Type: {row['benchmark_type']}<br>" +
                        format_dict_for_tooltip(row.get('benchmark_config', {}), "Benchmark Config") 
                        for _, row in group_df.iterrows()
                    ]
                )
            )
        
        # Create axis labels with "higher/lower is better" annotations
        x_label = f"{x_metric} - {'higher is better' if higher_is_better.get(x_metric, True) else 'lower is better'}"
        y_label = f"{y_metric} - {'higher is better' if higher_is_better.get(y_metric, True) else 'lower is better'}"
        
        # Update layout to match GTC keynote style
        fig.update_layout(
            # Dark theme with black background
            template="plotly_dark",
            paper_bgcolor='rgb(0, 0, 0)',
            plot_bgcolor='rgb(0, 0, 0)',
            
            # Title styling
            title=dict(
                text=title,
                font=dict(
                    family="Arial, sans-serif",
                    size=24,
                    color="white"
                ),
                x=0.5,  # Center title
                y=0.95  # Position at top
            ),
            
            # Axis styling
            xaxis=dict(
                title=dict(
                    text=x_label,
                    font=dict(
                        family="Arial, sans-serif",
                        size=14,
                        color="white"
                    )
                ),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.1)',
                showline=True,
                linecolor='rgba(255,255,255,0.5)',
                tickfont=dict(color='white')
            ),
            
            yaxis=dict(
                title=dict(
                    text=y_label,
                    font=dict(
                        family="Arial, sans-serif",
                        size=14,
                        color="white"
                    )
                ),
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=True,
                zerolinecolor='rgba(255,255,255,0.1)',
                showline=True,
                linecolor='rgba(255,255,255,0.5)',
                tickfont=dict(color='white')
            ),
            
            # Legend styling
            legend=dict(
                font=dict(
                    family="Arial, sans-serif",
                    size=12,
                    color="white"
                ),
                bgcolor='rgba(0,0,0,0.5)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1,
                orientation="h",  # Horizontal legend
                y=-0.15,  # Position at bottom
                x=0.5,   # Center
                xanchor="center"
            ),
            
            # Margins, size, hover behavior
            margin=dict(l=60, r=40, t=80, b=60),
            autosize=True,
            height=700,
            hovermode='closest',
        )
        
        logger.info("Created custom scatter plot successfully")
        return fig
        
    except Exception as e:
        logger.exception(f"Error creating custom scatter plot: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def create_top5_bar_chart(state, primary_metric, secondary_metric=None):
    """Create a vertical bar chart showing top 5 performers for the selected metric with optional secondary metric"""
    try:
        # Use shared data processing
        df = process_leaderboard_data(state)
        
        if df is None or df.empty:
            logger.warning("No data available for plotting")
            return None
        
        # Map friendly names to actual column names and sort direction
        metric_mapping = {
            "Input Throughput (tokens/s)": ("input_throughput", True),
            "Input $/1M tokens": ("input_cost_per_million", False),
            "Output Throughput (tokens/s)": ("output_throughput", True),
            "Output $/1M tokens": ("output_cost_per_million", False),
            "TTFT (ms)": ("mean_ttft_ms", False),
            "TPOT (ms)": ("mean_tpot_ms", False),
            "TPS for 1 User (tokens/s/req)": ("per_request_throughput", True)
        }
        
        primary_column, primary_higher_is_better = metric_mapping.get(primary_metric, ("input_throughput", True))
        
        # Ensure the specified column exists and has valid values
        if primary_column not in df.columns:
            logger.warning(f"Column {primary_column} does not exist in the data")
            return None
        
        # Only keep rows with valid data for this metric
        df = df[df[primary_column].notna() & (df[primary_column] > 0)]
        
        if df.empty:
            logger.warning("No valid data points after filtering")
            return None
            
        # Sort by the primary metric and select top 5
        if primary_higher_is_better:
            df_sorted = df.sort_values(by=primary_column, ascending=False).head(5)
        else:
            df_sorted = df.sort_values(by=primary_column, ascending=True).head(5)
        
        # Create simplified labels
        df_sorted['short_label'] = df_sorted.apply(
            lambda row: f"{row['model']}", 
            axis=1
        )
        
        # Create compact Engine Args string
        df_sorted['engine_args_str'] = df_sorted.apply(
            lambda row: get_short_args_string(row.get('engine_args', {})),
            axis=1
        )
        
        # Create compact Env Vars string
        df_sorted['env_vars_str'] = df_sorted.apply(
            lambda row: get_short_args_string(row.get('env_vars', {})),
            axis=1
        )
        
        # Create hover text with enhanced information including benchmark config
        df_sorted['hover_text'] = df_sorted.apply(
            lambda row: create_enhanced_hover_text(row, primary_metric, secondary_metric),
            axis=1
        )
        
        # Create a better title based on whether higher or lower is better
        if primary_higher_is_better:
            plot_title = f"Highest Performing Sub-runs by {primary_metric}"
        else:
            plot_title = f"Highest Performing Sub-runs by {primary_metric} (Lower is Better)"
        
        # Now directly create a Plotly figure instead of using matplotlib
        fig = go.Figure()
        
        # Set up the layout first with light theme
        fig.update_layout(
            title=dict(
                text=plot_title,
                font=dict(size=20, color="#333333", weight="bold"),
                x=0.5  
            ),
            paper_bgcolor='#F8F9FA',
            plot_bgcolor='#F8F9FA',
            margin=dict(l=0, r=0, t=0, b=0),  
            height=680, 
            hovermode="closest",
            hoverlabel=dict(
                bgcolor="white",
                font_size=12,
                font_family="Arial"
            )
        )
        
        # Create multi-line x-axis labels (Rank, model, and subrun_id on separate lines)
        model_labels = []
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            model_name = row['model']
            engine_name = row['engine']
            subrun_id = row.get('id', 'N/A')
            # Format subrun_id to show only the last 8 characters if it's too long
            if len(subrun_id) > 8:
                subrun_id = subrun_id[-8:]
                
            model_labels.append(f"Rank: {i+1}<br>{model_name}<br>Engine: {engine_name}<br>Subrun Short ID: {subrun_id}")

        bar_width = 0.35 if secondary_metric else 0.7

        if secondary_metric and secondary_metric in metric_mapping:
            secondary_column, secondary_higher_is_better = metric_mapping.get(secondary_metric)
            if secondary_column in df_sorted.columns and df_sorted[secondary_column].notna().all() and (df_sorted[secondary_column] > 0).all():

                centered_tickvals = [i + bar_width/2 for i in range(len(df_sorted))]
                x_range = [-0.45, len(df_sorted) - 0.2]
                fig.update_xaxes(
                    title=None,
                    ticktext=model_labels,
                    tickvals=centered_tickvals,  
                    tickangle=0,
                    showgrid=False,
                    range=x_range
                )
            else:
                # If only one bar series is shown (just the primary metric)
                fig.update_xaxes(
                    title=None,
                    ticktext=model_labels,
                    tickvals=list(range(len(df_sorted))),
                    tickangle=0,
                    showgrid=False,
                    range=x_range
                )
        else:
            # If only one bar series is shown (just the primary metric)
            fig.update_xaxes(
                title=None,
                ticktext=model_labels,
                tickvals=list(range(len(df_sorted))),
                tickangle=0,
                showgrid=False,
                range=[-0.45, len(df_sorted) - 0.2]
            )
                
        # Add label indicating if higher or lower is better for primary metric
        primary_label = f"{primary_metric} - {'higher is better' if primary_higher_is_better else 'lower is better'}"
        
        fig.update_yaxes(
            title=dict(
                text=primary_label,
                font=dict(
                    size=12,
                    color="#1F77B4",
                    family="Arial",
                    weight="bold"
                )
            ),
            showgrid=True,
            gridcolor='rgba(200,200,200,0.2)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0,0,0,0.2)'
        )

        bar_colors = f'rgb(31, 119, 180)'
        offset = -bar_width/2 if secondary_metric else 0
        
        # Add the primary metric bars
        fig.add_trace(go.Bar(
            x=list(range(len(df_sorted))),
            y=df_sorted[primary_column],
            text=df_sorted[primary_column].round(1),
            textposition='outside',
            hoverinfo='text',
            hovertext=df_sorted['hover_text'],
            marker_color=bar_colors,
            marker_line_color='white',
            marker_line_width=1.5,
            width=bar_width,
            name=primary_metric,
            offset=offset
        ))
        
        # If we have a secondary metric, add it as a second bar
        if secondary_metric and secondary_metric in metric_mapping:
            secondary_column, secondary_higher_is_better = metric_mapping.get(secondary_metric)
            
            if secondary_column in df_sorted.columns and df_sorted[secondary_column].notna().all() and (df_sorted[secondary_column] > 0).all():

                secondary_max = df_sorted[secondary_column].max() * 1.1
                primary_max = df_sorted[primary_column].max() * 1.1
                fig.update_yaxes(range=[0, primary_max])
                
                # Add label indicating if higher or lower is better for secondary metric
                secondary_label = f"{secondary_metric} - {'higher is better' if secondary_higher_is_better else 'lower is better'}"
                
                # Add a second y-axis with proper range to ensure alignment
                fig.update_layout(
                    title=dict(
                        text=plot_title,
                        font=dict(size=20, color="#333333", weight="bold"),
                        x=0.5
                    ),
                    yaxis2=dict(
                        title=dict(
                            text=secondary_label,
                            font=dict(
                                size=12,
                                color="#FF7F0E",
                                family="Arial",
                                weight="bold"
                            )
                        ),
                        tickfont=dict(color="#FF7F0E"),
                        overlaying="y",
                        side="right",
                        range=[0, secondary_max],  
                        showgrid=False,
                        zeroline=True,
                        zerolinewidth=1,
                        zerolinecolor='rgba(0,0,0,0.2)'
                    )
                )
                
                # Add secondary metric as bar
                fig.add_trace(go.Bar(
                    x=list(range(len(df_sorted))),
                    y=df_sorted[secondary_column],
                    text=df_sorted[secondary_column].round(1),
                    textposition='outside',
                    hoverinfo='text',
                    hovertext=df_sorted['hover_text'],
                    marker_color='#FF7F0E',
                    marker_line_color='white',
                    marker_line_width=1.5,
                    width=bar_width,
                    name=secondary_metric,
                    offset=bar_width/2,
                    yaxis="y2",
                    base=0  
                ))
        
        # Add filter information to the plot
        filter_text = []
        if state.model_filter and state.model_filter != "All Models":
            filter_text.append(f"Model: {state.model_filter}")
        if state.engine_filter and state.engine_filter != "All Engines":
            filter_text.append(f"Engine: {state.engine_filter}")
        if state.gpu_filter and state.gpu_filter != "All GPUs":
            filter_text.append(f"GPU: {state.gpu_filter}")
        if state.benchmark_filter and state.benchmark_filter != "All Benchmarks":
            filter_text.append(f"Benchmark: {state.benchmark_filter}")
        
        if filter_text:
            subtitle = " | ".join(filter_text)
            fig.add_annotation(
                text=subtitle,
                x=0.5,
                y=1.05,
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=10, color="#666666", style="italic"),
                bgcolor="#F4F4F4",
                borderpad=4,
                bordercolor="#DDDDDD",
                borderwidth=1
            )
        
        # Set legend at the bottom
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.15,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.9)",
                bordercolor="#DDDDDD",
                borderwidth=1
            ),
            bargap=0.15,
            bargroupgap=0.1
        )
        
        logger.info("Successfully created enhanced Top 5 ranking chart with improved hover and alignment")
        return fig
        
    except Exception as e:
        logger.exception(f"Error creating Top 5 chart: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Helper function: Create enhanced hover text with benchmark config details
def create_enhanced_hover_text(row, primary_metric, secondary_metric=None):

    hover_text = f"<b>Subrun ID:</b> {row.get('id', 'N/A')}<br><br>"

    hover_text += f"<b>Model:</b> {row.get('model', 'Unknown')}<br>"
    hover_text += f"<b>Engine:</b> {row.get('engine', 'Unknown')}<br>"
    hover_text += f"<b>Precision:</b> {row.get('precision', 'Unknown')}<br>"
    
    # Add hardware section 
    hover_text += "<br><b>Hardware:</b><br>"
    hover_text += f"{row.get('gpu', 'Unknown')}<br>"

    # If there are engine arguments, add these
    engine_args = row.get('engine_args', {})
    if engine_args and isinstance(engine_args, dict):
        hover_text += "<br><b>Engine Parameters:</b><br>"
        for key, value in engine_args.items():
            hover_text += f"{key}: {value}<br>"

    # If there are env arguments, add these
    env_vars = row.get('env_vars', {})
    if env_vars and isinstance(env_vars, dict):
        hover_text += "<br><b>Env Parameters:</b><br>"
        for key, value in env_vars.items():
            hover_text += f"{key}: {value}<br>"
    
    # Add benchmark config section with detailed information 
    benchmark_config = row.get('benchmark_config', {})
    hover_text += "<br><b>Benchmark Configuration:</b><br>"
    hover_text += f"Benchmark Type: {row.get('benchmark_type', 'Unknown')}<br>"
    
    if benchmark_config and isinstance(benchmark_config, dict):
        # Show up to 10 important parameters from benchmark config
        items_shown = 0
        for key, value in benchmark_config.items():
            hover_text += f"{key}: {value}<br>"
            items_shown += 1
    return hover_text

def create_interface():
    """Create the Gradio interface"""
    state = LeaderboardState()
    
    with gr.Blocks(css=CSS) as interface:
        title_html = gr.HTML(f"<h1 class='title'>Inference Engine Arena {state.source.capitalize()} Leaderboard</h1>")
        gr.HTML(f"<p class='subtitle'>Compare performance of different inference engines and models - v{PROJECT_VERSION}</p>")
        
        with gr.Row() as header:
            refresh_btn = gr.Button("Refresh Data", variant="primary")
            source_btn = gr.Button(state.get_button_text(), variant="secondary", link=state.get_redirect_link())
        
        last_update_html = gr.HTML(f"<div class='timer'>Last update: <span id='last-update'>{state.get_formatted_update_time()}</span></div>")
        
        # Regular filters section
        with gr.Row(elem_classes="filter-section"):
            with gr.Row():
                model_dropdown = gr.Dropdown(choices=state.get_models(), label="Model", value="All Models")
                benchmark_dropdown = gr.Dropdown(choices=state.get_benchmark_types(), label="Benchmark", value="All Benchmarks")
                engine_dropdown = gr.Dropdown(choices=state.get_engines(), label="Engine", value="All Engines")
                gpu_dropdown = gr.Dropdown(choices=state.get_gpus(), label="GPU", value="All GPUs")
                # Initialize precision dropdown but hide it initially
                precision_dropdown = gr.Dropdown(choices=state.get_precision_types(), label="Precision", value="All Precision", visible=state.show_all_precision)
            
            with gr.Row():
                show_details_checkbox = gr.Checkbox(label="Show details", value=state.show_details, info="When checked, shows additional columns with engine arguments, environment variables, and benchmark configuration")
                show_custom_benchmarks_checkbox = gr.Checkbox(label="Show custom benchmarks & filter", value=state.show_custom_benchmarks, info="When checked, shows custom benchmarks and allows filtering")
                show_verified_sources_checkbox = gr.Checkbox(label="Show only results from verified sources", value=state.show_verified_sources, info="When checked, only shows results from verified sources(global only)", visible=state.source=="global")
                show_all_precision_checkbox = gr.Checkbox(label="Show all precision", value=state.show_all_precision, info="When checked, shows all data types and quantization methods")

        # Advanced filters container as a separate section below the regular filters
        # Only visible when "Show details" is checked
        with gr.Column(visible=state.show_details, elem_classes="advanced-filter-section") as advanced_filters_container:
            gr.HTML("<div class='advanced-filter-title'>Advanced Filters</div>")
            
            with gr.Row(elem_classes="advanced-filters-container"):
                # argName-argValue pairs filter
                with gr.Column(elem_classes="advanced-filter-box"):
                    argnv_pairs_textbox = gr.Textbox(
                        label="Filter by Engine Args, Env Vars, Benchmark Config", 
                        placeholder="e.g. max-num-seqs:256,request-rate:20"
                    )
                
                # Subrun IDs filter
                with gr.Column(elem_classes="advanced-filter-box"):
                    subrun_ids_textbox = gr.Textbox(
                        label="Filter by Subrun IDs", 
                        placeholder="e.g. dc1a8d7f,1b466233"
                    )
            
            gr.HTML("<div class='filter-hint'>Filter by Subrun IDs will show all selected subruns,while Filter by Engine Args, Env Vars, Benchmark Config will filter subruns that meet all input conditions.</div>")

            # Search button aligned with other interface elements
            with gr.Row(elem_classes="search-button-container"):
                search_button = gr.Button("Apply Filters", variant="primary")
        
        # Create plot - directly use Gradio's plot component with the figure
        with gr.Column(visible=True):
            gr.HTML("<h2 style='text-align: center; margin-top: 20px;'>Inference Performance Comparison (TPS per user vs Total TPS per system)</h2>")
            gr.HTML("<h3 style='text-align: center; margin-top: 10px;'>Inspired by 'Inference at Scale Is Extreme Computing' section of <a href='https://www.youtube.com/live/_waPvOwL9Z8?t=3305s' target='_blank'>NVIDIA GTC 2025 Keynote</a></h3>")
            plot = gr.Plot(value=create_performance_scatter_plot(state))
            plot_message = gr.HTML("<div style='text-align: center; font-size: 0.9rem; margin-top: 10px; color: #888;'>X-axis: TPS per user (tokens/s/req) - " + METRIC_DEFINITIONS["TPS for 1 User (tokens/s/req)"] + "<br><br>Y-axis: Total TPS per system (tokens/s) - " + METRIC_DEFINITIONS["Output Throughput (tokens/s)"] + "<br><br>For more information, watch NVIDIA GTC 2025 Keynote Inference at Scale is Extreme Computing section: <a href='https://www.youtube.com/live/_waPvOwL9Z8?t=3305s' target='_blank'>https://www.youtube.com/live/_waPvOwL9Z8?t=3305s</a></div>")
            
        # Custom plot with selectable metrics
        with gr.Column(visible=True):
            gr.HTML("<h2 style='text-align: center; margin-top: 20px;'>Custom Inference Performance Comparison</h2>")
            
            with gr.Row():
                x_metric_dropdown = gr.Dropdown(
                    choices=state.get_available_metrics(),
                    label="X-Axis Metric",
                    value=state.x_metric
                )
                y_metric_dropdown = gr.Dropdown(
                    choices=state.get_available_metrics(),
                    label="Y-Axis Metric",
                    value=state.y_metric
                )
            
            custom_plot = gr.Plot(value=create_custom_scatter_plot(state, state.x_metric, state.y_metric))
            
            # Get metric explanations for initial display
            x_explanation = METRIC_DEFINITIONS.get(state.x_metric, "")
            y_explanation = METRIC_DEFINITIONS.get(state.y_metric, "")
            
            # Create a simple message with axis explanations
            custom_plot_message = gr.HTML(f"<div style='text-align: left; font-size: 0.9rem; margin-top: 10px; color: #666;'>X-axis: {state.x_metric} - {x_explanation}<br><br>Y-axis: {state.y_metric} - {y_explanation}<br><br> Reference: <a href='https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html' target='_blank'>NVIDIA LLM Benchmarking Metrics Documentation</a></div>")
        
        # Top 5 plot without title
        with gr.Column(visible=True):
            top5_plot = gr.Plot(value=create_top5_bar_chart(state, state.x_metric, state.y_metric), elem_id="top5-plot")
            
            
        # Leaderboard display
        leaderboard_html = gr.HTML(render_leaderboard(state))
        
        def wrapped_refresh_data():
            state.refresh_data()
            
            # Try to create plots
            plot_fig = create_performance_scatter_plot(state)
            custom_plot_fig = create_custom_scatter_plot(state, state.x_metric, state.y_metric)
            top5_plot_fig = create_top5_bar_chart(state, state.x_metric, state.y_metric)
            
            # Update all dropdowns with refreshed data
            return {
                model_dropdown: gr.Dropdown(choices=state.get_models(), value=state.model_filter or "All Models"),
                engine_dropdown: gr.Dropdown(choices=state.get_engines(), value=state.engine_filter or "All Engines"),
                benchmark_dropdown: gr.Dropdown(choices=state.get_benchmark_types(), value=state.benchmark_filter or "All Benchmarks"),
                gpu_dropdown: gr.Dropdown(choices=state.get_gpus(), value=state.gpu_filter or "All GPUs"),
                leaderboard_html: render_leaderboard(state),
                plot: plot_fig,
                custom_plot: custom_plot_fig,
                top5_plot: top5_plot_fig,
                last_update_html: f"<div class='timer'>Last update: <span id='last-update'>{state.get_formatted_update_time()}</span></div>"
            }
        
        def toggle_precision_dropdown(show_all_precision):
            """Toggle visibility of precision dropdown based on show_all_precision checkbox."""
            state.update_show_all_precision(show_all_precision)
            
            # Try to create plots
            plot_fig = create_performance_scatter_plot(state)
            custom_plot_fig = create_custom_scatter_plot(state, state.x_metric, state.y_metric)
            top5_plot_fig = create_top5_bar_chart(state, state.x_metric, state.y_metric)
            
            return {
                precision_dropdown: gr.Dropdown(visible=show_all_precision),
                leaderboard_html: render_leaderboard(state),
                plot: plot_fig,
                custom_plot: custom_plot_fig,
                top5_plot: top5_plot_fig,
                last_update_html: f"<div class='timer'>Last update: <span id='last-update'>{state.get_formatted_update_time()}</span></div>"
            }
        
        def toggle_advanced_filters(show_details):
            """Toggle visibility  of advanced filters based on show_details checkbox"""
            state.show_details = show_details
            state.update_show_details(show_details)
            if not state.show_details:
                state.update_subrun_ids_filter(None)
                state.update_argnv_pairs_filter(None)
            # Try to create plots
            plot_fig = create_performance_scatter_plot(state)
            custom_plot_fig = create_custom_scatter_plot(state, state.x_metric, state.y_metric)
            top5_plot_fig = create_top5_bar_chart(state, state.x_metric, state.y_metric)
            
            return {
                advanced_filters_container: gr.Column(visible=show_details),
                subrun_ids_textbox: "" if not show_details else subrun_ids_textbox.value,
                argnv_pairs_textbox: "" if not show_details else argnv_pairs_textbox.value,
                leaderboard_html: render_leaderboard(state),
                plot: plot_fig,
                custom_plot: custom_plot_fig,
                top5_plot: top5_plot_fig,
                last_update_html: f"<div class='timer'>Last update: <span id='last-update'>{state.get_formatted_update_time()}</span></div>"
            }
        
        def wrapped_update_filters(model, engine, benchmark, gpu, precision, subrun_ids, argnv_pairs, show_details, show_verified_sources, show_all_precision, show_custom_benchmarks):
            """Update all filters and refresh the leaderboard"""
            needs_refresh = True
            if state.show_details != show_details or state.show_all_precision != show_all_precision:
                needs_refresh = False

            # Update state with selected filters
            state.update_model_filter(model)
            state.update_engine_filter(engine)
            state.update_benchmark_filter(benchmark)
            state.update_gpu_filter(gpu)
            state.update_precision_filter(precision)
            state.update_subrun_ids_filter(subrun_ids)
            state.update_argnv_pairs_filter(argnv_pairs)
            state.update_show_details(show_details)
            state.update_show_verified_sources(show_verified_sources)
            state.update_show_all_precision(show_all_precision)
            state.update_show_custom_benchmarks(show_custom_benchmarks)
            state.show_details = show_details
            # Force refresh data to ensure we have the latest
            if needs_refresh:
                state.refresh_data()
            
            # Try to create plots
            plot_fig = create_performance_scatter_plot(state)
            custom_plot_fig = create_custom_scatter_plot(state, state.x_metric, state.y_metric)

            top5_plot_fig = create_top5_bar_chart(state, state.x_metric, state.y_metric)
            
            # Return updated leaderboard and timestamp
            return {
                leaderboard_html: render_leaderboard(state),
                plot: plot_fig,
                custom_plot: custom_plot_fig,
                top5_plot: top5_plot_fig,
                last_update_html: f"<div class='timer'>Last update: <span id='last-update'>{state.get_formatted_update_time()}</span></div>"
            }
        
        def update_custom_plot(x_metric, y_metric, model, engine, benchmark, gpu, precision, subrun_ids, argnv_pairs, show_details, show_verified_sources, show_all_precision, show_custom_benchmarks):
            """Update the custom plot with new metrics"""
            state.update_x_metric(x_metric)
            state.update_y_metric(y_metric)
            
            # Update existing filters to maintain context
            state.update_model_filter(model)
            state.update_engine_filter(engine)
            state.update_benchmark_filter(benchmark)
            state.update_gpu_filter(gpu)
            state.update_precision_filter(precision)
            state.update_subrun_ids_filter(subrun_ids)
            state.update_argnv_pairs_filter(argnv_pairs)
            state.update_show_details(show_details)
            state.update_show_verified_sources(show_verified_sources)
            state.update_show_all_precision(show_all_precision)
            state.update_show_custom_benchmarks(show_custom_benchmarks)
            
            # Create new custom plot with updated metrics
            custom_plot_fig = create_custom_scatter_plot(state, x_metric, y_metric)
            
            # Update Top 5 chart with the same X-axis metric as primary and Y-axis metric as secondary
            top5_plot_fig = create_top5_bar_chart(state, x_metric, y_metric)
            
            # Get metric explanations from the definitions dictionary
            x_explanation = METRIC_DEFINITIONS.get(x_metric, "")
            y_explanation = METRIC_DEFINITIONS.get(y_metric, "")
            
            return {
                custom_plot: custom_plot_fig,
                custom_plot_message: f"<div style='text-align: left; font-size: 0.9rem; margin-top: 10px; color: #666;'>X-axis: {x_metric} - {x_explanation}<br><br>Y-axis: {y_metric} - {y_explanation}<br><br> Reference: <a href='https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html' target='_blank'>NVIDIA LLM Benchmarking Metrics Documentation</a></div>",
                top5_plot: top5_plot_fig
            }
        
        def apply_advanced_filters(model, engine, benchmark, gpu, precision, subrun_ids, argnv_pairs, show_details, show_verified_sources, show_all_precision, show_custom_benchmarks):
            """Handle search button click for both filters"""
            return wrapped_update_filters(model, engine, benchmark, gpu, precision, subrun_ids, argnv_pairs, show_details, show_verified_sources, show_all_precision, show_custom_benchmarks)
        
        # Connect UI elements to event handlers
        # Don't need to connect the source_btn anymore since it has a direct link
        refresh_btn.click(wrapped_refresh_data, inputs=[], outputs=[model_dropdown, engine_dropdown, benchmark_dropdown, gpu_dropdown, leaderboard_html, plot, custom_plot, top5_plot, last_update_html])
        
        # Special handler for show details checkbox to toggle advanced filters visibility
        show_details_checkbox.change(
            toggle_advanced_filters,
            inputs=[show_details_checkbox],
            outputs=[advanced_filters_container, subrun_ids_textbox, argnv_pairs_textbox, leaderboard_html, plot, custom_plot, top5_plot, last_update_html]
        )
        
        # Special handler for precision checkbox to toggle precision dropdown visibility
        show_all_precision_checkbox.change(
            toggle_precision_dropdown, 
            inputs=[show_all_precision_checkbox], 
            outputs=[precision_dropdown, leaderboard_html, plot, custom_plot, top5_plot, last_update_html]
        )
        
        # Connect standard filters to update handler
        model_dropdown.change(wrapped_update_filters, inputs=[model_dropdown, engine_dropdown, benchmark_dropdown, gpu_dropdown, precision_dropdown, subrun_ids_textbox, argnv_pairs_textbox, show_details_checkbox, show_verified_sources_checkbox, show_all_precision_checkbox, show_custom_benchmarks_checkbox], outputs=[leaderboard_html, plot, custom_plot, top5_plot, last_update_html])
        engine_dropdown.change(wrapped_update_filters, inputs=[model_dropdown, engine_dropdown, benchmark_dropdown, gpu_dropdown, precision_dropdown, subrun_ids_textbox, argnv_pairs_textbox, show_details_checkbox, show_verified_sources_checkbox, show_all_precision_checkbox, show_custom_benchmarks_checkbox], outputs=[leaderboard_html, plot, custom_plot, top5_plot, last_update_html])
        benchmark_dropdown.change(wrapped_update_filters, inputs=[model_dropdown, engine_dropdown, benchmark_dropdown, gpu_dropdown, precision_dropdown, subrun_ids_textbox, argnv_pairs_textbox, show_details_checkbox, show_verified_sources_checkbox, show_all_precision_checkbox, show_custom_benchmarks_checkbox], outputs=[leaderboard_html, plot, custom_plot, top5_plot, last_update_html])
        gpu_dropdown.change(wrapped_update_filters, inputs=[model_dropdown, engine_dropdown, benchmark_dropdown, gpu_dropdown, precision_dropdown, subrun_ids_textbox, argnv_pairs_textbox, show_details_checkbox, show_verified_sources_checkbox, show_all_precision_checkbox, show_custom_benchmarks_checkbox], outputs=[leaderboard_html, plot, custom_plot, top5_plot, last_update_html])
        precision_dropdown.change(wrapped_update_filters, inputs=[model_dropdown, engine_dropdown, benchmark_dropdown, gpu_dropdown, precision_dropdown, subrun_ids_textbox, argnv_pairs_textbox, show_details_checkbox, show_verified_sources_checkbox, show_all_precision_checkbox, show_custom_benchmarks_checkbox], outputs=[leaderboard_html, plot, custom_plot, top5_plot, last_update_html])
        show_verified_sources_checkbox.change(wrapped_update_filters, inputs=[model_dropdown, engine_dropdown, benchmark_dropdown, gpu_dropdown, precision_dropdown, subrun_ids_textbox, argnv_pairs_textbox, show_details_checkbox, show_verified_sources_checkbox, show_all_precision_checkbox, show_custom_benchmarks_checkbox], outputs=[leaderboard_html, plot, custom_plot, top5_plot, last_update_html])
        show_custom_benchmarks_checkbox.change(wrapped_update_filters, inputs=[model_dropdown, engine_dropdown, benchmark_dropdown, gpu_dropdown, precision_dropdown, subrun_ids_textbox, argnv_pairs_textbox, show_details_checkbox, show_verified_sources_checkbox, show_all_precision_checkbox, show_custom_benchmarks_checkbox], outputs=[leaderboard_html, plot, custom_plot, top5_plot, last_update_html])
        
        # Connect single search button to the advanced filter handler
        search_button.click(
            apply_advanced_filters, 
            inputs=[model_dropdown, engine_dropdown, benchmark_dropdown, gpu_dropdown, precision_dropdown, subrun_ids_textbox, argnv_pairs_textbox, show_details_checkbox, show_verified_sources_checkbox, show_all_precision_checkbox, show_custom_benchmarks_checkbox], 
            outputs=[leaderboard_html, plot, custom_plot, top5_plot, last_update_html]
        )
        
        # Connect metric dropdowns to update handler
        x_metric_dropdown.change(
            update_custom_plot,
            inputs=[x_metric_dropdown, y_metric_dropdown, model_dropdown, engine_dropdown, benchmark_dropdown, gpu_dropdown, precision_dropdown, subrun_ids_textbox, argnv_pairs_textbox, show_details_checkbox, show_verified_sources_checkbox, show_all_precision_checkbox, show_custom_benchmarks_checkbox],
            outputs=[custom_plot, custom_plot_message, top5_plot]
        )
        
        y_metric_dropdown.change(
            update_custom_plot,
            inputs=[x_metric_dropdown, y_metric_dropdown, model_dropdown, engine_dropdown, benchmark_dropdown, gpu_dropdown, precision_dropdown, subrun_ids_textbox, argnv_pairs_textbox, show_details_checkbox, show_verified_sources_checkbox, show_all_precision_checkbox, show_custom_benchmarks_checkbox],
            outputs=[custom_plot, custom_plot_message, top5_plot]
        )
        
    return interface

def find_available_port(start_port, end_port):
    """Find an available port in the given range."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    raise RuntimeError(f"No available ports in range {start_port}-{end_port}")

# Helper function: Create simplified parameter string from args dict
def get_short_args_string(args_dict, max_length=30):
    if not args_dict:
        return ""
    
    # Select the most important parameters
    important_keys = []
    for key in args_dict.keys():
        if key in ['precision', 'engine_type', 'num_gpus', 'quantize', 'max-num-seqs', 'request_rate', 'batch_size']:
            important_keys.append(key)
    
    # If no important parameters found, select the first 2
    if not important_keys and len(args_dict) > 0:
        important_keys = list(args_dict.keys())[:2]
    
    result = []
    for key in important_keys:
        value = args_dict.get(key)
        if isinstance(value, bool):
            if value:
                result.append(f"{key}")
        else:
            result.append(f"{key}={value}")
    
    result_str = ", ".join(result)
    
    # If string is too long, truncate and add ...
    if len(result_str) > max_length:
        return result_str[:max_length-3] + "..."
    
    return result_str

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Engine Arena Leaderboard")
    parser.add_argument("--daemon", action="store_true", help="Run the leaderboard as a background process")
    parser.add_argument("--port", type=int, default=3000, help="Starting port to run the leaderboard on")
    parser.add_argument("--port-end", type=int, default=3015, help="End of port range to try if starting port is unavailable")
    parser.add_argument("--no-share", action="store_true", help="Disable Gradio sharing")
    args = parser.parse_args()
    
    # Find an available port
    try:
        port = find_available_port(args.port, args.port_end)
        if port != args.port:
            print(f"Port {args.port} is in use, using port {port} instead")
    except RuntimeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if args.daemon:
        # Import required libraries for daemon process
        import sys
        import subprocess
        
        # Command to run the leaderboard in the background
        cmd = [sys.executable, __file__, "--port", str(port)]
        if args.no_share:
            cmd.append("--no-share")
            
        # Start the process
        print(f"Starting leaderboard in background on port {port}...")
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Leaderboard started! Access at http://localhost:{port}")
    else:
        # Regular start
        demo = create_interface()
        demo.launch(
            server_port=port, 
            server_name="0.0.0.0", 
            share=not args.no_share
        )
