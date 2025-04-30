import os
import sys
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import argparse
import logging

# Add parent directory to path to import the data_service module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.data_service import (
    get_file_watcher_instance, load_runs, format_timestamp
)

# Import global leaderboard module
try:
    from src.utils.upload_leaderboard import upload_json_file
    GLOBAL_LEADERBOARD_AVAILABLE = True
except ImportError:
    GLOBAL_LEADERBOARD_AVAILABLE = False


# CSS similar to Chatbot Arena but adapted for our dashboard
CSS = """
body, .gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
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
.sidebar {
    padding: 1rem;
    background-color: #f7f7f7;
    border-radius: 0.5rem;
}
.sidebar-title {
    font-weight: 600;
    margin-bottom: 1rem;
    font-size: 1.2rem;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 2rem;
    table-layout: fixed;
}
th {
    background-color: #f1f1f1;
    padding: 12px 8px;
    text-align: left;
    border-bottom: 1px solid #ddd;
    font-weight: 600;
    vertical-align: top;
    word-wrap: break-word;
    line-height: 1.4;
}
td {
    padding: 12px 8px;
    border-bottom: 1px solid #ddd;
    vertical-align: top;
    word-wrap: break-word;
    min-height: 120px;
    line-height: 1.4;
}
.cell-content {
    max-height: 200px;
    overflow-y: auto;
    overflow-x: hidden;
    padding-right: 8px;
    scrollbar-width: thin;
    border-radius: 4px;
    background-color: #f9f9f9;
    padding: 8px;
    margin-bottom: 4px;
}
.cell-content::-webkit-scrollbar {
    width: 6px;
}
.cell-content::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 3px;
}
.cell-content::-webkit-scrollbar-thumb {
    background: #ccc;
    border-radius: 3px;
}
.cell-content::-webkit-scrollbar-thumb:hover {
    background: #aaa;
}
tr:hover {
    background-color: #f5f5f5;
}
.table-responsive {
    overflow-x: auto;
    max-width: 100%;
    margin-bottom: 1.5rem;
    border-radius: 4px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
@media screen and (max-width: 1200px) {
    table {
        display: block;
        overflow-x: auto;
        white-space: nowrap;
    }
    .cell-content {
        max-height: 150px;
        max-width: 300px;
        white-space: normal;
    }
}
.tooltip {
    position: relative;
    display: inline-block;
    cursor: pointer;
}
.tooltip .tooltiptext {
    visibility: hidden;
    width: 300px;
    background-color: #333;
    color: #fff;
    text-align: left;
    border-radius: 6px;
    padding: 10px;
    position: absolute;
    z-index: 1;
    top: 125%;
    left: 50%;
    margin-left: -150px;
    opacity: 0;
    transition: opacity 0.3s;
    font-size: 0.9rem;
    white-space: pre-wrap;
}
.tooltip:hover .tooltiptext {
    visibility: visible;
    opacity: 1;
}
.run-row {
    font-weight: 600;
    cursor: pointer;
    padding: 8px 4px;
    border-radius: 4px;
    margin-bottom: 4px;
    background-color: #eaeaea;
}
.run-row:hover {
    background-color: #d9d9d9;
}
.subrun-row {
    padding: 4px 8px;
    margin-left: 12px;
    border-left: 2px solid #ddd;
    cursor: pointer;
}
.subrun-row:hover {
    background-color: #f0f0f0;
}
.selected {
    background-color: #d0e8ff !important;
}
.metrics-good {
    color: #28a745;
}
.metrics-bad {
    color: #dc3545;
}
.result-container {
    margin-top: 1rem;
    overflow-x: auto;
    min-height: 400px;
}
.app {
  max-width: 100% !important;
  padding-left: 5% !important;
  padding-right: 5% !important;
}
h2, h3 {
  margin-top: 1.5rem;
  margin-bottom: 1rem;
}
.plot-container {
  margin-bottom: 2rem;
}
#performance-comparison {
  margin-top: 1rem;
  margin-bottom: 2rem;
}
"""

logger = logging.getLogger(__name__)

class BenchmarkState:
    def __init__(self):
        self.current_run = None
        self.current_subrun = None
        self.last_update_time = 0
        self.file_watcher = get_file_watcher_instance()
        self.is_refreshing = False  # Flag to prevent recursive refreshes
        
        # Initial data load
        self.refresh_runs()
        
    def refresh_runs(self, force=False):
        """Refresh data from cache"""
        # Prevent recursive refreshes
        if self.is_refreshing:
            logger.warning("Already refreshing, skipping redundant refresh")
            return False
        
        try:
            self.is_refreshing = True
            current_time = time.time()
            self.file_watcher.register_listener("*", self.on_file_changed)
            self.runs = load_runs(source="local")
            self.last_update_time = current_time
            return True
        finally:
            self.is_refreshing = False
    
    def on_file_changed(self, file_path):
        """Callback for when a file changes"""
        # Avoid recursive refreshes
        if self.is_refreshing:
            return
            
        self.last_update_time = 0  # Force refresh on next check
        # Immediately reload data on file change
        self.refresh_runs(force=True)
        
    def select_run(self, run_index):
        """Select a run by index"""
        if 0 <= run_index < len(self.runs):
            self.current_run = self.runs[run_index]
            self.current_subrun = None
            return self.current_run
        return None
    
    def select_subrun(self, run_index, subrun_index):
        """Select a subrun by indices"""
        if 0 <= run_index < len(self.runs):
            run = self.runs[run_index]
            if 0 <= subrun_index < len(run.get("subruns", [])):
                self.current_run = run
                self.current_subrun = run["subruns"][subrun_index]
                return self.current_subrun
        return None

def create_performance_charts(run):
    """Create performance charts for a run's subruns"""
    if not run or not run.get("subruns"):
        return None
    
    subruns = run.get("subruns", [])
    
    # Prepare data for visualization
    subrun_ids = []
    full_ids = []
    input_throughputs = []
    output_throughputs = []
    ttfts = []
    tpots = []
    per_request_throughputs = []  # New metric
    benchmark_types = []

    # Additional data for enhanced hover text
    engine_args_info = []
    env_vars_info = []
    benchmark_config_info = []
    
    for subrun in subruns:
        # Get subrun ID and extract the last part for display
        full_id = subrun.get("id", "Unknown")
        # Extract the last part after the final hyphen
        short_id = full_id.split('-')[-1] if '-' in full_id else full_id
        subrun_ids.append(short_id)
        full_ids.append(full_id)
        
        # Extract metrics
        metrics = subrun.get("metrics", {})
        input_throughputs.append(float(metrics.get("input_throughput", 0)))
        output_throughputs.append(float(metrics.get("output_throughput", 0)))
        ttfts.append(float(metrics.get("mean_ttft_ms", 0)))
        
        # Get TPOT and calculate per request throughput (1/TPOT)
        tpot = float(metrics.get("mean_tpot_ms", 0))
        tpots.append(tpot)
        
        # Calculate per request throughput (1/TPOT) - avoid division by zero
        if tpot > 0:
            per_request_throughputs.append(1000.0 / tpot)  # Convert to requests per second
        else:
            per_request_throughputs.append(0)

        # Extract engine args, env vars, and benchmark config for hover text
        engine = subrun.get("engine", {})
        benchmark = subrun.get("benchmark", {})
        
        # Format engine args
        engine_args = engine.get("engine_args", {})
        engine_args_str = ""
        if isinstance(engine_args, dict):
            for key, value in engine_args.items():
                engine_args_str += f"{key}: {value}<br>"
        elif isinstance(engine_args, list):
            engine_args_str = "<br>".join(engine_args)
        else:
            engine_args_str = str(engine_args)
        engine_args_info.append(engine_args_str)
        
        # Format env vars
        env_vars = engine.get("env_vars", {})
        env_vars_str = ""
        if isinstance(env_vars, dict):
            for key, value in env_vars.items():
                env_vars_str += f"{key}: {value}<br>"
        else:
            env_vars_str = str(env_vars)
        env_vars_info.append(env_vars_str)
        
        # Format benchmark config with type at the beginning
        benchmark_config = benchmark.get("config", {})
        benchmark_type = benchmark.get("type", "Unknown")
        benchmark_config_str = f"type: {benchmark_type}<br>"  # Add type at the beginning
        if isinstance(benchmark_config, dict):
            for key, value in benchmark_config.items():
                benchmark_config_str += f"{key}: {value}<br>"
        else:
            benchmark_config_str = str(benchmark_config)
        benchmark_config_info.append(benchmark_config_str)

    # Create enhanced hovertext with engine args, env vars, and benchmark config
    def create_hover_text(i, metric_name, value):
        return (
            f"<b>ID:</b> {full_ids[i]}<br>"
            f"<b>Engine:</b> {engine.get('name', 'N/A')}<br>"
            f"<b>{metric_name}:</b> {value}<br><br>"
            f"<b>Engine Args:</b><br>{engine_args_info[i]}<br>"
            f"<b>Env Vars:</b><br>{env_vars_info[i]}<br>"
            f"<b>Benchmark Config:</b><br>{benchmark_config_info[i]}"
        )
    
    input_hovertext = [create_hover_text(i, "Input Throughput", val) 
                      for i, val in enumerate(input_throughputs)]
    output_hovertext = [create_hover_text(i, "Output Throughput", val) 
                       for i, val in enumerate(output_throughputs)]
    ttft_hovertext = [create_hover_text(i, "TTFT", val) 
                     for i, val in enumerate(ttfts)]
    tpot_hovertext = [create_hover_text(i, "TPOT", val) 
                     for i, val in enumerate(tpots)]
    per_request_hovertext = [create_hover_text(i, "Per Request Throughput", f"{val:.2f}") 
                            for i, val in enumerate(per_request_throughputs)]
    
    # Create subplots with 2 rows, 3 columns (adding one for per request throughput)
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=('Input Throughput', 'Output Throughput', 'Per Request Throughput', 
                        'Time to First Token (TTFT)', 'Time Per Output Token (TPOT)', ''),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # Add bars for input throughput (higher is better)
    fig.add_trace(
        go.Bar(
            x=subrun_ids,
            y=input_throughputs,
            marker_color='royalblue',
            name='Input Throughput',
            hovertext=input_hovertext
        ),
        row=1, col=1
    )
    
    # Add bars for output throughput (higher is better)
    fig.add_trace(
        go.Bar(
            x=subrun_ids,
            y=output_throughputs,
            marker_color='mediumseagreen',
            name='Output Throughput',
            hovertext=output_hovertext
        ),
        row=1, col=2
    )
    
    # Add bars for per request throughput (higher is better)
    fig.add_trace(
        go.Bar(
            x=subrun_ids,
            y=per_request_throughputs,
            marker_color='purple',  # Different color for new metric
            name='Per Request Throughput',
            hovertext=per_request_hovertext
        ),
        row=1, col=3
    )
    
    # Add bars for TTFT (lower is better)
    fig.add_trace(
        go.Bar(
            x=subrun_ids,
            y=ttfts,
            marker_color='darkorange',
            name='TTFT',
            hovertext=ttft_hovertext
        ),
        row=2, col=1
    )
    
    # Add bars for TPOT (lower is better)
    fig.add_trace(
        go.Bar(
            x=subrun_ids,
            y=tpots,
            marker_color='indianred',
            name='TPOT',
            hovertext=tpot_hovertext
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="Performance Metrics by Sub-run",
        title_x=0.5,
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
            size=12
        ),
        margin=dict(t=100, b=100)  # Add more margin at top and bottom
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Tokens/sec", row=1, col=1)
    fig.update_yaxes(title_text="Tokens/sec", row=1, col=2)
    fig.update_yaxes(title_text="Tokens/sec/req", row=1, col=3)  # New label for per request throughput
    fig.update_yaxes(title_text="Milliseconds", row=2, col=1)
    fig.update_yaxes(title_text="Milliseconds", row=2, col=2)
    
    # Update x-axis layout for better readability with long IDs
    fig.update_xaxes(tickangle=45, row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_xaxes(tickangle=45, row=1, col=3)
    fig.update_xaxes(tickangle=45, row=2, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=2)
    
    return fig

def display_run(state, run_index):
    """Display a run's results with visual charts for metrics."""
    # Refresh data if needed
    state.refresh_runs()
    
    run = state.select_run(run_index)
    
    if not run or not run.get("subruns"):
        return "No data available for this run.", None
    
    # Sorted By Subrun Start Time
    run["subruns"] = sorted(run["subruns"], key=lambda x: x.get("start_time", ""), reverse=False)

    # Basic run information
    html = "<h2>Run Summary</h2>"
    html += f"<p><strong>Run ID:</strong> {run.get('id', 'N/A')}</p>"
    html += f"<p><strong>Start Time:</strong> {format_timestamp(run.get('start_time', 'N/A'))}</p>"
    
    # Create visualization
    fig = create_performance_charts(run)
    
    # Add metrics table for detailed comparison
    html += "<h3>Sub-runs Metrics Comparison</h3>"
    html += "<div class='table-responsive'>"
    html += "<table>"
    html += "<tr>"
    html += "<th style='width: 10%;'>ID</th>"
    html += "<th style='width: 5%;'>Success</th>"
    html += "<th style='width: 8%;'>Engine</th>"
    html += "<th style='width: 15%;'>Env Vars</th>"  
    html += "<th style='width: 15%;'>Engine Args</th>" 
    html += "<th style='width: 8%;'>Benchmark</th>"
    html += "<th style='width: 15%;'>Benchmark Config</th>"
    html += "<th style='width: 5%;'>Input Throughput</th>"
    html += "<th style='width: 5%;'>Output Throughput</th>"
    html += "<th style='width: 5%;'>TTFT (ms)</th>"
    html += "<th style='width: 5%;'>TPOT (ms)</th>"
    html += "<th style='width: 5%;'>Per Request Throughput (t/s/req)</th>"
    html += "</tr>"
    
    for subrun in run.get("subruns", []):
        engine = subrun.get("engine", {})
        benchmark = subrun.get("benchmark", {})
        metrics = subrun.get("metrics", {})
        exit_code = subrun.get("exit_code")
        success_emoji = "✅" if exit_code == 0 else "❌"
        
         # Helper to format metrics safely
        def format_metric(value):
            try:
                return f"{float(value):.2f}"
            except (ValueError, TypeError):
                return "N/A"
            
        # Calculate per request throughput
        tpot = float(metrics.get("tpot", 0))
        per_request_throughput = "N/A"
        if tpot > 0:
            per_request_throughput = f"{1000.0 / tpot:.2f}"  # Convert to requests per second with 2 decimal places
        
        # Format environment variables
        env_vars_html = "<div class='cell-content'>"
        if engine.get("env_vars"):
            for key, value in engine["env_vars"].items():
                env_vars_html += f"{key}: {value}<br>"
        else:
            env_vars_html += ""
        env_vars_html += "</div>"
        
        # Format engine args
        engine_args_html = "<div class='cell-content'>"
        if engine.get("engine_args"):
            if isinstance(engine["engine_args"], dict):
                for key, value in engine["engine_args"].items():
                    engine_args_html += f"--{key}: {value}<br>"
            elif isinstance(engine["engine_args"], list):
                engine_args_html += "<br>".join(engine["engine_args"])
            else:
                engine_args_html += str(engine["engine_args"])
        else:
            engine_args_html += "N/A"
        engine_args_html += "</div>"
        
        # Format benchmark config
        benchmark_config_html = "<div class='cell-content'>"
        if benchmark.get("config"):
            for key, value in benchmark["config"].items():
                benchmark_config_html += f"{key}: {value}<br>"
        else:
            benchmark_config_html += "N/A"
        benchmark_config_html += "</div>"
        
        html += "<tr>"
        html += f"<td>{subrun.get('id', 'N/A')}</td>"
        html += f"<td style='text-align: center;'>{success_emoji}</td>" 
        html += f"<td>{engine.get('name', 'N/A')}</td>"
        html += f"<td>{env_vars_html}</td>"
        html += f"<td>{engine_args_html}</td>"
        html += f"<td>{benchmark.get('type', 'N/A')}</td>"
        html += f"<td>{benchmark_config_html}</td>"
        html += f"<td>{format_metric(metrics.get('input_throughput'))}</td>"
        html += f"<td>{format_metric(metrics.get('output_throughput'))}</td>"
        html += f"<td>{format_metric(metrics.get('mean_ttft_ms'))}</td>"
        html += f"<td>{format_metric(metrics.get('mean_tpot_ms'))}</td>"
        html += f"<td>{per_request_throughput}</td>"
        html += "</tr>"
    
    html += "</table>"
    html += "</div>"
    
    return html, fig

def display_subrun(state, run_index, subrun_index):
    """Display a subrun's detailed results."""
    # Refresh data if needed
    state.refresh_runs()
    
    subrun = state.select_subrun(run_index, subrun_index)
    
    if not subrun:
        return "No data available for this sub-run."
    
    engine = subrun.get("engine", {})
    benchmark = subrun.get("benchmark", {})
    metrics = subrun.get("metrics", {})
    
    # Calculate per request throughput
    tpot = float(metrics.get("tpot", 0))
    per_request_throughput = "N/A"
    if tpot > 0:
        per_request_throughput = f"{1000.0 / tpot:.2f}"  # Convert to requests per second
    
    html = "<h2>Sub-run Details</h2>"
    
    # Determine success status
    exit_code = subrun.get("exit_code")
    success_status = "✅ Success" if exit_code == 0 else "❌ Failed"
    html += f"<p><strong>Success:</strong> {success_status}</p>"
    
    html += f"<p><strong>ID:</strong> {subrun.get('id', 'N/A')}</p>"
    html += f"<p><strong>Start Time:</strong> {format_timestamp(subrun.get('start_time', 'N/A'))}</p>"
    html += f"<p><strong>Model:</strong> {subrun.get('model', 'N/A')}</p>"
    
    html += "<h3>Engine Information</h3>"
    html += "<div class='table-responsive'>"
    html += "<table>"
    html += "<tr><th style='width: 25%;'>Name</th><td>" + engine.get("name", "N/A") + "</td></tr>"
    
    gpus = engine.get("gpu_info", {}).get("gpus", [])
    gpu_names = [gpu.get("name", "Unknown") for gpu in gpus]
    html += f"<tr><th>GPUs</th><td>{', '.join(gpu_names)} (x{len(gpus)})</td></tr>"
    
    html += "<tr><th>Environment Variables</th><td><div class='cell-content'>"
    for key, value in engine["env_vars"].items():
        html += f"{key}: {value}<br>"
    html += "</div></td></tr>"
    
    html += "<tr><th>Engine Arguments</th><td><div class='cell-content'>"
    if isinstance(engine["engine_args"], dict):
        for key, value in engine["engine_args"].items():
            html += f"--{key}: {value}<br>"
    elif isinstance(engine["engine_args"], list):
        html += "<br>".join(engine["engine_args"])
    html += "</div></td></tr>"
    
    html += "</table>"
    html += "</div>"
    
    html += "<h3>Benchmark Information</h3>"
    html += "<div class='table-responsive'>"
    html += "<table>"
    html += "<tr><th style='width: 25%;'>Type</th><td>" + benchmark.get("type", "N/A") + "</td></tr>"
    
    # if benchmark.get("config"):
    html += "<tr><th>Configuration</th><td><div class='cell-content'>"
    for key, value in benchmark["config"].items():
        html += f"{key}: {value}<br>"
    html += "</div></td></tr>"
    
    html += "</table>"
    html += "</div>"
    
    html += "<h3>Performance Metrics</h3>"
    html += "<div class='table-responsive'>"
    html += "<table>"
    
    for key, value in metrics.items():
        html += f"<tr><th style='width: 25%;'>{key}</th><td>{value}</td></tr>"
    
    # Add per request throughput to metrics
    html += f"<tr><th style='width: 25%;'>Per Request Throughput (t/s/req)</th><td>{per_request_throughput}</td></tr>"
    
    html += "</table>"
    html += "</div>"
    
    # Execution details section with better separation
    # if subrun.get("exit_code") is not None:
    html += "<h3>Execution Status</h3>"
    html += f"<p><strong>Exit Code:</strong> {subrun['exit_code']}</p>"
    
    # Output in its own section
    # if subrun.get("stdout"):
    html += "<h3>Standard Output</h3>"
    html += "<div style='margin-bottom: 20px;'>"
    html += f"<pre style='white-space: pre-wrap; word-break: break-word; background-color: #f8f8f8; padding: 12px; border-radius: 4px; max-height: 300px; overflow-y: auto;'>{subrun['stdout']}</pre>"
    html += "</div>"
    
    # Error in its own section
    # if subrun.get("stderr"):
    html += "<h3>Standard Error</h3>"
    html += "<div style='margin-bottom: 20px;'>"
    html += f"<pre style='white-space: pre-wrap; word-break: break-word; background-color: #f8f8f8; padding: 12px; border-radius: 4px; max-height: 300px; overflow-y: auto;'>{subrun['stderr']}</pre>"
    html += "</div>"
    
    # Full engine arguments in its own section with better styling
    # if engine.get("full_engine_args"):
    html += "<h3>Full Engine Arguments</h3>"
    html += "<div style='margin-bottom: 20px;'>"
    html += "<pre style='white-space: pre-wrap; word-break: break-word; background-color: #f8f8f8; padding: 12px; border-radius: 4px; max-height: 300px; overflow-y: auto;'>"
    if engine.get("full_engine_args"):
        if isinstance(engine["full_engine_args"], dict):
            for key, value in sorted(engine["full_engine_args"].items()):
                html += f"--{key}={value}\n"
        elif isinstance(engine["full_engine_args"], list):
            html += "\n".join(engine["full_engine_args"])
        else:
            html += str(engine["full_engine_args"])
    else:
        html += "N/A"
    html += "</pre>"
    html += "</div>"
    
    # Full environment variables in its own section with better styling
    # if engine.get("full_env_vars"):
    html += "<h3>Full Environment Variables</h3>"
    html += "<div style='margin-bottom: 20px;'>"
    html += "<pre style='white-space: pre-wrap; word-break: break-word; background-color: #f8f8f8; padding: 12px; border-radius: 4px; max-height: 300px; overflow-y: auto;'>"
    if engine.get("full_env_vars"):
        if isinstance(engine["full_env_vars"], dict):
            for key, value in sorted(engine["full_env_vars"].items()):
                html += f"{key}={value}\n"
        else:
            html += str(engine["full_env_vars"])
    else:
        html += "N/A"
    html += "</pre>"
    html += "</div>"

    html += "<h3>Full Benchmark Config</h3>"
    html += "<div style='margin-bottom: 20px;'>"
    html += "<pre style='white-space: pre-wrap; word-break: break-word; background-color: #f8f8f8; padding: 12px; border-radius: 4px; max-height: 300px; overflow-y: auto;'>"
    html += str(benchmark["namespace_params"])
    html += "</pre>"
    html += "</div>"
    
    return html

def share_to_global_leaderboard(state, run_index, subrun_index=None):
    """Share a subrun to the global leaderboard"""
    # If global leaderboard is not available, return error message
    if not GLOBAL_LEADERBOARD_AVAILABLE:
        return "Global leaderboard module is not available. Make sure it's properly installed."
    
    # Only allow sharing subruns, not main runs
    if subrun_index is None:
        return "Please select a specific sub-run to share. Main runs cannot be shared directly."
    
    # Share a specific subrun
    subrun = state.select_subrun(run_index, subrun_index)
    if not subrun:
        return "No data available for this sub-run."
    
    # Construct the JSON file path from the subrun data
    subrun_id = subrun.get("id")
    parent_run_id = subrun.get("parent_run_id")
    
    if not subrun_id or not parent_run_id:
        return "Missing ID information for this sub-run."
    
    # Construct the path based on the standard pattern:
    # results/run-XXX/subrun-YYY/subrun-YYY.json
    json_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
        parent_run_id,
        subrun_id,
        f"{subrun_id}.json"
    )
    
    # Verify the file exists
    if not os.path.exists(json_file):
        return f"File not found at expected path: {json_file}"
    
    # Upload to global leaderboard
    try:
        success, message = upload_json_file(json_file)
        print(success, message)
        if success:
            return f"Successfully shared to global leaderboard: {message}"
        else:
            return f"Failed to share to global leaderboard: {message}"
    except Exception as e:
        return f"Error sharing to global leaderboard: {str(e)}"

def interface():
    """Create the Gradio interface."""
    state = BenchmarkState()
    
    # If no runs, display a message
    if not state.runs:
        return gr.HTML("<h2>No benchmark runs found in the results directory.</h2>")
    
    # Create variables to store the default run information
    default_run_index = 0  # Select the first run by default
    default_run_label = None
    default_result_html = None
    default_chart = None
    default_subrun_choices = None
    
    # Creating wrapper functions that don't require passing the state
    def wrapped_update_subrun_choices(run_label, value_map):
        if not run_label:
            return gr.Dropdown(
                choices=["No subruns available - select a run first"],
                interactive=False
            ), gr.HTML("Select a run from the dropdown to view results."), None
        
        # Extract the dictionary from the State object if needed
        if hasattr(value_map, 'value'):
            value_map = value_map.value
        
        run_index = int(value_map.get(run_label, 0))
        run = state.runs[run_index]
        
        # Display run details
        run_html, run_chart = display_run(state, run_index)
        
        # Build subrun choices
        subrun_choices = []
        for j, subrun in enumerate(run.get("subruns", [])):
            full_id = subrun.get("id", f"Subrun-{j}")
            # Extract the last part after the final hyphen for display
            short_id = full_id.split('-')[-1] if '-' in full_id else full_id
            subrun_choices.append(f"Subrun: {short_id} ({full_id})")
        
        # If no subruns are found
        if not subrun_choices:
            return gr.Dropdown(
                choices=["No subruns available for this run"],
                interactive=False
            ), gr.HTML(run_html), run_chart
        
        # Return an updated dropdown with the new choices
        return gr.Dropdown(
            choices=subrun_choices,
            interactive=True
        ), gr.HTML(run_html), run_chart
    
    def wrapped_display_selected_subrun(run_label, subrun_choice, value_map):
        if not run_label:
            return "Select a run first."
            
        if not subrun_choice or subrun_choice in ["No subruns available - select a run first", "No subruns available for this run"]:
            return "Select a valid sub-run to view detailed metrics."
        
        # Extract the dictionary from the State object if needed
        if hasattr(value_map, 'value'):
            value_map = value_map.value
        
        run_index = int(value_map.get(run_label, 0))
        run = state.runs[run_index]
        
        # Get full subrun ID from the selected choice
        # Format is now "Subrun: short_id (full_id)"
        if "(" in subrun_choice and ")" in subrun_choice:
            # Extract full ID from between parentheses
            selected_subrun_id = subrun_choice.split("(")[1].split(")")[0]
        else:
            # Backward compatibility - remove "Subrun: " prefix if the old format is still in use
            selected_subrun_id = subrun_choice.replace("Subrun: ", "")
        
        # Find the subrun index directly by ID
        subrun_index = None
        for j, subrun in enumerate(run.get("subruns", [])):
            if subrun.get("id") == selected_subrun_id:
                subrun_index = j
                break
        
        if subrun_index is not None:
            return display_subrun(state, run_index, subrun_index)
        else:
            return f"Failed to find the selected sub-run: {selected_subrun_id}"
    
    def wrapped_refresh_data(run_label, subrun_choice, value_map):
        # Force a cache refresh to get the latest data
        state.refresh_runs(force=True)
        
        # Extract the dictionary from the State object if needed
        if hasattr(value_map, 'value'):
            value_map = value_map.value
        
        # Rebuild run choices and value map
        run_choices = []
        new_value_map = {}
        for i, run in enumerate(state.runs):
            run_id = run.get("id", f"Run-{i}")
            label = f"Run: {run_id}"
            run_choices.append(label)
            new_value_map[label] = str(i)
        
        # Handle case when there are no runs
        if not run_choices:
            return (
                gr.Dropdown(choices=["No runs available"]), 
                gr.Dropdown(choices=["No subruns available"]),
                "No benchmark runs found. Please add data to the results directory.",
                gr.State(value={})  # Update value map
            )
        
        # Always select the first run by default
        current_run_index = 0
        selected_run_label = run_choices[0]
        
        # Only try to find the previous run if one was selected
        if run_label and run_label not in ["No runs available"]:
            # Try to find the same run ID in the updated list
            old_index = int(value_map.get(run_label, 0))
            if old_index < len(state.runs):
                old_run = state.runs[old_index]
                old_run_id = old_run.get("id", "")
                
                # Look for the same run ID in the updated list
                found_match = False
                for i, run in enumerate(state.runs):
                    if run.get("id") == old_run_id:
                        current_run_index = i
                        selected_run_label = run_choices[i]
                        found_match = True
                        break
        
        # Update run dropdown with the selected run
        updated_run_dropdown = gr.Dropdown(
            choices=run_choices,
            value=selected_run_label  # Always set a value
        )
        
        # Now use wrapped_update_subrun_choices to get all the data we need
        # We'll pass the selected run label and the new value map to get consistent results
        updated_subrun_dropdown, run_html, run_chart = wrapped_update_subrun_choices(selected_run_label, new_value_map)
        
        # Return the complete set of updated components
        return updated_run_dropdown, updated_subrun_dropdown, run_html, gr.State(value=new_value_map)
    
    # Build the layout
    with gr.Blocks(theme=gr.themes.Default(), css=CSS) as demo:
        gr.HTML('<h1 class="title">Inference Engine Arena Dashboard</h1>')
        
        # Add JavaScript for auto-refresh
        auto_refresh_js = """
        <script>
        let lastRefreshTime = Date.now();
        
        // Auto-refresh function
        function checkForRefresh() {
            const now = Date.now();
            // Refresh every 10 seconds
            if (now - lastRefreshTime > 10000) {
                // Find and click the refresh button
                const refreshBtn = document.querySelector('button[aria-label="Refresh Data"]');
                if (refreshBtn) {
                    refreshBtn.click();
                    console.log('Auto-refreshed dashboard');
                    lastRefreshTime = now;
                }
            }
            
            // Continue checking
            setTimeout(checkForRefresh, 2000);
        }
        
        // Start the refresh cycle
        setTimeout(checkForRefresh, 10000);
        </script>
        """
        gr.HTML(auto_refresh_js)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Create a run selection component
                run_choices = []
                for i, run in enumerate(state.runs):
                    # Use the full run ID as the display text to ensure uniqueness
                    run_id = run.get("id", f"Run-{i}")
                    model = run.get("model", "Unknown")
                    
                    # Add the run to the choices with minimal formatting
                    run_choices.append({
                        "label": f"Run: {run_id}",
                        "value": str(i)
                    })
                    
                    # Set the default selection to the first run
                    if i == default_run_index:
                        default_run_label = f"Run: {run_id}"
                        default_result_html, default_chart = display_run(state, default_run_index)
                
                # Create run selector with default selection
                gr.Markdown("### Select a Run")
                run_selector = gr.Dropdown(
                    choices=[item["label"] for item in run_choices],
                    value=default_run_label,  # Set default selection
                    label="Benchmark Runs",
                    info="Select a run to view details",
                    elem_id="run-selector"
                )
                
                # Create a hidden state to store the mapping from run labels to values
                run_values = {item["label"]: item["value"] for item in run_choices}
                run_value_map = gr.State(value=run_values)
                
                # Prepare the default subrun choices
                if default_run_index < len(state.runs):
                    default_run = state.runs[default_run_index]
                    default_subrun_choices = []
                    
                    for j, subrun in enumerate(default_run.get("subruns", [])):
                        full_id = subrun.get("id", f"Subrun-{j}")
                        # Extract the last part after the final hyphen for display
                        short_id = full_id.split('-')[-1] if '-' in full_id else full_id
                        default_subrun_choices.append(f"Subrun: {short_id} ({full_id})")
                    
                    if not default_subrun_choices:
                        default_subrun_choices = ["No subruns available for this run"]
                
                # Create subrun selector with default choices
                gr.Markdown("### Select a Sub-run (optional)")
                subrun_selector = gr.Dropdown(
                    choices=default_subrun_choices if default_subrun_choices else ["No subruns available - select a run first"],
                    value=None,  # No default subrun selection
                    label="Sub-runs",
                    info="Select a sub-run for detailed metrics",
                    elem_id="subrun-selector",
                    interactive=bool(default_subrun_choices)  # Enable if there are subruns
                )
                
                # Add refresh button
                refresh_btn = gr.Button("Refresh Data", variant="primary")
                
                # Add share button if global leaderboard is available
                if GLOBAL_LEADERBOARD_AVAILABLE:
                    share_btn = gr.Button("Share Subrun to Global Leaderboard", variant="secondary")
                
            with gr.Column(scale=3):
                # Result container for HTML content
                result_html = gr.HTML(
                    value=default_result_html if default_result_html else "<div class='result-container'>Select a run from the dropdown to view results.</div>",
                    elem_id="result-container"
                )
                
                # Performance chart - Use gr.Group instead of gr.Box
                with gr.Group(elem_id="performance-comparison", elem_classes="plot-container"):
                    performance_chart = gr.Plot(
                        value=default_chart,
                        visible=True, 
                        label="Performance Comparison"
                    )
        
        # Update subrun choices when run selection changes
        run_selector.change(
            fn=wrapped_update_subrun_choices,
            inputs=[run_selector, run_value_map],
            outputs=[subrun_selector, result_html, performance_chart]
        )
        
        # Update result when subrun selection changes
        def safe_subrun_update(run_label, subrun_choice, value_map):
            try:
                return wrapped_display_selected_subrun(run_label, subrun_choice, value_map)
            except Exception as e:
                logger.error(f"Error updating subrun: {e}")
                # Handle error by resetting selection
                return "There was an error displaying the selected subrun. Please try selecting a different one."
        
        subrun_selector.change(
            fn=safe_subrun_update,
            inputs=[run_selector, subrun_selector, run_value_map],
            outputs=result_html
        )
        
        # Make the share button work
        if GLOBAL_LEADERBOARD_AVAILABLE:
            def wrapped_share_to_global_leaderboard(run_label, subrun_choice, value_map):
                """Wrapper for the share function to handle UI state"""
                if not run_label:
                    return "<div style='padding: 10px; background-color: #fff3cd; color: #856404; border-radius: 4px;'>⚠️ Select a run first before sharing.</div>"
                
                # Only allow sharing if a subrun is selected
                if not subrun_choice or subrun_choice in ["No subruns available - select a run first", "No subruns available for this run"]:
                    return "<div style='padding: 10px; background-color: #fff3cd; color: #856404; border-radius: 4px;'>⚠️ Please select a specific sub-run to share. Main runs cannot be shared directly.</div>"
                
                # Extract the dictionary from the State object if needed
                if hasattr(value_map, 'value'):
                    value_map = value_map.value
                
                run_index = int(value_map.get(run_label, 0))
                run = state.runs[run_index]
                
                # Get full subrun ID from the selected choice
                if "(" in subrun_choice and ")" in subrun_choice:
                    # Extract full ID from between parentheses
                    selected_subrun_id = subrun_choice.split("(")[1].split(")")[0]
                else:
                    # Backward compatibility - remove "Subrun: " prefix if the old format is still in use
                    selected_subrun_id = subrun_choice.replace("Subrun: ", "")
                
                # Find the subrun index directly by ID
                subrun_index = None
                for j, subrun in enumerate(run.get("subruns", [])):
                    if subrun.get("id") == selected_subrun_id:
                        subrun_index = j
                        break
                
                if subrun_index is not None:
                    result = share_to_global_leaderboard(state, run_index, subrun_index)
                    if "Successfully" in result:
                        return f"<div style='padding: 10px; background-color: #d4edda; color: #155724; border-radius: 4px;'>✅ {result}</div>"
                    else:
                        return f"<div style='padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 4px;'>❌ {result}</div>"
                else:
                    return f"<div style='padding: 10px; background-color: #f8d7da; color: #721c24; border-radius: 4px;'>❌ Failed to find the selected sub-run: {selected_subrun_id}</div>"
            
            share_btn.click(
                fn=wrapped_share_to_global_leaderboard,
                inputs=[run_selector, subrun_selector, run_value_map],
                outputs=result_html
            )
        
        # Add back the refresh button functionality
        refresh_btn.click(
            fn=wrapped_refresh_data,
            inputs=[run_selector, subrun_selector, run_value_map],
            outputs=[run_selector, subrun_selector, result_html, run_value_map]
        )
        
        # Add a separate JavaScript function to ensure the subrun selector is always cleared when refreshing
        refresh_js = """
        <script>
        // Find the refresh button and add an event listener
        document.addEventListener('DOMContentLoaded', function() {
            // Wait for Gradio to fully load
            setTimeout(function() {
                const refreshBtn = document.querySelector('button[aria-label="Refresh Data"]');
                const subrunSelector = document.querySelector('#subrun-selector select');
                const runSelector = document.querySelector('#run-selector select');
                
                if (refreshBtn && subrunSelector) {
                    refreshBtn.addEventListener('click', function() {
                        // Clear the subrun selector value
                        subrunSelector.value = '';
                        console.log('Cleared subrun selection');
                        
                        // Add code to manually trigger the run_selector change event after a short delay
                        setTimeout(function() {
                            // Create and dispatch a change event on the run selector
                            const event = new Event('change', { bubbles: true });
                            runSelector.dispatchEvent(event);
                            console.log('Manually triggered run selector change event');
                        }, 500); // 500ms delay to ensure the UI has updated with the refreshed data
                    });
                }
            }, 2000);
        });
        </script>
        """
        gr.HTML(refresh_js)
        
        # Initial load, don't use 'every' parameter as it's not supported in this version
        demo.load(
            fn=lambda: None,  # No-op function
            inputs=None,
            outputs=None
        )
    
    return demo

import socket
def find_available_port(start_port, end_port):
    """Find an available port in the given range."""
    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(('localhost', port)) != 0:
                return port
    raise RuntimeError(f"No available ports in range {start_port}-{end_port}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference Engine Arena Dashboard")
    parser.add_argument("--daemon", action="store_true", help="Run the dashboard as a background process")
    parser.add_argument("--port", type=int, default=3000, help="Starting port to run the dashboard on")
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
        
        # Command to run the dashboard in the background
        cmd = [sys.executable, __file__, "--port", str(port)]
        if args.no_share:
            cmd.append("--no-share")
            
        # Start the process
        print(f"Starting dashboard in background on port {port}...")
        subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Dashboard started! Access at http://localhost:{port}")
    else:
        # Regular start
        demo = interface()
        demo.launch(
            server_port=port, 
            server_name="0.0.0.0", 
            share=not args.no_share
        ) 