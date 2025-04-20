# src/llmbench/cli.py
import argparse
import os
import sys
import json
from pathlib import Path
from datetime import datetime

from flask import Flask, send_from_directory
from llmbench import build_static_site, available_benchmarks

def main():
    parser = argparse.ArgumentParser(description="LLM Benchmark Comparison Tools")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Build command
    build_parser = subparsers.add_parser("build", help="Build static site")
    build_parser.add_argument(
        "--output", "-o", 
        default="./docs", 
        help="Output directory (default: ./docs)"
    )
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve site locally")
    serve_parser.add_argument(
        "--port", "-p", 
        type=int, 
        default=5000, 
        help="Port to serve on (default: 5000)"
    )
    serve_parser.add_argument(
        "--directory", "-d", 
        default="./docs", 
        help="Directory to serve (default: ./docs)"
    )
    
    # Add benchmark command
    add_parser = subparsers.add_parser("add-benchmark", help="Add a new benchmark")
    add_parser.add_argument(
        "--id", required=True,
        help="Benchmark ID (e.g., 'simplebench')"
    )
    add_parser.add_argument(
        "--name", required=True,
        help="Benchmark name (e.g., 'SimpleBench')"
    )
    add_parser.add_argument(
        "--description", required=True,
        help="Benchmark description"
    )
    add_parser.add_argument(
        "--source-url", required=True,
        help="URL of the benchmark source"
    )
    add_parser.add_argument(
        "--type", required=True,
        choices=["human_pref", "task_based", "knowledge", "reasoning", "other"],
        help="Type of the benchmark"
    )
    add_parser.add_argument(
        "--config", "-c",
        help="Path to JSON file containing metrics, dimensions, and display configuration"
    )
    add_parser.add_argument(
        "--config-json",
        help="JSON string containing metrics, dimensions, and display configuration"
    )
    
    # List benchmarks command
    subparsers.add_parser("list-benchmarks", help="List available benchmarks")
    
    args = parser.parse_args()
    
    if args.command == "build":
        output_path = build_static_site(args.output)
        print(f"Static site built at {output_path.resolve()}")
    
    elif args.command == "serve":
        app = Flask(__name__)
        
        directory = Path(args.directory).resolve()
        if not directory.exists():
            print(f"Directory {directory} does not exist. Building it first...")
            directory = build_static_site(args.directory).resolve()
        
        print(f"Serving files from: {directory}")
        
        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve_file(path):
            if path == '' or path == '/':
                return send_from_directory(directory, 'index.html')
            elif path == 'favicon.ico':
                return '', 204  # No content for favicon requests
            else:
                # Handle subdirectories like css and js
                if '/' in path:
                    parts = path.split('/')
                    subdir = parts[0]
                    filepath = '/'.join(parts[1:])
                    
                    if os.path.exists(directory / subdir):
                        return send_from_directory(directory / subdir, filepath)
                
                # Default case
                return send_from_directory(directory, path)
        
        print(f"Serving at http://localhost:{args.port}")
        app.run(host='0.0.0.0', port=args.port, debug=True)
    
    elif args.command == "add-benchmark":
        data_path = Path(__file__).parent.parent.parent / "data"
        
        # Load existing benchmarks
        benchmarks_path = data_path / "benchmarks.json"
        try:
            with open(benchmarks_path, "r") as f:
                benchmarks = json.load(f)
        except FileNotFoundError:
            benchmarks = []
        
        # Check if benchmark already exists
        if any(b["id"] == args.id for b in benchmarks):
            print(f"Error: Benchmark with ID '{args.id}' already exists")
            sys.exit(1)
        
        # Add new benchmark
        today = datetime.now().strftime("%Y-%m-%d")
        benchmarks.append({
            "id": args.id,
            "name": args.name,
            "description": args.description,
            "source_url": args.source_url,
            "last_updated": today,
            "type": args.type
        })
        
        # Save updated benchmarks
        with open(benchmarks_path, "w") as f:
            json.dump(benchmarks, f, indent=2)
        
        # Create benchmark directory structure
        benchmark_dir = data_path / "benchmarks" / args.id
        os.makedirs(benchmark_dir, exist_ok=True)
        
        # Default configuration
        default_config = {
            "metrics": [
                {
                    "id": "score",
                    "name": "Score",
                    "description": "Overall score",
                    "min": 0,
                    "max": 100,
                    "better": "higher"
                }
            ],
            "dimensions": [
                {
                    "id": "overall",
                    "name": "Overall",
                    "description": "Overall performance"
                }
            ],
            "display": {
                "primary_metric": "score",
                "primary_dimension": "overall",
                "chart_type": "bar"
            }
        }
        
        # Get configuration from file or JSON string if provided
        config = default_config
        if args.config:
            try:
                with open(args.config, "r") as f:
                    user_config = json.load(f)
                    # Update configuration with user-provided values
                    for key in ["metrics", "dimensions", "display"]:
                        if key in user_config:
                            config[key] = user_config[key]
            except (FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Error loading config file: {e}")
                sys.exit(1)
        elif args.config_json:
            try:
                user_config = json.loads(args.config_json)
                # Update configuration with user-provided values
                for key in ["metrics", "dimensions", "display"]:
                    if key in user_config:
                        config[key] = user_config[key]
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON config: {e}")
                sys.exit(1)
        
        # Create benchmark info.json template
        info = {
            "name": args.name,
            "description": args.description,
            "methodology": "",
            "source_url": args.source_url,
            "type": args.type,
            "metrics": config["metrics"],
            "dimensions": config["dimensions"],
            "display": config["display"]
        }
        
        with open(benchmark_dir / "info.json", "w") as f:
            json.dump(info, f, indent=2)
        
        # Create empty benchmark data file
        data = {
            "date": today,
            "source_url": args.source_url,
            "scores": []
        }
        
        with open(benchmark_dir / f"{today}.json", "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"Benchmark '{args.name}' added successfully")
    
    elif args.command == "list-benchmarks":
        benchmarks_path = Path(__file__).parent.parent.parent / "data" / "benchmarks.json"
        try:
            with open(benchmarks_path, "r") as f:
                benchmarks = json.load(f)
                
            if benchmarks:
                print("Available benchmarks:")
                for benchmark in benchmarks:
                    print(f"  - {benchmark['id']}: {benchmark['name']} ({benchmark['last_updated']})")
            else:
                print("No benchmarks found.")
        except FileNotFoundError:
            print("Benchmarks file not found.")
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()