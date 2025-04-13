import argparse
import os
import sys
from pathlib import Path

from llmbench import build_static_site, available_benchmarks
from flask import Flask, send_from_directory

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
    
    # List benchmarks command
    subparsers.add_parser("list-benchmarks", help="List available benchmarks")
    
    args = parser.parse_args()
    
    if args.command == "build":
        output_path = build_static_site(args.output)
        print(f"Static site built at {output_path.resolve()}")
    
    elif args.command == "serve":
        app = Flask(__name__)
        
        directory = Path(args.directory)
        if not directory.exists():
            print(f"Directory {directory} does not exist. Building it first...")
            directory = build_static_site(args.directory)
        
        @app.route('/', defaults={'path': ''})
        @app.route('/<path:path>')
        def serve_file(path):
            if path == '' or path == '/':
                return send_from_directory(directory, 'index.html')
            return send_from_directory(directory, path)
        
        print(f"Serving at http://localhost:{args.port}")
        app.run(host='0.0.0.0', port=args.port)
    
    elif args.command == "list-benchmarks":
        benchmarks = available_benchmarks()
        if benchmarks:
            print("Available benchmarks:")
            for benchmark in benchmarks:
                print(f"  - {benchmark}")
        else:
            print("No benchmarks found.")
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()