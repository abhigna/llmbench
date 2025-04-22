#!/bin/bash

# Get current directory
CURRENT_DIR=$(pwd)
echo "Current directory: $CURRENT_DIR"

# Check if data/benchmarks directory exists
if [ ! -d "$CURRENT_DIR/data/benchmarks" ]; then
    echo "Error: data/benchmarks directory not found!"
    exit 1
fi

# List all benchmark directories
echo "Available benchmark directories:"
ls -la "$CURRENT_DIR/data/benchmarks"

echo "Running all benchmarks..."
for benchmark in "$CURRENT_DIR/data/benchmarks"/*; do
    if [ -d "$benchmark" ]; then
        benchmark_name=$(basename "$benchmark")
        echo "Running benchmark: $benchmark_name"
        python -m llmbench.benchmark_updater_agent --root-dir "$CURRENT_DIR" --log-level DEBUG -b "$benchmark_name"
    fi
done

echo "Benchmark execution completed."