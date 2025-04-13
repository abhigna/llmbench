import os
import json
import shutil
from pathlib import Path

def get_project_root():
    """Returns the path to the project root directory."""
    # In a src-based layout, the project root is 2 levels up from this file
    return Path(__file__).parent.parent.parent

def get_data_path():
    """Returns the path to the data directory."""
    return get_project_root() / "data"

def get_static_path():
    """Returns the path to the static files directory."""
    return get_project_root() / "static"

def get_templates_path():
    """Returns the path to the templates directory."""
    return get_project_root() / "templates"

def load_model_data():
    """Load the model data from the JSON file."""
    with open(get_data_path() / "models.json", "r") as f:
        return json.load(f)

def load_benchmark_data(benchmark_name):
    """Load benchmark data from the JSON file."""
    benchmark_file = f"{benchmark_name}.json"
    try:
        with open(get_data_path() / benchmark_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise ValueError(f"Benchmark '{benchmark_name}' not found")

def available_benchmarks():
    """Return a list of available benchmarks."""
    return [
        path.stem for path in get_data_path().glob("*.json") 
        if path.stem != "models"
    ]

def build_static_site(output_dir):
    """Build a static site in the specified output directory."""
    output_path = Path(output_dir)
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(output_path / "css", exist_ok=True)
    os.makedirs(output_path / "js", exist_ok=True)
    
    # Copy static files
    css_path = get_static_path() / "css"
    js_path = get_static_path() / "js"
    
    for css_file in css_path.glob("*.css"):
        shutil.copy(css_file, output_path / "css" / css_file.name)
    
    for js_file in js_path.glob("*.js"):
        shutil.copy(js_file, output_path / "js" / js_file.name)
    
    # Copy data files
    for data_file in get_data_path().glob("*.json"):
        shutil.copy(data_file, output_path / data_file.name)
    
    # Copy and process template
    with open(get_templates_path() / "index.html", "r") as f:
        template = f.read()
    
    with open(output_path / "index.html", "w") as f:
        f.write(template)
    
    return output_path