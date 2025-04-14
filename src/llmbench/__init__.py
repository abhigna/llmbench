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
    
    # Create css and js directories if they don't exist
    css_output_dir = output_path / "css"
    js_output_dir = output_path / "js"
    os.makedirs(css_output_dir, exist_ok=True)
    os.makedirs(js_output_dir, exist_ok=True)
    
    # Copy static files
    css_path = get_static_path() / "css"
    js_path = get_static_path() / "js"
    
    # Copy CSS files
    if css_path.exists():
        for css_file in css_path.glob("*.css"):
            print(f"Copying CSS file: {css_file} to {css_output_dir / css_file.name}")
            shutil.copy(css_file, css_output_dir / css_file.name)
    else:
        print(f"Warning: CSS directory not found at {css_path}")
    
    # Copy JS files
    if js_path.exists():
        for js_file in js_path.glob("*.js"):
            print(f"Copying JS file: {js_file} to {js_output_dir / js_file.name}")
            shutil.copy(js_file, js_output_dir / js_file.name)
    else:
        print(f"Warning: JS directory not found at {js_path}")
    
    # Copy data files
    data_path = get_data_path()
    if data_path.exists():
        # Copy top-level JSON files
        for data_file in data_path.glob("*.json"):
            print(f"Copying data file: {data_file} to {output_path / data_file.name}")
            shutil.copy(data_file, output_path / data_file.name)
        
        # Copy benchmark data files and maintain directory structure
        benchmarks_path = data_path / "benchmarks"
        if benchmarks_path.exists():
            benchmarks_output_dir = output_path / "benchmarks"
            os.makedirs(benchmarks_output_dir, exist_ok=True)
            
            for benchmark_dir in benchmarks_path.iterdir():
                if benchmark_dir.is_dir():
                    benchmark_output_dir = benchmarks_output_dir / benchmark_dir.name
                    os.makedirs(benchmark_output_dir, exist_ok=True)
                    
                    for json_file in benchmark_dir.glob("*.json"):
                        print(f"Copying benchmark file: {json_file} to {benchmark_output_dir / json_file.name}")
                        shutil.copy(json_file, benchmark_output_dir / json_file.name)
    else:
        print(f"Warning: Data directory not found at {data_path}")
    
    # Copy and process template
    template_path = get_templates_path() / "index.html"
    if template_path.exists():
        with open(template_path, "r") as f:
            template = f.read()
        
        with open(output_path / "index.html", "w") as f:
            print(f"Creating index.html at {output_path / 'index.html'}")
            f.write(template)
    else:
        print(f"Warning: Template file not found at {template_path}")
        raise FileNotFoundError(f"Template file not found at {template_path}")
    
    print(f"Build complete. Files written to {output_path.resolve()}")
    return output_path