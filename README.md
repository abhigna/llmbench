# LLM Benchmark Comparison

A static website generator for comparing different LLM models and their benchmark scores. 

## Features

- Responsive layout with filtering options
- Support for multiple benchmark types (LMSYS, MMLU, MT-Bench, etc.)
- Easy deployment to GitHub Pages
- Local development server
- Automated data updates via GitHub Actions

## Installation

```bash
# Clone the repository
git clone https://github.com/abhigna/llmbench.git
cd llmbench

# Install the package in development mode
pip install -e .
```

## Usage

### Local Development

To run a local development server:

```bash
# Build the static site
llmbench build

# Start a local server
llmbench serve
```

This will build the site to the `./docs` directory and serve it at `http://localhost:5000`.

### Updating Benchmark Data

The package includes tools to update benchmark data:

```bash
# Update all benchmarks
update-benchmarks --benchmark all

# Update a specific benchmark
update-benchmarks --benchmark lmsys
```

### Adding a New Benchmark

1. Create a new JSON file in the `data/` directory with your benchmark scores
2. Update the benchmark selector in `templates/index.html`
3. Add an update function in `src/llmbench/update.py`

## Deploying to GitHub Pages

This project is configured to automatically deploy to GitHub Pages using GitHub Actions. The workflow will:

1. Update benchmark data daily
2. Build the static site
3. Deploy to the `gh-pages` branch

You can also manually trigger a deployment from the Actions tab in your GitHub repository.

## Project Structure

```
llmbench/
├── static/                # CSS, JS files
├── templates/             # HTML templates
├── data/                  # JSON data files
├── docs/                  # GitHub Pages deployment target
├── src/llmbench/          # Python package
├── setup.py               # Package setup script
└── README.md              # This file
```

## Customization

### Adding New Models

To add a new model, update the `models.json` file in the `data/` directory with the model details.

### Changing the Layout

The site layout is defined in the following files:

- `templates/index.html` - Main layout structure
- `static/css/style.css` - Styling
- `static/js/main.js` - Filtering and data loading logic