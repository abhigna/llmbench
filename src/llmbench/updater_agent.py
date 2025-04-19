import requests
import json
import os
import argparse
from datetime import datetime
from pathlib import Path
import sys
import logging
import dotenv

dotenv.load_dotenv()

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Endpoints and Model
JINA_READER_URL = 'https://r.jina.ai/'
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
# Use a capable but cost-effective model like Gemini Flash
LLM_MODEL = "google/gemini-2.0-flash-001"

# --- Environment Variables ---
JINA_API_KEY = os.getenv('JINA_API_KEY')
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')

if not JINA_API_KEY:
    logging.error("Error: JINA_API_KEY environment variable not set.")
    sys.exit(1)
if not OPENROUTER_API_KEY:
    logging.error("Error: OPENROUTER_API_KEY environment variable not set.")
    sys.exit(1)

# --- Project Paths ---
# Assumes the script is run from the project root or ../src relative to data
PROJECT_ROOT = Path(os.getcwd()) # Adjust if script location changes
DATA_PATH = PROJECT_ROOT / "data"
BENCHMARKS_LIST_PATH = DATA_PATH / "benchmarks.json"

# --- Helper Functions ---

def get_project_paths(benchmark_id):
    """Gets relevant paths for a benchmark."""
    benchmark_dir = DATA_PATH / "benchmarks" / benchmark_id
    info_path = benchmark_dir / "info.json"
    return benchmark_dir, info_path

def load_json_file(path):
    """Loads JSON data from a file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"Error: File not found at {path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {path}")
        return None

def save_json_file(path, data):
    """Saves data to a JSON file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Successfully saved data to {path}")
    except IOError as e:
        logging.error(f"Error saving file {path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during file save: {e}")


def call_jina_reader(url: str, instruction: str = "Extract the main content, focusing on tables or lists containing model names and scores.") -> str | None:
    """Calls the Jina Reader API to get processed content from a URL."""
    logging.info(f"Calling Jina Reader for URL: {url}")
    headers = {
        'Authorization': f'Bearer {JINA_API_KEY}',
        'Accept': 'application/json', # Request JSON instead of stream for easier handling
        'Content-Type': 'application/json',
    }
    data = {
        "url": url,
        "target_selector": "body", # Target the whole body initially
        "include_raw_html": False,
         # Jina Cloud takes an 'reader_options' object now for parameters
        "reader_options": {
            "output_format": "markdown" # Or 'text'
        }
        # Add custom instruction if needed via prompt engineering later
    }

    try:
        response = requests.post(JINA_READER_URL, headers=headers, json=data, timeout=120) # Increased timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        response_data = response.json()
        if isinstance(response_data, list) and len(response_data) > 0:
             content = response_data[0].get('content', '')
             logging.info(f"Jina Reader returned content (length: {len(content)}).")
             return content
        elif isinstance(response_data, dict) and 'data' in response_data and len(response_data['data']) > 0:
             # Handle potential alternative structure { "data": [ { "content": "..." } ] }
             content = response_data['data'][0].get('content', '')
             logging.info(f"Jina Reader returned content (length: {len(content)}).")
             return content
        else:
            logging.warning(f"Jina Reader returned empty or unexpected response format for {url}: {response_data}")
            return None

    except requests.exceptions.RequestException as e:
        logging.error(f"Jina Reader API request failed for {url}: {e}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Failed to decode JSON response from Jina Reader for {url}. Response text: {response.text[:500]}")
        return None

def call_openrouter_llm(content: str, benchmark_info: dict, source_url: str) -> str | None:
    """Calls OpenRouter LLM to extract structured data from content."""
    logging.info(f"Calling OpenRouter LLM ({LLM_MODEL})...")

    # --- Construct a Detailed Prompt ---
    # Extract primary metric/dimension to guide the LLM
    primary_metric = benchmark_info.get('display', {}).get('primary_metric', 'score')
    primary_dimension = benchmark_info.get('display', {}).get('primary_dimension', 'overall')
    benchmark_name = benchmark_info.get('name', 'this benchmark')

    system_prompt = f"""
You are an expert data extraction agent. Your task is to analyze the provided text content from a webpage about the '{benchmark_name}' benchmark ({source_url}) and extract benchmark scores for various models.

Focus on identifying model names and their corresponding scores, specifically the '{primary_metric}' score, ideally within the '{primary_dimension}' dimension if applicable.

Also, try to identify the date the benchmark results were published or updated from the text content. If found, use 'YYYY-MM-DD' format. If no date is explicitly mentioned in the content, state 'null' for the date.

Your output MUST be a single JSON object containing two keys:
1.  `date`: The publication/update date extracted from the text in "YYYY-MM-DD" format, or null if not found.
2.  `scores`: A JSON array, where each element is an object representing a model's score. Each object MUST have the following keys:
    *   `model_id`: A *unique*, *lowercase*, *hyphenated* identifier for the model (e.g., "claude-3-opus", "gpt-4-turbo", "gemini-1.5-pro"). Create a sensible ID if the exact one isn't obvious.
    *   `dimensions`: A JSON object containing the scores for different dimensions. It MUST include at least the primary dimension ('{primary_dimension}') which itself MUST be an object containing the primary metric ('{primary_metric}'). Example: `{{"{primary_dimension}": {{"{primary_metric}": <score_value>}}}}`. Include other dimensions/metrics if readily available in the text. Ensure the score value is a number (int or float), not a string. If a score is explicitly mentioned as unavailable (e.g., "N/A", "-"), represent it as `null`.

Example Output Structure:
{{
  "date": "2024-03-15",
  "scores": [
    {{
      "model_id": "claude-3-opus",
      "dimensions": {{
        "{primary_dimension}": {{ "{primary_metric}": 95.2 }}
      }}
    }},
    {{
      "model_id": "gpt-4-turbo",
      "dimensions": {{
        "{primary_dimension}": {{ "{primary_metric}": 94.8 }},
        "coding": {{ "pass@1": 80.1 }}
      }}
    }},
    {{
      "model_id": "some-other-model",
      "dimensions": {{
         "{primary_dimension}": {{ "{primary_metric}": null }}
      }}
    }}
  ]
}}

If you cannot find any model scores in the provided text, return a JSON object with an empty `scores` array and `null` date: `{{"date": null, "scores": []}}`. Do not add any explanations outside the JSON structure.
"""

    user_prompt = f"Here is the text content extracted from {source_url}:\n\n```text\n{content}\n```\n\nPlease extract the benchmark data according to the instructions and provide the JSON output."

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "response_format": {"type": "json_object"} # Request JSON output if model supports it
    }

    try:
        response = requests.post(OPENROUTER_API_URL, headers=headers, json=data, timeout=180) # Longer timeout for LLM
        response.raise_for_status()
        result = response.json()
        llm_output = result.get('choices', [{}])[0].get('message', {}).get('content')
        if llm_output:
            logging.info("OpenRouter LLM returned a response.")
            # Clean potential markdown code block formatting
            if llm_output.strip().startswith("```json"):
                llm_output = llm_output.strip()[7:-3].strip()
            elif llm_output.strip().startswith("```"):
                 llm_output = llm_output.strip()[3:-3].strip()
            return llm_output
        else:
            logging.warning(f"OpenRouter LLM returned empty content. Response: {result}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"OpenRouter API request failed: {e}")
        # Log request details if possible (without exposing sensitive data in production logs)
        logging.debug(f"Failed request data (partial): {json.dumps(data)[:200]}")
        if e.response is not None:
             logging.error(f"Response status: {e.response.status_code}, Response text: {e.response.text[:500]}")
        return None
    except (KeyError, IndexError) as e:
        logging.error(f"Could not parse OpenRouter response structure: {e}. Response: {result}")
        return None

def parse_llm_output(llm_response_text: str) -> dict | None:
    """Safely parses the JSON string from the LLM response."""
    if not llm_response_text:
        return None
    try:
        parsed_data = json.loads(llm_response_text)
        # Basic validation
        if isinstance(parsed_data, dict) and 'scores' in parsed_data and isinstance(parsed_data['scores'], list):
            # Further validation could be added here (e.g., check score structure)
            logging.info(f"Successfully parsed LLM JSON output. Found {len(parsed_data['scores'])} scores.")
            return parsed_data
        else:
            logging.warning(f"LLM output parsed but doesn't match expected structure (missing 'scores' list?). Output: {llm_response_text[:500]}")
            return None
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from LLM response: {e}. Response text: {llm_response_text[:500]}")
        return None

def validate_date(date_str):
    """Validates if date string is YYYY-MM-DD format."""
    if not date_str:
        return False
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return True
    except ValueError:
        return False

# --- Main Agent Logic ---

def update_benchmark(benchmark_id: str):
    """Main function to update a single benchmark."""
    logging.info(f"--- Starting update process for benchmark: {benchmark_id} ---")

    benchmark_dir, info_path = get_project_paths(benchmark_id)
    if not info_path.exists():
        logging.error(f"Info file not found for benchmark '{benchmark_id}' at {info_path}. Skipping.")
        return False

    benchmark_info = load_json_file(info_path)
    if not benchmark_info:
        return False # Error logged in load_json_file

    source_url = benchmark_info.get('source_url')
    if not source_url:
        logging.error(f"Missing 'source_url' in {info_path}. Skipping.")
        return False

    # 1. Get content using Jina Reader
    jina_content = call_jina_reader(source_url)
    if not jina_content:
        logging.warning(f"Could not retrieve content from {source_url} using Jina Reader. Skipping LLM step.")
        # Consider if you want to proceed without Jina content (e.g., just try LLM on URL - less reliable)
        return False

    # 2. Extract data using LLM via OpenRouter
    llm_output_str = call_openrouter_llm(jina_content, benchmark_info, source_url)
    extracted_data = parse_llm_output(llm_output_str)

    if not extracted_data or not extracted_data.get("scores"):
        logging.warning(f"LLM did not return valid score data for {benchmark_id}. No update possible.")
        return False # No valid data extracted

    # 3. Determine Date
    llm_date_str = extracted_data.get("date")
    if llm_date_str and validate_date(llm_date_str):
        new_date_str = llm_date_str
        logging.info(f"Using date from LLM: {new_date_str}")
    else:
        new_date_str = datetime.now().strftime('%Y-%m-%d')
        logging.info(f"LLM date '{llm_date_str}' invalid or not found. Using current date: {new_date_str}")

    # 4. Check if data is newer
    benchmarks_list = load_json_file(BENCHMARKS_LIST_PATH)
    if not benchmarks_list:
        logging.error("Could not load master benchmarks list. Cannot proceed.")
        return False

    current_last_updated = None
    benchmark_index = -1
    for i, b in enumerate(benchmarks_list):
        if b.get("id") == benchmark_id:
            current_last_updated = b.get("last_updated")
            benchmark_index = i
            break

    if benchmark_index == -1:
         logging.error(f"Benchmark ID '{benchmark_id}' not found in {BENCHMARKS_LIST_PATH}. Cannot update.")
         return False

    if current_last_updated and new_date_str <= current_last_updated:
        logging.info(f"Extracted data date ({new_date_str}) is not newer than current last updated date ({current_last_updated}). No update needed.")
        return True # Considered success as no update was required

    # 5. Prepare and Save New Data
    new_data_file_path = benchmark_dir / f"{new_date_str}.json"
    # Add the source URL and final date to the data to be saved
    output_payload = {
        "date": new_date_str,
        "source_url": source_url,
        "scores": extracted_data["scores"]
    }
    save_json_file(new_data_file_path, output_payload)

    # 6. Update Master Benchmark List
    benchmarks_list[benchmark_index]["last_updated"] = new_date_str
    save_json_file(BENCHMARKS_LIST_PATH, benchmarks_list)

    logging.info(f"--- Successfully updated benchmark: {benchmark_id} with data from {new_date_str} ---")
    return True


# --- Script Execution ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update LLM benchmark data using Jina and OpenRouter.")
    parser.add_argument(
        "--benchmark", "-b",
        required=True,
        help="The ID of the benchmark to update (e.g., 'lmsys')."
    )
    # Potentially add '--all' flag later to iterate through benchmarks.json

    args = parser.parse_args()

    if not (DATA_PATH / "benchmarks" / args.benchmark).exists():
         logging.error(f"Benchmark directory not found: {DATA_PATH / 'benchmarks' / args.benchmark}")
         sys.exit(1)

    success = update_benchmark(args.benchmark)

    if success:
        logging.info(f"Update process completed for benchmark '{args.benchmark}'. Check logs for details.")
        sys.exit(0)
    else:
        logging.error(f"Update process failed for benchmark '{args.benchmark}'. Check logs for details.")
        sys.exit(1)