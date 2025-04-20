# benchmark_update_agent.py
import os
import json
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import requests
import instructor
from pydantic import BaseModel, Field
import dotenv
import re

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Endpoints and Configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

# Define structured data models using Pydantic
class ModelInfo(BaseModel):
    name: str = Field(..., description="The display name of the model")
    organization: str = Field(..., description="The organization that created the model")
    release_date: str = Field(..., description="The release date of the model in YYYY-MM-DD format")
    license: str = Field(..., description="The license of the model (e.g., 'Proprietary', 'Apache 2.0', etc.)")


class BenchmarkScore(BaseModel):
    model_id: str = Field(..., description="The ID of the model (e.g., 'claude-3-7-sonnet')")
    dimensions: Dict[str, Dict[str, Any]] = Field(
        ..., 
        description="Scores for different dimensions, with metrics inside each dimension"
    )


class BenchmarkData(BaseModel):
    date: str = Field(..., description="The date of the benchmark data in YYYY-MM-DD format")
    source_url: str = Field(..., description="The source URL of the benchmark data")
    scores: List[BenchmarkScore] = Field(..., description="The scores for different models")



class ProjectPaths:
    def __init__(self, root_dir: str = None):
        if root_dir:
            self.root = Path(root_dir)
        else:
            # Assume this script is in the project root
            self.root = Path(__file__).parent
        
        self.data = self.root / "data"
        self.models_json = self.data / "models.json"
        self.benchmarks_json = self.data / "benchmarks.json"
        self.benchmarks_dir = self.data / "benchmarks"
        self.output_dir = self.root / "output"
        self.output_dir.mkdir(exist_ok=True)
        
        # File to log unknown models
        self.unknown_models_log = self.output_dir / "benchmark_models.log"


import asyncio
from crawl4ai import AsyncWebCrawler

async def crawl4ai_fetch(url: str) -> str | None:
    """
    Fetches content from a URL using crawl4ai library
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        str: The markdown content, or None if fetching failed
    """
    try:
        async with AsyncWebCrawler() as crawler:
            result = await crawler.arun(url=url)
            if result and hasattr(result, 'markdown'):
                return result.markdown
            return None
    except Exception as e:
        logging.error(f"Error fetching content with crawl4ai: {e}")
        return None

def call_crawl4ai(url: str) -> str | None:
    """
    Synchronous wrapper for the async crawl4ai fetcher
    
    Args:
        url: The URL to fetch content from
        
    Returns:
        str: The markdown content, or None if fetching failed
    """
    try:
        return asyncio.run(crawl4ai_fetch(url))
    except Exception as e:
        logging.error(f"Error running crawl4ai: {e}")
        return None
    
class BenchmarkUpdateAgent:
    def __init__(self, root_dir: str = None):
        self.paths = ProjectPaths(root_dir)
        self.models = self._load_models()
        self.benchmarks = self._load_benchmarks()
        

        if not OPENROUTER_API_KEY:
            logging.warning("OPENROUTER_API_KEY is not set. LLM extraction will not work.")
        
        # Set up OpenRouter client with instructor
        try:
            from openai import OpenAI
            
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
            
            # Enable instructor patches for OpenAI client with OpenRouter mode
            self.client = instructor.from_openai(
                client, mode=instructor.Mode.OPENROUTER_STRUCTURED_OUTPUTS
            )
            logging.info("OpenRouter client initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize OpenRouter client: {e}")
            self.client = None
    
    def _load_models(self) -> Dict[str, Any]:
        """Load existing models from models.json"""
        if self.paths.models_json.exists():
            with open(self.paths.models_json, "r") as f:
                return json.load(f)
        return {}
    
    def _load_benchmarks(self) -> List[Dict[str, Any]]:
        """Load benchmark metadata from benchmarks.json"""
        if self.paths.benchmarks_json.exists():
            with open(self.paths.benchmarks_json, "r") as f:
                return json.load(f)
        return []
    
    def _get_benchmark_by_id(self, benchmark_id: str) -> Optional[Dict[str, Any]]:
        """Find benchmark metadata by ID"""
        for benchmark in self.benchmarks:
            if benchmark["id"] == benchmark_id:
                return benchmark
        return None

    
    def validate_date(self, date_str):
        """Validates if date string is YYYY-MM-DD format."""
        if not date_str:
            return False
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def update_benchmark(self, benchmark_id: str, url: Optional[str] = None) -> bool:
        """
        Update benchmark data for the specified benchmark ID.
        
        Args:
            benchmark_id: ID of the benchmark to update
            url: Optional URL to fetch data from (otherwise uses the one in benchmarks.json)
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        benchmark = self._get_benchmark_by_id(benchmark_id)
        if not benchmark:
            logging.error(f"Benchmark with ID '{benchmark_id}' not found")
            return False
        
        source_url = url or benchmark["source_url"]
        logging.info(f"Updating benchmark '{benchmark_id}' from {source_url}")
        
        # Create benchmark directory if it doesn't exist
        benchmark_dir = self.paths.benchmarks_dir / benchmark_id
        benchmark_dir.mkdir(exist_ok=True, parents=True)
        
        # Get today's date for the new data file
        today = datetime.now().strftime("%Y-%m-%d")
        
        try:
            benchmark_data = self._extract_benchmark_data(benchmark_id, source_url)
            
            # Save the new benchmark data
            output_file = benchmark_dir / f"{today}.json"
            with open(output_file, "w") as f:
                json.dump(benchmark_data.model_dump(), f, indent=2)
            
            # Update the last_updated field in benchmarks.json
            benchmark["last_updated"] = today
            with open(self.paths.benchmarks_json, "w") as f:
                json.dump(self.benchmarks, f, indent=2)
            
            # Check for unknown models and log them
            self._check_for_unknown_models(benchmark_data)
            
            logging.info(f"Successfully updated benchmark '{benchmark_id}'. Data saved to {output_file}")
            return True
            
        except Exception as e:
            logging.error(f"Error updating benchmark '{benchmark_id}': {e}")
            import IPython; IPython.embed()
            return False

    def _extract_benchmark_data(self, benchmark_id: str, source_url: str) -> BenchmarkData:
        """
        Extract benchmark data from the source URL using crawl4ai and instructor.
        
        This method:
        1. Uses crawl4ai to extract content from the benchmark website
        2. Falls back to regular request if crawl4ai fails
        3. Uses instructor with OpenRouter to parse the content into structured data
        4. Returns a BenchmarkData object with the extracted information
        """
        # Ensure output directory exists for debugging
        debug_dir = self.paths.output_dir / "debug"
        debug_dir.mkdir(exist_ok=True, parents=True)
        
        debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Load the benchmark info.json to get metrics, dimensions, and primary settings
        benchmark_info_path = self.paths.benchmarks_dir / benchmark_id / "info.json"
        if not benchmark_info_path.exists():
            raise Exception(f"Benchmark info file not found for {benchmark_id}")
        
        with open(benchmark_info_path, "r") as f:
            benchmark_info = json.load(f)
        
        # Extract key information from benchmark_info
        metrics = benchmark_info.get("metrics", [])
        dimensions = benchmark_info.get("dimensions", [])
        primary_metric = benchmark_info.get("display", {}).get("primary_metric")
        primary_dimension = benchmark_info.get("display", {}).get("primary_dimension", "overall")
        
        if not primary_metric:
            raise Exception(f"Primary metric not defined for benchmark {benchmark_id}")
        
        # 1. Get content using crawl4ai
        logging.info(f"Fetching content from {source_url} using crawl4ai")
        content = call_crawl4ai(source_url)
        
        # Save crawl4ai content for debugging
        if content:
            with open(debug_dir / f"{benchmark_id}_{debug_timestamp}_crawl4ai_content.txt", "w") as f:
                f.write(content)  # Save first 50k chars to avoid excessively large files
            logging.info(f"Crawl4ai returned content (length: {len(content)})")
        
        # Fall back to regular request if crawl4ai fails
        if not content:
            logging.warning(f"Crawl4ai failed for {source_url}, falling back to regular request")
            try:
                response = requests.get(source_url)
                if response.status_code != 200:
                    raise Exception(f"Failed to fetch data from {source_url}. Status code: {response.status_code}")
                content = response.text
                
                # Save HTTP content for debugging
                with open(debug_dir / f"{benchmark_id}_{debug_timestamp}_http_content.txt", "w") as f:
                    f.write(content)  # Save first 50k chars
                    
            except Exception as e:
                logging.error(f"HTTP request failed: {e}")
                raise Exception(f"Failed to fetch data from {source_url} using all available methods")
        
        # 2. Prepare a much clearer prompt that emphasizes the exact structure required
        # Provide known model IDs to the LLM so it can reuse exact formatting
        known_model_ids = list(self.models.keys())
        known_model_ids.sort()
        known_model_ids_str = ", ".join(known_model_ids)
        
        system_prompt = f"""
        Your primary task is to extract benchmark data from the {benchmark_id} leaderboard into a structured JSON format.

        THE RESPONSE FORMAT IS CRITICAL. Your output MUST follow this exact structure:
        {{
        "date": "YYYY-MM-DD",
        "source_url": "URL of the benchmark",
        "scores": [
            {{
            "model_id": "normalized-model-identifier-1",
            "dimensions": {{
                "{primary_dimension}": {{
                "{primary_metric}": numeric_score_value
                }}
            }}
            }},
            {{
            "model_id": "normalized-model-identifier-2",
            "dimensions": {{
                "{primary_dimension}": {{
                "{primary_metric}": numeric_score_value
                }}
            }}
            }}
            // ... more scores ...
        ]
        }}

        IMPORTANT NOTES ON STRUCTURE:
        1. Each item in "scores" MUST be a complete object with "model_id" and "dimensions" fields.
        2. Inside dimensions, include the primary dimension "{primary_dimension}" with the primary metric "{primary_metric}".
        3. Score values should be numeric (not strings).
        4. The "model_id" field MUST contain the model identifier resulting from applying the normalization rules below.

        ---
        MODEL ID NORMALIZATION:

        The source benchmark might use different names for models. When extracting, normalize the model names you find in the text to create a standardized 'model_id' for the output JSON.

        Consider the following list as examples of typical canonical IDs, but your output 'model_id' should be derived from the benchmark name using the rules below, even if it's not on this list initially:
        {known_model_ids_str}

        NORMALIZATION RULES:
        1.  **Specific Mappings (Apply First):** If a model name exactly matches or is a known alias for one of these specific cases, use the provided canonical ID:
            *   If the text is 'o4-mini (high)', use 'o4-mini-high'.
            *   If the text is 'o3 (high)', use 'o3-high'.
            *   If the text is 'o3-mini (high)', use 'o3-mini-high'.
            *   If the text is 'DeepSeek Chat V3 (prev)', use 'deepseek-v3'.

        2.  **General Cleaning (Apply if no specific mapping):** If the name doesn't match any specific mapping above, apply these general cleaning steps to derive the 'model_id':
            *   Convert the name to lowercase.
            *   Replace spaces, periods (`.`), and underscores (`_`) with hyphens (`-`).
            *   Remove content within parentheses (e.g., '(high)', '(low)', '(prev)', '(date)').
            *   Remove leading/trailing hyphens.

        The current date is {datetime.now().strftime('%Y-%m-%d')}.
        """

        user_prompt = f"Extract the benchmark data from this page, applying the normalization rules for model IDs and skipping combination entries. Pay special attention to model names and their scores for {primary_metric} in the {primary_dimension} dimension. Remember that each item in the scores array MUST be a complete object with model_id (normalized using the rules) and dimensions fields:\n\n{content}..."

        # Save prompts for debugging
        with open(debug_dir / f"{benchmark_id}_{debug_timestamp}_system_prompt.txt", "w") as f:
            f.write(system_prompt)
            
        with open(debug_dir / f"{benchmark_id}_{debug_timestamp}_user_prompt.txt", "w") as f:
            f.write(user_prompt)  # Save first 10k chars
        
        # 3. Call the LLM first to get the raw JSON response
        model_to_use = "google/gemini-2.5-flash-preview"  # You can adjust the model as needed
        
        try:
            # Create a non-instructor client to get raw response
            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.getenv("OPENROUTER_API_KEY"),
            )
            
            logging.info(f"Getting raw JSON response from {model_to_use}")
            
            raw_response = client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            raw_json = raw_response.choices[0].message.content if raw_response.choices else "{}"
            
            # Save the raw JSON response for debugging
            with open(debug_dir / f"{benchmark_id}_{debug_timestamp}_raw_response.json", "w") as f:
                f.write(raw_json)
                
            logging.info(f"Saved raw response to {debug_dir / f'{benchmark_id}_{debug_timestamp}_raw_response.json'}")
            
            # Parse the raw JSON response manually instead of using instructor
            try:
                data = json.loads(raw_json)
                
                # Validate basic structure
                if not isinstance(data, dict):
                    raise ValueError("Raw JSON is not a dictionary")
                    
                if "scores" not in data:
                    raise ValueError("Raw JSON has no 'scores' key")
                
                if not isinstance(data["scores"], list):
                    raise ValueError("'scores' is not a list")
                
                # Fix the scores structure if needed
                fixed_scores = []
                for item in data["scores"]:
                    if isinstance(item, str):
                        # Convert string to proper structure
                        fixed_scores.append({
                            "model_id": item,
                            "dimensions": {
                                primary_dimension: {
                                    primary_metric: None  # We don't know the score
                                }
                            }
                        })
                    elif isinstance(item, dict) and "model_id" in item and "dimensions" in item:
                        # Already in correct format
                        fixed_scores.append(item)
                    elif isinstance(item, dict) and "model_id" in item:
                        # Has model_id but missing dimensions
                        fixed_scores.append({
                            "model_id": item["model_id"],
                            "dimensions": {
                                primary_dimension: {
                                    primary_metric: None
                                }
                            }
                        })
                
                # Create BenchmarkData with fixed scores
                return BenchmarkData(
                    date=data.get("date", datetime.now().strftime("%Y-%m-%d")),
                    source_url=data.get("source_url", source_url),
                    scores=fixed_scores
                )
                
            except json.JSONDecodeError as e:
                logging.error(f"Failed to parse raw JSON response: {e}")
                raise ValueError(f"Failed to parse LLM response as JSON: {e}")
                
        except Exception as e:
            error_message = str(e)
            logging.error(f"Error extracting data with {model_to_use}: {error_message}")
            
            # Save the error message
            with open(debug_dir / f"{benchmark_id}_{debug_timestamp}_error.txt", "w") as f:
                f.write(error_message)
                
            # Re-raise the exception
            raise
    
    def _check_for_unknown_models(self, benchmark_data: BenchmarkData):
        """Check for models in the benchmark data that aren't in models.json and log them"""
        unknown_models = []
        
        for score in benchmark_data.scores:
            model_id = score.model_id
            if model_id not in self.models:
                unknown_models.append(model_id)
                
                # Extract model info using instructor
                model_info =  None # self._extract_model_info(model_id)
                if model_info:
                    # Log the unknown model
                    with open(self.paths.unknown_models_log, "a") as f:
                        f.write(f"Date: {datetime.now().isoformat()}\n")
                        f.write(f"Unknown model: {model_id}\n")
                        f.write(f"Extracted info: {json.dumps(model_info.model_dump(), indent=2)}\n")
                        f.write("-" * 50 + "\n")
        
        if unknown_models:
            logging.info(f"Found {len(unknown_models)} unknown models: {', '.join(unknown_models)}")
            logging.info(f"Details have been logged to {self.paths.unknown_models_log}")
    
    def _extract_model_info(self, model_id: str) -> Optional[ModelInfo]:
        """
        Extract model information using instructor with OpenRouter.
        
        This method tries to find detailed information about an unknown model
        using the model_id as a reference.
        """
        try:
            # For demonstration purposes, we'll use a detailed prompt
            # that guides the extraction process
            system_prompt = """
            Extract factual information about an AI language model.
            
            IMPORTANT GUIDELINES:
            1. Only extract factual, verifiable information
            2. If the exact information isn't known, indicate this with "Unknown" rather than guessing
            3. For release dates, use YYYY-MM-DD format if full date is known, or YYYY-MM if only month is known, or YYYY if only year is known
            4. For organization, provide the full company/lab name, not abbreviations
            5. For license, specify the exact license type if known (e.g., "Apache 2.0", "Proprietary", "MIT", etc.)
            
            If you're unsure about any field, it's better to mark it as "Unknown" than to provide potentially incorrect information.
            """
            
            user_prompt = f"""
            Extract structured information about the AI model with ID '{model_id}'.
            
            Required fields:
            - name: The display name of the model (if different from the ID)
            - organization: The company, lab, or entity that created the model
            - release_date: When the model was released (in YYYY-MM-DD format if possible)
            - license: The license type of the model
            
            If you can't find specific information for any field, use "Unknown" as the value.
            """
            
            model_to_use = "google/gemini-2.5-flash-preview"  # You can adjust the model as needed
            
            model_info = self.client.chat.completions.create(
                model=model_to_use,
                response_model=ModelInfo,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                extra_body={"provider": {"require_parameters": True}}
            )
            return model_info
        except Exception as e:
            logging.error(f"Error extracting info for model '{model_id}': {e}")
            return None

    def update_all_benchmarks(self) -> Dict[str, bool]:
        """Update all benchmarks and return their status"""
        results = {}
        for benchmark in self.benchmarks:
            benchmark_id = benchmark["id"]
            results[benchmark_id] = self.update_benchmark(benchmark_id)
        return results


def main():
    parser = argparse.ArgumentParser(description="Update LLM benchmark data")
    parser.add_argument("--benchmark", "-b", help="Benchmark ID to update (default: update all)", default="all")
    parser.add_argument("--url", help="Override the source URL for the benchmark")
    parser.add_argument("--root-dir", help="Root directory of the project")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO", 
                        help="Set logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # Set logging level based on argument
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Check if required environment variables are set

    if not OPENROUTER_API_KEY:
        logging.error("OPENROUTER_API_KEY environment variable not set. Add it to your .env file.")
        sys.exit(1)
    
    try:
        agent = BenchmarkUpdateAgent(args.root_dir)
        
        if args.benchmark.lower() == "all":
            results = agent.update_all_benchmarks()
            successful = sum(1 for status in results.values() if status)
            logging.info(f"Updated {successful}/{len(results)} benchmarks successfully")
            sys.exit(0 if successful > 0 else 1)
        else:
            success = agent.update_benchmark(args.benchmark, args.url)
            logging.info(f"Benchmark update {'successful' if success else 'failed'}")
            sys.exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Unhandled exception: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()