import os
import json
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple # Added Tuple
import asyncio
import requests # Added for fallback HTTP request

from openai import OpenAI
import instructor
from pydantic import BaseModel, Field, ValidationError, TypeAdapter
import dotenv
# Removed re as it wasn't actively used after prompt refinement

# Import crawl4ai safely
try:
    from crawl4ai import AsyncWebCrawler
except ImportError:
    AsyncWebCrawler = None
    logging.warning("crawl4ai not installed. Web fetching will rely on basic requests.")


# Import the classification module and its models
from .model_classify import ModelClassifier, ModelClassificationResult, ClassificationStatus, CanonicalModel

# Load environment variables
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# API Endpoints and Configuration
OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')

# --- Define Structured Data Models using Pydantic ---

# Model for Model Info
class ModelInfo(BaseModel):
    name: str = Field(..., description="The display name of the model")
    organization: str = Field(..., description="The organization that created the model")
    release_date: str = Field(..., description="The release date of the model in YYYY-MM-DD format")
    license: str = Field(..., description="The license of the model (e.g., 'Proprietary', 'Apache 2.0', etc.)")

# Model for a single Benchmark Score
class BenchmarkScore(BaseModel):
    model_id: str = Field(..., description="The ID of the model (e.g., 'claude-3-5-sonnet')") # This will be updated
    dimensions: Dict[str, Dict[str, Any]] = Field(
        ...,
        description="Scores for different dimensions, with metrics inside each dimension"
    )

# Model for the full Benchmark Data structure
class BenchmarkData(BaseModel):
    date: str = Field(..., description="The date of the benchmark data in YYYY-MM-DD format")
    source_url: str = Field(..., description="The source URL of the benchmark data")
    scores: List[BenchmarkScore] = Field(..., description="The scores for different models")


class ProjectPaths:
    def __init__(self, root_dir: str = None):
        if root_dir:
            self.root = Path(root_dir).resolve() # Use resolve for absolute path
        else:
            # Assume this script is in src/llmbench, go up two levels
            self.root = Path(__file__).parent.parent.parent.resolve()

        self.data = self.root / "data"
        self.models_json = self.data / "models.json"
        self.benchmarks_json = self.data / "benchmarks.json"
        self.benchmarks_dir = self.data / "benchmarks"
        self.output_dir = self.root / "output"
        self.output_dir.mkdir(exist_ok=True)
        (self.output_dir / "debug").mkdir(exist_ok=True) # Ensure debug subdir exists

        # File to log classification reports
        self.model_classification_log = self.output_dir / "benchmark_model_classification.log"
        # File to save raw LLM responses that cause validation errors (overwrite per attempt)
        self.raw_llm_response_debug = self.output_dir / "debug" / "raw_llm_response_debug.json" # Place in debug

        # Ensure base data directory exists
        self.data.mkdir(exist_ok=True)
        self.benchmarks_dir.mkdir(exist_ok=True)

        # Touch log file
        try:
            if not self.model_classification_log.exists(): self.model_classification_log.touch()
        except OSError as e:
             logging.error(f"Failed to touch log file {self.model_classification_log}: {e}")


async def crawl4ai_fetch(url: str) -> str | None:
    """
    Fetches content from a URL using crawl4ai library

    Args:
        url: The URL to fetch content from

    Returns:
        str: The markdown content, or None if fetching failed
    """
    if AsyncWebCrawler is None:
        logging.warning("Attempted to use crawl4ai_fetch, but crawl4ai is not installed.")
        return None
    try:
        # Use a longer timeout if needed, default is 30s
        async with AsyncWebCrawler(parser_config={'parsing_method':'beautifulsoup'}) as crawler: # Explicitly using BS4 might be more robust sometimes
            result = await crawler.arun(url=url)
            if result and hasattr(result, 'markdown'):
                return result.markdown
            logging.warning(f"crawl4ai completed for {url} but returned no markdown content.")
            return None
    except Exception as e:
        # Don't log full traceback here unless debugging, keep it concise
        logging.error(f"Error fetching content with crawl4ai from {url}: {e}")
        return None

def call_crawl4ai(url: str) -> str | None:
    """
    Synchronous wrapper for the async crawl4ai fetcher

    Args:
        url: The URL to fetch content from

    Returns:
        str: The markdown content, or None if fetching failed
    """
    if AsyncWebCrawler is None: return None # Early exit if not installed
    try:
        # Check if an event loop is already running (e.g., in Jupyter)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            # If in an async context already, schedule it properly
            # This might require more complex handling depending on the outer context
            # For a simple script, running directly might be okay, but let's log a warning
            logging.warning("call_crawl4ai invoked within an existing event loop. Attempting direct await.")
            # This might fail if called from a sync function within an async context
            # A more robust solution would involve asyncio.run_coroutine_threadsafe if needed
            # For now, we proceed with asyncio.run which might raise errors in some environments
            return asyncio.run(crawl4ai_fetch(url))
        else:
            # If no loop running, use asyncio.run
            return asyncio.run(crawl4ai_fetch(url))
    except Exception as e:
        logging.error(f"Error running crawl4ai fetch task for {url}: {e}")
        return None


class BenchmarkUpdateAgent:
    def __init__(self, root_dir: str = None):
        self.paths = ProjectPaths(root_dir)
        logging.info(f"Project root resolved to: {self.paths.root}")
        logging.info(f"Data directory: {self.paths.data}")

        self.models = self._load_models()
        self.benchmarks = self._load_benchmarks()

        self.client = None # Instructor-patched client
        self.model_classifier = None
        self.base_openai_client = None # Base client for raw calls

        if not OPENROUTER_API_KEY:
            logging.error("OPENROUTER_API_KEY is not set. LLM functionality disabled.")
            return

        try:
            self.base_openai_client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
                timeout=120.0, # Increased timeout further
            )
            self.client = instructor.from_openai(
                self.base_openai_client, mode=instructor.Mode.TOOLS # Use TOOLS mode for broader compatibility including Azure
            )
            logging.info("OpenRouter client initialized successfully with Instructor (Mode: TOOLS)")

            if self.models is not None: # Only initialize classifier if models loaded ok
                self.model_classifier = ModelClassifier(client=self.client, known_models=self.models)
            else:
                logging.error("Models data failed to load. ModelClassifier cannot be initialized.")
                self.model_classifier = None # Ensure it's None

        except Exception as e:
            logging.error(f"Failed to initialize OpenRouter client or ModelClassifier: {e}")
            self.client = None
            self.model_classifier = None
            self.base_openai_client = None


    def _load_models(self) -> Optional[Dict[str, Any]]:
        """Load existing models from models.json"""
        if not self.paths.models_json.exists():
             logging.warning(f"'{self.paths.models_json}' not found. Starting with empty model list.")
             return {} # Return empty dict if file not found
        try:
            with open(self.paths.models_json, "r", encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {self.paths.models_json}: {e}. Cannot load models.")
            return None # Signal failure
        except Exception as e:
            logging.error(f"Unexpected error loading {self.paths.models_json}: {e}. Cannot load models.")
            return None # Signal failure


    def _load_benchmarks(self) -> List[Dict[str, Any]]:
        """Load benchmark metadata from benchmarks.json"""
        if not self.paths.benchmarks_json.exists():
             logging.error(f"'{self.paths.benchmarks_json}' not found. Cannot update any benchmarks.")
             return []
        try:
            with open(self.paths.benchmarks_json, "r", encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logging.error(f"Error decoding JSON from {self.paths.benchmarks_json}: {e}")
            return []
        except Exception as e:
            logging.error(f"Unexpected error loading {self.paths.benchmarks_json}: {e}")
            return []

    def _save_models(self) -> bool:
        """Saves the current self.models dictionary to models.json"""
        if self.models is None:
             logging.error("Cannot save models, self.models is None (likely due to loading error).")
             return False
        try:
            with open(self.paths.models_json, "w", encoding='utf-8') as f:
                json.dump(self.models, f, indent=2, sort_keys=True) # Sort keys for consistency
            logging.info(f"Successfully saved updated models to {self.paths.models_json}")
            return True
        except Exception as e:
            logging.error(f"Error saving models to {self.paths.models_json}: {e}")
            return False

    def _get_benchmark_by_id(self, benchmark_id: str) -> Optional[Dict[str, Any]]:
        """Find benchmark metadata by ID"""
        for benchmark in self.benchmarks:
            if benchmark["id"] == benchmark_id:
                return benchmark
        return None


    def update_benchmark(self, benchmark_id: str, url: Optional[str] = None) -> bool:
        """
        Update benchmark data for the specified benchmark ID.
        Returns True on success, False on failure of a critical step.
        """
        # --- Pre-checks ---
        if not self.base_openai_client or not self.client or not self.model_classifier:
             logging.error(f"LLM client or Model Classifier not initialized. Cannot update benchmark '{benchmark_id}'.")
             return False
        if self.models is None:
             logging.error(f"Models data not loaded. Cannot update benchmark '{benchmark_id}'.")
             return False

        benchmark = self._get_benchmark_by_id(benchmark_id)
        if not benchmark:
            logging.error(f"Benchmark with ID '{benchmark_id}' not found.")
            return False

        source_url = url or benchmark.get("source_url") # Use .get for safety
        if not source_url:
             logging.error(f"No source_url for benchmark '{benchmark_id}'.")
             return False

        logging.info(f"Starting update process for benchmark '{benchmark_id}' from {source_url}")

        benchmark_dir = self.paths.benchmarks_dir / benchmark_id
        benchmark_dir.mkdir(exist_ok=True, parents=True)
        today = datetime.now().strftime("%Y-%m-%d")

        # Load benchmark info for primary metric/dimension
        benchmark_info_path = benchmark_dir / "info.json"
        if not benchmark_info_path.exists():
            logging.error(f"Benchmark info file not found: {benchmark_info_path}. Cannot proceed.")
            return False
        try:
             with open(benchmark_info_path, "r", encoding='utf-8') as f:
                 benchmark_info = json.load(f)
             primary_metric = benchmark_info.get("display", {}).get("primary_metric")
             primary_dimension = benchmark_info.get("display", {}).get("primary_dimension", "overall")
             if not primary_metric:
                 logging.error(f"Primary metric not defined in {benchmark_info_path}.")
                 return False
        except Exception as e:
             logging.error(f"Error loading benchmark info {benchmark_info_path}: {e}")
             return False

        models_were_updated = False # Flag to track if self.models changed
        id_mapping: Dict[str, str] = {} # Store original_id -> canonical_id

        try:
            # --- Step 1: Extract Data (using base client, with validation/correction) ---
            extracted_data = self._extract_benchmark_data(
                benchmark_id, source_url, primary_dimension, primary_metric, self.base_openai_client
            )
            if not extracted_data:
                # Error logged within _extract_benchmark_data
                logging.error(f"Failed to extract benchmark data for '{benchmark_id}'. Aborting update.")
                return False # Critical step failed

            # --- Step 2: Classify Models & Update self.models if new ones found ---
            # This now returns the mapping needed to update the benchmark data
            models_were_updated, id_mapping = self._classify_and_process_models(
                extracted_data, benchmark_id
            )
            # Error handling for classification failure is inside _classify_and_process_models
            # It might raise an exception, which will be caught by the outer try-except block.

            # --- Step 3: Update Benchmark Data with Canonical IDs ---
            logging.info(f"Updating benchmark scores with canonical model IDs for '{benchmark_id}'...")
            updated_scores_count = 0
            original_ids_kept_count = 0
            if extracted_data and extracted_data.scores: # Check if scores exist
                for score in extracted_data.scores:
                    original_id = score.model_id
                    canonical_id = id_mapping.get(original_id)
                    if canonical_id and canonical_id != original_id:
                        logging.debug(f"Mapping '{original_id}' -> '{canonical_id}' for benchmark '{benchmark_id}'")
                        score.model_id = canonical_id
                        updated_scores_count += 1
                    else:
                        # Keep original ID if no mapping found or mapping is the same
                        original_ids_kept_count += 1
                        if not canonical_id:
                           logging.debug(f"No canonical ID found for '{original_id}', keeping original.")
                logging.info(f"Applied canonical IDs: {updated_scores_count} updated, {original_ids_kept_count} kept original for '{benchmark_id}'.")
            else:
                logging.warning(f"No scores found in extracted data for '{benchmark_id}' to update IDs.")


            # --- Step 4: Save *Updated* Extracted Data ---
            output_file = benchmark_dir / f"{today}.json"
            if extracted_data: # Ensure data exists before trying to save
                try:
                    with open(output_file, "w", encoding='utf-8') as f:
                        f.write(extracted_data.model_dump_json(indent=2))
                    logging.info(f"Successfully saved updated benchmark data to {output_file}")
                except Exception as e:
                     logging.error(f"Error saving updated benchmark data to {output_file}: {e}")
                     # Consider this critical? If saving fails, the update isn't really complete.
                     return False
            else:
                # This case should ideally be caught earlier, but double-check
                logging.error(f"Extracted data object is missing before saving for '{benchmark_id}'.")
                return False # This indicates a problem upstream

            # --- Step 5: Save Updated Models (only if changed) ---
            if models_were_updated:
                if not self._save_models():
                    # Log error, but maybe continue? Depends on requirements.
                    # Let's consider failing the update if saving models fails.
                    logging.error(f"Failed to save updated models file for benchmark '{benchmark_id}'.")
                    # return False # Uncomment if this should be a critical failure

            # --- Step 6: Update Benchmark Metadata (last_updated) ---
            success_meta_update = False
            for i, b in enumerate(self.benchmarks):
                if b["id"] == benchmark_id:
                    self.benchmarks[i]["last_updated"] = today
                    success_meta_update = True
                    break
            if success_meta_update:
                try:
                    with open(self.paths.benchmarks_json, "w", encoding='utf-8') as f:
                        json.dump(self.benchmarks, f, indent=2, sort_keys=True)
                    logging.info(f"Updated last_updated timestamp in {self.paths.benchmarks_json}")
                except Exception as e:
                     logging.error(f"Error saving updated {self.paths.benchmarks_json}: {e}")
                     # Log error, but update is considered successful overall if we got here
            else:
                 logging.warning(f"Benchmark ID '{benchmark_id}' not found in self.benchmarks list during metadata update.")


            logging.info(f"Benchmark update process for '{benchmark_id}' completed successfully.")
            return True # Indicate overall success for this benchmark

        except Exception as e:
            # Catch errors from extraction, classification (if re-raised), saving, etc.
            # Log concisely as requested, don't include traceback by default here.
            logging.error(f"Benchmark update failed for '{benchmark_id}': {type(e).__name__} - {e}")
            # Optionally log traceback for easier debugging during development
            # logging.exception(f"Full traceback for failure in benchmark '{benchmark_id}':")
            return False # Indicate failure for this specific benchmark


    def _extract_benchmark_data(
        self,
        benchmark_id: str,
        source_url: str,
        primary_dimension: str,
        primary_metric: str,
        openai_client: OpenAI # Expecting the base client
    ) -> Optional[BenchmarkData]:
        """
        Extracts benchmark data using raw LLM call, validates, attempts correction.
        Returns BenchmarkData object on success, None on failure.
        """
        if not openai_client:
             logging.error("Base OpenAI client is required for extraction.")
             return None

        debug_dir = self.paths.output_dir / "debug"
        debug_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        extraction_debug_filepath = debug_dir / f"{benchmark_id}_{debug_timestamp}_extraction_debug.log"

        # 1. Get content
        logging.info(f"Fetching content from {source_url}")
        content = call_crawl4ai(source_url)

        content_save_path_md = debug_dir / f"{benchmark_id}_{debug_timestamp}_crawl4ai_content.md"
        content_save_path_html = debug_dir / f"{benchmark_id}_{debug_timestamp}_http_content.html"

        if content:
            logging.info(f"Crawl4ai returned content (length: {len(content)}).")
            self._log_extraction_debug(extraction_debug_filepath, f"Crawl4ai returned content (length: {len(content)}).")
            try:
                 with open(content_save_path_md, "w", encoding='utf-8') as f: f.write(content)
            except Exception as save_err: logging.warning(f"Failed to save crawl4ai content: {save_err}")
        else:
            logging.warning(f"Crawl4ai failed for {source_url}. Falling back to HTTP request.")
            self._log_extraction_debug(extraction_debug_filepath, "Crawl4ai failed, falling back to HTTP request.")
            try:
                response = requests.get(source_url, timeout=60, headers={'User-Agent': 'Mozilla/5.0'}) # Add User-Agent
                response.raise_for_status()
                content = response.text
                logging.info(f"HTTP request successful (length: {len(content)}).")
                self._log_extraction_debug(extraction_debug_filepath, f"HTTP request successful (length: {len(content)}).")
                try:
                     with open(content_save_path_html, "w", encoding='utf-8') as f: f.write(content)
                except Exception as save_err: logging.warning(f"Failed to save HTTP content: {save_err}")
            except requests.exceptions.RequestException as e:
                logging.error(f"HTTP request failed for {source_url}: {e}")
                self._log_extraction_debug(extraction_debug_filepath, f"HTTP request failed: {e}")
                return None

        if not content:
             logging.error(f"Could not retrieve content from {source_url}.")
             self._log_extraction_debug(extraction_debug_filepath, "Failed to retrieve content.")
             return None

        # 2. Prepare prompt for raw JSON extraction
        # (Keep the prompt focused on getting the data out, including the original ID)
        system_prompt = f"""
        You are an expert data extraction bot. Extract benchmark data from the provided text into a VALID JSON object.
        Output ONLY the JSON object, nothing else.
        The JSON object MUST conform to this structure:
        {{
          "date": "YYYY-MM-DD", // Date data was published, or today's date ({datetime.now().strftime('%Y-%m-%d')})
          "source_url": "{source_url}",
          "scores": [
            {{
              "model_id": "model-id-extracted-from-text", // Extract the ID as faithfully as possible from the text, using the rules below.
              "dimensions": {{
                "{primary_dimension}": {{
                  "{primary_metric}": numerical_score // Use null if score not found/numeric
                  // Optionally include other dimensions/metrics if clearly available
                }}
              }}
            }}
            // ... include ALL models found ...
          ]
        }}

        EXTRACTION RULES for 'model_id':
        - Derive from the name in the text.
        - Convert to lowercase.
        - Replace spaces, periods, underscores, parentheses with hyphens.
        - REMOVE common trailing suffixes like -preview, -latest, -beta, -v1, -v2, and date suffixes (e.g., -2024-07-18).
        - KEEP core identifiers like -mini, -high, -instruct, -chat, -sonnet, -haiku, -opus.
        - Examples: "GPT-4.5 Preview 2025" -> "gpt-4-5", "Claude 3 Opus" -> "claude-3-opus", "o4-mini (high)" -> "o4-mini-high", "DeepSeek Chat V3 (prev)" -> "deepseek-v3".
        - The ID should be derived from the text, NOT from a predefined list.
        - SKIP entries combining multiple models (e.g., "Model A + Model B").
        - Be precise. If the text says "Claude 3.5 Sonnet", the ID should be "claude-3-5-sonnet". If it says "GPT-4o", it should be "gpt-4o".

        Focus on the benchmark '{benchmark_id}'. Extract the primary metric '{primary_metric}' within the '{primary_dimension}' dimension.
        """
        max_content_length = 100000 # Truncate long content to avoid excessive token usage
        user_prompt = f"Extract benchmark data from the following text, adhering STRICTLY to the JSON structure and rules provided in the system prompt:\n\n{content[:min(len(content), max_content_length)]}"
        if len(content) > max_content_length: user_prompt += "\n...[content truncated]"

        # Save prompts
        prompt_save_path_sys = debug_dir / f"{benchmark_id}_{debug_timestamp}_extraction_system_prompt.txt"
        prompt_save_path_user = debug_dir / f"{benchmark_id}_{debug_timestamp}_extraction_user_prompt.txt"
        try:
            with open(prompt_save_path_sys, "w", encoding='utf-8') as f: f.write(system_prompt)
            with open(prompt_save_path_user, "w", encoding='utf-8') as f: f.write(user_prompt)
        except Exception as save_err: logging.warning(f"Failed to save extraction prompts: {save_err}")

        # 3. Call LLM for RAW JSON string
        model_to_use = "gpt-4o-mini" # Or "google/gemini-flash-1.5" etc. Adjust as needed
        logging.info(f"Calling LLM ({model_to_use}) for raw JSON extraction...")
        self._log_extraction_debug(extraction_debug_filepath, f"Calling LLM ({model_to_use}) for raw JSON extraction.")

        raw_json_string = None
        try:
            # Using the base client for raw JSON output
            raw_response = openai_client.chat.completions.create(
                model=model_to_use,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}, # Request JSON output mode
                temperature=0.1, # Low temperature for factual extraction
            )
            raw_json_string = raw_response.choices[0].message.content if raw_response.choices and raw_response.choices[0].message else None

            if raw_json_string:
                logging.info(f"LLM returned raw JSON string (length: {len(raw_json_string)}).")
                self._log_extraction_debug(extraction_debug_filepath, f"LLM returned raw JSON string (length: {len(raw_json_string)}).")
                try:
                    # Save raw response for debugging validation issues
                    with open(self.paths.raw_llm_response_debug, "w", encoding='utf-8') as f:
                         f.write(raw_json_string)
                    logging.info(f"Saved raw LLM response to {self.paths.raw_llm_response_debug}")
                except Exception as save_err:
                    logging.warning(f"Failed to save raw LLM response: {save_err}")
            else:
                logging.error("LLM response was empty.")
                self._log_extraction_debug(extraction_debug_filepath, "LLM response was empty.")
                return None

        except Exception as e:
            # Catch API errors etc.
            logging.error(f"Error getting raw JSON response from LLM ({model_to_use}): {e}")
            self._log_extraction_debug(extraction_debug_filepath, f"Error getting raw JSON response: {e}")
            return None

        # 4. Validate and potentially correct the raw JSON string
        benchmark_data: Optional[BenchmarkData] = None
        try:
            logging.info("Attempting Pydantic validation on raw JSON string...")
            self._log_extraction_debug(extraction_debug_filepath, "Attempting Pydantic validation on raw JSON.")
            benchmark_data = BenchmarkData.model_validate_json(raw_json_string)
            logging.info("Raw JSON string passed initial Pydantic validation.")
            self._log_extraction_debug(extraction_debug_filepath, "Raw JSON passed initial Pydantic validation.")

        except ValidationError as e_initial:
            logging.warning(f"Initial Pydantic validation failed. Attempting manual correction. Error: {e_initial.errors()}")
            self._log_extraction_debug(extraction_debug_filepath, f"Initial Pydantic validation failed. Errors: {e_initial.errors()}. Attempting correction.")

            try:
                # Manually parse string -> dict
                data_dict = json.loads(raw_json_string)

                # Attempt correction: Check if 'scores' is a list of strings
                if isinstance(data_dict, dict) and "scores" in data_dict and isinstance(data_dict["scores"], list):
                    corrected_scores = []
                    needs_correction = False
                    for item in data_dict["scores"]:
                        if isinstance(item, str):
                            needs_correction = True
                            try:
                                # Try parsing the string item as JSON -> dict
                                parsed_item = json.loads(item)
                                corrected_scores.append(parsed_item)
                            except json.JSONDecodeError:
                                logging.warning(f"Score item was string but not valid JSON: {item[:100]}... Skipping.")
                                self._log_extraction_debug(extraction_debug_filepath, f"Score item string not valid JSON: {item[:100]}...")
                                # Skip this item or add as is? Skipping is safer for validation.
                        else:
                            # Assume it's already a dict (or will fail validation later)
                            corrected_scores.append(item)

                    if needs_correction:
                        logging.info("Applied correction for string items in 'scores' list.")
                        self._log_extraction_debug(extraction_debug_filepath,"Applied correction for string items in 'scores' list.")
                        data_dict["scores"] = corrected_scores

                # Retry validation with the potentially corrected dictionary
                logging.info("Attempting Pydantic validation on potentially corrected dictionary...")
                self._log_extraction_debug(extraction_debug_filepath, "Attempting Pydantic validation on corrected dict.")
                # Use TypeAdapter for validating a Python object
                benchmark_data = TypeAdapter(BenchmarkData).validate_python(data_dict)
                logging.info("Corrected dictionary passed Pydantic validation.")
                self._log_extraction_debug(extraction_debug_filepath,"Corrected dictionary passed Pydantic validation.")

            except (json.JSONDecodeError, ValidationError, Exception) as e_correct:
                # If manual parsing, correction, or re-validation fails
                logging.error(f"Manual JSON correction or re-validation failed: {e_correct}")
                self._log_extraction_debug(extraction_debug_filepath, f"Manual JSON correction or re-validation failed: {e_correct}")
                benchmark_data = None # Ensure it's None

        except Exception as e_other:
            # Catch any other unexpected errors during validation phase
            logging.error(f"Unexpected error during validation/correction: {e_other}")
            self._log_extraction_debug(extraction_debug_filepath, f"Unexpected validation/correction error: {e_other}")
            benchmark_data = None

        # Final check and return
        if benchmark_data and benchmark_data.scores:
             logging.info(f"Extraction successful, found {len(benchmark_data.scores)} model scores.")
             self._log_extraction_debug(extraction_debug_filepath, f"Extraction successful, found {len(benchmark_data.scores)} scores.")
             return benchmark_data
        elif benchmark_data:
             logging.warning(f"Extraction successful but no scores found in the data for '{benchmark_id}'.")
             self._log_extraction_debug(extraction_debug_filepath, "Extraction successful but no scores found.")
             # Return the data structure even if scores are empty, maybe it's valid but empty
             return benchmark_data
        else:
            logging.error(f"Final extraction attempt failed for benchmark '{benchmark_id}'. Check debug logs.")
            self._log_extraction_debug(extraction_debug_filepath, "Extraction method finished with failure.")
            return None


    def _log_extraction_debug(self, filepath: Path, message: str):
        """Appends a message to the benchmark-specific extraction debug log."""
        try:
            with open(filepath, "a", encoding='utf-8') as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")
        except Exception as e:
            # Log error about logging itself to the main log
            logging.error(f"Failed to write to debug log {filepath}: {e}")


    def _classify_and_process_models(
        self,
        benchmark_data: BenchmarkData,
        benchmark_id: str
    ) -> Tuple[bool, Dict[str, str]]: # Return (models_updated, id_mapping)
        """
        Classifies extracted model IDs, logs results, updates self.models
        if new models are identified, and returns the mapping of
        original_id -> canonical_id.

        Args:
            benchmark_data: The extracted benchmark data.
            benchmark_id: The ID of the benchmark being processed.

        Returns:
            Tuple[bool, Dict[str, str]]:
                - bool: True if self.models was updated, False otherwise.
                - Dict[str, str]: Mapping of original_id -> canonical_id for matched models.
        """
        id_mapping: Dict[str, str] = {} # Stores original_id -> canonical_id
        models_updated = False # Track if we add any new models

        if not self.model_classifier:
            logging.error("Model Classifier not available. Skipping classification.")
            return models_updated, id_mapping
        if not benchmark_data or not benchmark_data.scores:
             logging.info(f"No scores in extracted data for '{benchmark_id}'. Skipping classification.")
             return models_updated, id_mapping

        # Filter out potential null/empty IDs before sending to classifier
        extracted_ids_list = list(set(
            score.model_id.strip() for score in benchmark_data.scores
            if score.model_id and score.model_id.strip()
        ))

        if not extracted_ids_list:
             logging.info(f"No valid model IDs found in scores for '{benchmark_id}'. Skipping classification.")
             return models_updated, id_mapping

        logging.info(f"Classifying {len(extracted_ids_list)} unique model IDs for '{benchmark_id}'...")

        try:
            classification_results: List[ModelClassificationResult] = self.model_classifier.classify_batch(
                extracted_ids=extracted_ids_list,
                benchmark_id=benchmark_id
            )

            # Process results and log
            results_map = {res.original_id: res for res in classification_results}
            with open(self.paths.model_classification_log, "a", encoding='utf-8') as f:
                f.write(f"[{datetime.now().isoformat()}] Benchmark: {benchmark_id} - Model Classification Report\n")
                f.write("-" * 60 + "\n")

                if not classification_results:
                    f.write("No classification results returned by the LLM.\n")
                    logging.warning(f"LLM classifier returned no results for '{benchmark_id}'.")

                processed_ids_for_log = set()
                for original_id in extracted_ids_list: # Iterate through the IDs we sent
                    if original_id in processed_ids_for_log: continue

                    classification = results_map.get(original_id)
                    log_entry = f"  Input ID: '{original_id}'\n"

                    if not classification:
                        log_entry += (
                            f"  Classification: UNCLASSIFIED (LLM result missing)\n"
                            f"  Explanation: LLM did not return a classification for this ID.\n"
                        )
                        logging.warning(f"Classifier did not return result for ID '{original_id}' in benchmark '{benchmark_id}'.")
                    else:
                        log_entry += f"  Classification: {classification.status.value}\n"
                        if classification.status == ClassificationStatus.EXISTING:
                            if classification.matched_id:
                                log_entry += f"    Matched Canonical ID: '{classification.matched_id}'\n"
                                # --- Store the mapping ---
                                id_mapping[classification.original_id] = classification.matched_id
                            else:
                                log_entry += f"    Matched Canonical ID: NOT FOUND (Status was EXISTING, but no ID provided!)\n"
                                logging.warning(f"Model '{original_id}' classified as EXISTING but no matched_id provided.")

                        elif classification.status == ClassificationStatus.NEW:
                            # --- Add New Model Logic ---
                            if self.models is not None and original_id not in self.models:
                                logging.info(f"Identified NEW model: '{original_id}'. Adding to models list.")
                                self.models[original_id] = ModelInfo(
                                    name=original_id, # Use ID as name initially
                                    organization="Unknown", # Placeholder
                                    release_date="Unknown",  # Placeholder
                                    license="Unknown"       # Placeholder
                                ).model_dump() # Store as dict
                                models_updated = True # Signal that models file needs saving
                                log_entry += f"    Action: Added as new model to internal list.\n"
                                # We don't add NEW models to the id_mapping; their original ID is now the canonical one.
                            elif self.models is None:
                                 logging.error("Cannot add new model '{original_id}', self.models is None.")
                            else: # Already exists (maybe added in a previous step or manually)
                                 logging.info(f"Model '{original_id}' classified as NEW, but already exists in models.json. No action taken.")
                                 log_entry += f"    Action: Already exists in models.json.\n"

                        elif classification.status == ClassificationStatus.UNCLASSIFIED:
                             log_entry += f"    Action: Kept original ID '{original_id}'.\n"
                             # No mapping needed, we keep the original ID


                        if classification.explanation:
                            log_entry += f"  Explanation: {classification.explanation}\n"

                    f.write(log_entry)
                    f.write("-" * 30 + "\n")
                    processed_ids_for_log.add(original_id)

                f.write("-" * 60 + "\n\n")

            logging.info(f"Model classification results appended to {self.paths.model_classification_log}")
            if models_updated:
                 logging.info(f"Found new models for benchmark '{benchmark_id}'. Models list updated in memory.")

            return models_updated, id_mapping # Return whether models were updated AND the mapping

        except Exception as e:
            # Catch exceptions raised by classify_batch (e.g., InstructorRetryException)
            # Error is already logged in classify_batch
            logging.error(f"Classification process failed for benchmark '{benchmark_id}'.")
            # Re-raise the exception to be caught by update_benchmark's main try-except
            raise e


    def update_all_benchmarks(self) -> Dict[str, bool]:
        """Update all benchmarks and return their status"""
        if not self.benchmarks:
             logging.warning("No benchmarks loaded to update.")
             return {}
        if not self.base_openai_client or not self.client or not self.model_classifier:
             logging.error("LLM client or Model Classifier not initialized. Cannot update benchmarks.")
             return {b.get("id", f"unknown_{i}"): False for i, b in enumerate(self.benchmarks)}
        if self.models is None:
             logging.error("Models data not loaded. Cannot update benchmarks.")
             return {b.get("id", f"unknown_{i}"): False for i, b in enumerate(self.benchmarks)}


        results = {}
        logging.info(f"Attempting to update {len(self.benchmarks)} benchmarks.")
        for benchmark in self.benchmarks:
            benchmark_id = benchmark.get("id")
            if not benchmark_id:
                 logging.warning(f"Skipping benchmark with missing ID: {benchmark}")
                 continue

            # update_benchmark now handles its own critical errors and returns True/False
            results[benchmark_id] = self.update_benchmark(benchmark_id)
            if results[benchmark_id]:
                 logging.info(f"Successfully updated benchmark '{benchmark_id}'.")
            else:
                 logging.warning(f"Failed to update benchmark '{benchmark_id}'. Check logs for details.")
                 # Continue to the next benchmark

        return results


def main():
    parser = argparse.ArgumentParser(description="Update LLM benchmark data")
    parser.add_argument("--benchmark", "-b", help="Benchmark ID to update (default: update all)", default="all")
    parser.add_argument("--url", help="Override the source URL for the benchmark")
    parser.add_argument("--root-dir", help="Root directory of the project (containing data/, src/, etc.)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], default="INFO",
                        help="Set logging level (default: INFO)")

    args = parser.parse_args()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level.upper()))

    # Check environment variable
    if not OPENROUTER_API_KEY:
        logging.critical("OPENROUTER_API_KEY environment variable not set. Exiting.")
        sys.exit(1)

    try:
        agent = BenchmarkUpdateAgent(args.root_dir)

        # Perform checks after initialization attempt
        if not agent.base_openai_client or not agent.client or not agent.model_classifier:
             logging.critical("BenchmarkUpdateAgent failed to initialize LLM components. Exiting.")
             sys.exit(1)
        if agent.models is None:
             logging.critical("BenchmarkUpdateAgent failed to load models data. Exiting.")
             sys.exit(1)
        # Allow continuing if benchmarks list is empty initially, but log warning
        if not agent.benchmarks:
             logging.warning("No benchmarks loaded from benchmarks.json. Update process may not find benchmarks to run.")
             # Do not exit here, allow specific benchmark update if ID is provided


        if args.benchmark.lower() == "all":
            if not agent.benchmarks:
                 logging.warning("No benchmarks loaded. Nothing to update in 'all' mode. Exiting.")
                 sys.exit(0)
            results = agent.update_all_benchmarks()
            successful_count = sum(1 for status in results.values() if status)
            total_count = len(results)
            logging.info(f"Finished updating all benchmarks. Succeeded: {successful_count}/{total_count}")
            # Exit code 0 if *any* succeeded, 1 if *all* failed or none were run
            sys.exit(0 if successful_count > 0 else 1)
        else:
            success = agent.update_benchmark(args.benchmark, args.url)
            logging.info(f"Finished updating benchmark '{args.benchmark}'. Success: {success}")
            sys.exit(0 if success else 1)

    except Exception as e:
        # Catch any unexpected top-level errors
        logging.critical(f"Unhandled exception in main execution: {e}", exc_info=True) # Log traceback for critical failures
        sys.exit(1)


if __name__ == "__main__":
    # This block for making imports work when run as script is often problematic.
    # Running as a module `python -m llmbench.benchmark_update_agent` is preferred.
    # However, keeping it for potential direct script execution.
    script_dir = Path(__file__).parent.resolve()
    project_root = script_dir.parent.parent.resolve() # Assumes src/llmbench/benchmark_update_agent.py

    # Add project root and src to sys.path if not already present
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
        # logging.info(f"Added {project_root} to sys.path") # Keep logging minimal before config
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))
        # logging.info(f"Added {project_root / 'src'} to sys.path")

    # Re-check imports after path modification, though module execution is better.
    try:
        from llmbench.model_classify import ModelClassifier, ModelClassificationResult, ClassificationStatus, CanonicalModel # Adjust based on actual location
    except ModuleNotFoundError as e:
        print(f"ERROR: Could not import classification module: {e}", file=sys.stderr)
        print("Ensure the script is run as a module (python -m llmbench.benchmark_update_agent) or PYTHONPATH is set correctly.", file=sys.stderr)
        sys.exit(1)

    main()