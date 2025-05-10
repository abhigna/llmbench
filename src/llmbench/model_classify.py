import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Iterable
from pydantic import BaseModel, Field, ValidationError, model_validator, ValidationInfo
from enum import Enum

import instructor
from instructor.exceptions import InstructorRetryException # Import specific exception
from openai import OpenAI

# Configure logging (keep INFO level for core steps)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Pydantic Models for LLM Classification ---

# Model to represent a known canonical model ID for the LLM to choose from
class CanonicalModel(BaseModel):
    id: str = Field(..., description="The canonical ID of the known model (e.g., 'claude-3-opus')")
    name: str = Field(..., description="The display name of the known model (e.g., 'Claude 3 Opus')")

# Enum for classification status
class ClassificationStatus(str, Enum):
    EXISTING = "existing" # Matched to a known canonical ID
    NEW = "new"         # No match found

# Model for the LLM's classification output for a single input ID
class ModelClassificationResult(BaseModel):
    original_id: str = Field(..., description="The original model ID string that was extracted from the benchmark data.")
    status: ClassificationStatus = Field(..., description="Classification status: 'existing' if matched to a known canonical ID from the provided list, 'new' otherwise.")
    matched_id: Optional[str] = Field(None, description="The canonical ID from the known list that the original_id matched to, if status is 'existing'. Must be one of the allowed_models IDs.")
    explanation: Optional[str] = Field(None, description="Brief explanation for the classification decision, especially for 'existing' matches or 'new' status.")

    @model_validator(mode='after')
    def validate_match(self):
        context = None
        allowed_models_list = None # Initialize as None to indicate if we successfully retrieved it

        # --- Safely attempt to get context and the allowed models list ---
        if self.model_extra:
             context = self.model_extra.get('_instructor_context')
             if context and "allowed_models" in context:
                  # Successfully retrieved the list (it might be empty)
                  allowed_models_list = context["allowed_models"]

        # --- Validation Logic ---
        if self.status == ClassificationStatus.EXISTING:
            # Rule 1: 'existing' status MUST have a matched_id
            if self.matched_id is None:
                 raise ValueError("matched_id must be provided if status is 'existing'")

            # Rule 2: If we have a NON-EMPTY list of allowed models, the matched_id MUST be in it.
            if allowed_models_list is not None: # Check if we successfully got the list from context
                if not allowed_models_list:
                     # We got the context, but the list was empty. LLM shouldn't return EXISTING.
                     logging.warning(
                         f"LLM classified '{self.original_id}' as 'existing' ('{self.matched_id}') but the list of known models provided was empty. "
                         f"This suggests an LLM error. Allowing for now, but should ideally be 'new'."
                         # Consider changing status here if strictness is required: self.status = ClassificationStatus.NEW
                     )
                     # We don't raise an error here to be more lenient, but log a warning.
                else:
                    # We got the context, and the list is NOT empty. Perform the check.
                    allowed_ids = {model.id for model in allowed_models_list}
                    if self.matched_id not in allowed_ids:
                        # Raise error for Instructor retry if ID is invalid against the non-empty list
                        raise ValueError(f"matched_id '{self.matched_id}' not found in the provided non-empty list of allowed canonical models.")
            else:
                 # Context was missing (allowed_models_list is still None).
                 # We cannot validate against the list. Log a warning.
                 logging.warning(
                     f"Validation context (allowed_models) missing for 'existing' status check on '{self.original_id}'. "
                     f"Cannot validate matched_id ('{self.matched_id}') against allowed list."
                 )

        elif self.status == ClassificationStatus.NEW:
            # Rule 3: 'new' status should ideally not have a matched_id. Correct if LLM provided one.
            if self.matched_id is not None:
                 logging.warning(f"LLM returned status 'new' for original_id '{self.original_id}' but provided non-null matched_id '{self.matched_id}'. Correcting matched_id to None.")
                 self.matched_id = None

        return self

# Wrapper model for the list of results - often helps with provider compatibility
class ClassificationResponse(BaseModel):
    results: List[ModelClassificationResult] = Field(..., description="List of classification results for the input IDs.")


class ModelClassifier:
    def __init__(self, client: OpenAI, known_models: Dict[str, Any]):
        """
        Initializes the ModelClassifier with an LLM client and known canonical models.

        Args:
            client: An initialized OpenAI client (patched by Instructor).
            known_models: A dictionary where keys are the canonical model IDs.
        """
        self.client = client
        self.allowed_models: List[CanonicalModel] = [
            CanonicalModel(id=model_id, name=model_info.get("name", model_id))
            for model_id, model_info in known_models.items()
        ]
        logging.info(f"ModelClassifier initialized with {len(self.allowed_models)} known canonical models.")

    def classify_batch(self, extracted_ids: List[str], benchmark_id: str) -> List[ModelClassificationResult]:
        """
        Classifies a batch of extracted model IDs using an LLM.

        Args:
            extracted_ids: A list of model ID strings extracted from a benchmark source.
            benchmark_id: The ID of the benchmark for context in the prompt.

        Returns:
            A list of ModelClassificationResult objects. Raises exceptions on failure.
        """
        if not extracted_ids:
            logging.warning("No extracted IDs provided for classification.")
            return []

        unique_extracted_ids = list(set(id.strip() for id in extracted_ids if id is not None and id.strip()))
        if not unique_extracted_ids:
             logging.warning("No valid unique extracted IDs after cleaning.")
             return []

        logging.info(f"Attempting to classify {len(unique_extracted_ids)} unique IDs for benchmark '{benchmark_id}' using LLM Classifier.")

        allowed_models_list_str = "\n".join([f"- ID: `{model.id}`, Name: '{model.name}'" for model in self.allowed_models])

        # System prompt remains largely the same, emphasizing the output schema (now ClassificationResponse containing the list)
        system_prompt = f"""
        You are an expert AI model classifier. Your task is to review a list of model identifiers extracted from a benchmark and match each one to the MOST LIKELY candidate from a provided list of known, canonical model IDs.

        For each input identifier, determine if it corresponds to one of the known canonical models.

        Your output MUST be a valid JSON object conforming to the `ClassificationResponse` schema, containing a 'results' field which is a list of JSON objects, where each object strictly follows the `ModelClassificationResult` schema.

        List of KNOWN Canonical Models to match against:
        {allowed_models_list_str}

        Classification Rules:
        1. For each 'original_id' from the user prompt, carefully consider its text.
        2. **Normalization for Matching:** Apply these normalization steps to the 'original_id' before comparing to the KNOWN list:
           - Convert to lowercase.
           - Replace common separators (spaces, '.', '_', '()') with hyphens (-).
           - Remove common suffixes like '-preview', '-latest', '-beta', '-v1', '-v2', and date suffixes (e.g., '-YYYYMMDD') *only if they appear at the very end*.
           - Specifically remove '-exp' or '-experimental' if they appear at the very end.
           - Keep core identifiers intact (e.g., '-mini', '-high', '-instruct', '-chat', '-sonnet', '-haiku', '-opus').
           - Examples:
             - 'gpt-4.5-preview-2025-01-15' suggests 'gpt-4-5' if known.
             - 'Claude 3 Opus' suggests 'claude-3-opus'.
             - 'gemini-2-5-pro-exp' suggests 'gemini-2-5-pro'.
             - 'gemini-2-0-pro-experimental' suggests 'gemini-2-0-pro'.
        3. **Matching Decision:**
           - If the normalized ID matches a KNOWN model: 'status' = '{ClassificationStatus.EXISTING.value}', 'matched_id' = the EXACT canonical ID from the KNOWN list. Provide brief 'explanation'.
           - If no clear match after normalization: 'status' = '{ClassificationStatus.NEW.value}', 'matched_id' = null. Provide brief 'explanation'.
        4. **Accuracy:** Only classify as 'existing' if confident.
        5. **Strict Output:** Ensure the output is *only* the JSON object conforming to the `ClassificationResponse` schema, containing the `results` list. Do NOT include entries for IDs not in the user prompt.
        """

        user_prompt = f"""
        Classify the following list of extracted model IDs based on the provided list of KNOWN Canonical Models for the '{benchmark_id}' benchmark. Apply the normalization logic described in the system prompt when searching for a match.

        Return a JSON object containing a 'results' list, where each element is a classification result object for one Input ID provided below.

        Input IDs to classify:
        {json.dumps(unique_extracted_ids, indent=2)}

        Return ONLY the JSON object.
        """

        logging.debug("Classification System Prompt:\n" + system_prompt)
        logging.debug("Classification User Prompt:\n" + user_prompt)

        try:
            # Use the wrapper model ClassificationResponse
            classification_response: ClassificationResponse = self.client.chat.completions.create(
                model="openai/gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_model=ClassificationResponse, # Use the wrapper model
                validation_context={"allowed_models": self.allowed_models},
                temperature=0,
                max_retries=1 # Limit retries if needed, default is often 1 or 2
            )

            # Return the list of results from the wrapper object
            return classification_response.results

        except (ValidationError, InstructorRetryException) as e:
            # Log the specific error type and message but don't include full traceback by default
            logging.error(f"LLM classification failed after retries for benchmark '{benchmark_id}': {type(e).__name__} - {e}")
            # Re-raise the exception to signal failure to the calling agent, halting the specific benchmark update
            raise e
        except Exception as e:
            # Catch other potential API errors
            logging.error(f"Unexpected error during LLM classification for benchmark '{benchmark_id}': {e}")
            # Re-raise to halt the process
            raise e