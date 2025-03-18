import os
import json
import re
from google import genai
from google.genai import types
from utils import system_instruction
import streamlit as st


class GeminiProductExtractor:
    """Product and price extractor using Google's Gemini API"""

    def __init__(self, api_key: str = None, model_name: str = "gemini-2.0-flash"):
        """Initialize the Gemini-based product extractor

        Args:
            api_key: The Google API key. If not provided, will use GEMINI_API_KEY env var
            model_name: The Gemini model to use
        """
        # Set up Gemini API
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key is required. Set GEMINI_API_KEY environment variable or pass api_key parameter.")

        self.client = genai.Client(api_key=self.api_key)
        self.model = model_name

    def extract_products(self, text: str) -> list[dict[str, object]]:
        """Extract product names and prices from text using Gemini

        Args:
            text: The text to extract products from

        Returns:
            List of dictionaries containing product information
        """
        try:
            response = self.client.models.generate_content(
                model=self.model,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction.instruction,
                    max_output_tokens=2048,
                    temperature=0.1
                ),
                contents=[text]
            )

            # Parse the response JSON
            try:
                result = json.loads(response.text)

                # Add source text and model info to each product entry
                for product in result:
                    product["source_text"] = text
                    product["model"] = self.model

                return result
            except json.JSONDecodeError:
                # Fallback if response isn't valid JSON - look for code blocks
                json_match = re.search(
                    r'```json\s*(.*?)\s*```', response.text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        for product in result:
                            product["source_text"] = text
                            product["model"] = self.model
                        return result
                    except:
                        pass

                if hasattr(st, 'error'):
                    st.error(
                        f"Failed to parse Gemini response as JSON: {response.text}")
                print(
                    f"Failed to parse Gemini response as JSON: {response.text}")
                return []
        except Exception as e:
            if hasattr(st, 'error'):
                st.error(f"Error calling Gemini API: {e}")
            print(f"Error calling Gemini API: {e}")
            return []

    def process_text(self, text: str) -> list[dict[str, object]]:
        """Process text to extract products

        Args:
            text: The text to process

        Returns:
            List of extracted products
        """
        return self.extract_products(text)
