import json
import re
import ollama
from utils import system_instruction
import streamlit as st


class OllamaProductExtractor:
    """Product and price extractor using a local Ollama model"""

    def __init__(self, model_name: str = 'gemma3:4b'):
        """Initialize the Ollama-based product extractor

        Args:
            model_name: The Ollama model name to use
        """
        self.model_name = model_name

    def extract_products(self, text: str) -> list[dict[str, object]]:
        """Extract product names and prices from text using Ollama

        Args:
            text: The text to extract products from

        Returns:
            List of dictionaries containing product information
        """
        system_prompt = system_instruction.instruction

        try:
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ]
            )

            response_text = response['message']['content']

            # Parse the response JSON
            try:
                # Try direct JSON parsing
                result = json.loads(response_text)

                # Add source text and model info to each product entry
                for product in result:
                    product["source_text"] = text
                    product["model"] = f"ollama_{self.model_name}"

                return result

            except json.JSONDecodeError:
                # Fallback if response isn't valid JSON - look for code blocks
                json_match = re.search(
                    r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    try:
                        result = json.loads(json_match.group(1))
                        for product in result:
                            product["source_text"] = text
                            product["model"] = f"ollama_{self.model_name}"
                        return result
                    except:
                        pass

                if hasattr(st, 'error'):
                    st.error(
                        f"Failed to parse Ollama response as JSON: {response_text}")
                print(
                    f"Failed to parse Ollama response as JSON: {response_text}")
                return []

        except Exception as e:
            if hasattr(st, 'error'):
                st.error(f"Error calling Ollama API: {e}")
            print(f"Error calling Ollama API: {e}")
            return []

    def process_text(self, text: str) -> list[dict[str, object]]:
        """Process text to extract products

        Args:
            text: The text to process

        Returns:
            List of extracted products
        """
        return self.extract_products(text)
