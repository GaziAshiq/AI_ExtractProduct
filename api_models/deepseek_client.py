import os
import json
import re
from openai import OpenAI
from utils import system_instruction
import streamlit as st


class DeepSeekProductExtractor:
    """Product and price extractor using DeepSeek API"""

    def __init__(self, api_key: str = None, model_name: str = "deepseek-chat"):
        """Initialize the DeepSeek-based product extractor

        Args:
            api_key: The DeepSeek API key. If not provided, will use DEEPSEEK_API_KEY env var
        """
        # Set up DeepSeek API
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable or pass api_key parameter.")

        self.client = OpenAI(api_key=self.api_key,
                             base_url="https://api.deepseek.com")
        self.model = model_name

    def extract_products(self, text: str) -> list[dict[str, object]]:
        """Extract product names and prices from text using DeepSeek

        Args:
            text: The text to extract products from

        Returns:
            List of dictionaries containing product information
        """
        system_prompt = system_instruction.instruction

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.1,
                max_tokens=1024
            )

            response_text = response.choices[0].message.content

            # Parse the response JSON
            try:
                # Try direct JSON parsing
                result = json.loads(response_text)

                # Add source text and model info to each product entry
                for product in result:
                    product["source_text"] = text
                    product["model"] = self.model

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
                            product["model"] = self.model
                        return result
                    except:
                        pass

                if hasattr(st, 'error'):
                    st.error(
                        f"Failed to parse DeepSeek response as JSON: {response_text}")
                print(
                    f"Failed to parse DeepSeek response as JSON: {response_text}")
                return []

        except Exception as e:
            if hasattr(st, 'error'):
                st.error(f"Error calling DeepSeek API: {e}")
            print(f"Error calling DeepSeek API: {e}")
            return []

    def process_text(self, text: str) -> list[dict[str, object]]:
        """Process text to extract products

        Args:
            text: The text to process

        Returns:
            List of extracted products
        """
        return self.extract_products(text)
