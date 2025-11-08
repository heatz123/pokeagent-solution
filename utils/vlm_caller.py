#!/usr/bin/env python3
"""
VLM caller for visual observations using Ollama
Handles screenshot + prompt -> typed result
"""

import base64
import io
import json
import time
from typing import Any
from PIL import Image


class VLMCaller:
    """
    Calls Ollama VLM to answer visual questions and coerces results to specified types
    """

    def __init__(self, model: str = "qwen3-vl:2b", keep_alive: int = -1):
        """
        Initialize VLM caller

        Args:
            model: Ollama model name (e.g., "qwen3-vl:2b", "llava")
            keep_alive: How long to keep model in memory (-1 = indefinitely, 0 = unload immediately, N = seconds)
        """
        self.model = model
        self.keep_alive = keep_alive

    def call(
        self,
        screenshot: Image.Image,
        prompt: str,
        return_type: type
    ) -> Any:
        """
        Call Ollama VLM with screenshot and prompt, return typed result

        Args:
            screenshot: PIL Image
            prompt: Question to ask about the image
            return_type: Expected type (bool, int, str, float, list, dict)

        Returns:
            Result coerced to return_type

        Raises:
            ValueError: If result cannot be coerced to return_type
        """
        start_time = time.time()

        # Build full prompt with type instruction
        full_prompt = self._build_typed_prompt(prompt, return_type)

        # Convert screenshot to base64
        screenshot_b64 = self._image_to_base64(screenshot)

        # Log request
        print(f"  [VLM Request] model={self.model}, return_type={return_type.__name__}")
        print(f"  [VLM Prompt] {prompt[:100]}...")

        # Call Ollama
        response_text = self._call_ollama(full_prompt, screenshot_b64)

        # Parse and coerce result
        result = self._coerce_result(response_text, return_type)

        # Calculate latency
        latency_ms = (time.time() - start_time) * 1000

        # Log response
        print(f"  [VLM Response] raw='{response_text}' -> {return_type.__name__}={result}")
        print(f"  [VLM Latency] {latency_ms:.1f}ms")

        return result

    def _build_typed_prompt(self, user_prompt: str, return_type: type) -> str:
        """
        Build prompt with type instruction

        Examples:
            bool: "Answer with only: true or false"
            int: "Answer with only: an integer number"
            str: "Answer with only: a short text string"
        """
        type_instructions = {
            bool: "Answer with ONLY 'true' or 'false' (no other text).",
            int: "Answer with ONLY an integer number (no other text).",
            float: "Answer with ONLY a decimal number (no other text).",
            str: "Answer with ONLY a short descriptive string (no other text).",
            list: "Answer with ONLY a JSON list (no other text).",
            dict: "Answer with ONLY a JSON object (no other text)."
        }

        instruction = type_instructions.get(return_type, "")

        return f"""{user_prompt}

{instruction}

Format your response as just the value, nothing else."""

    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
        return base64.b64encode(buffer.getvalue()).decode()

    def _call_ollama(self, prompt: str, screenshot_b64: str) -> str:
        """
        Call Ollama API

        Args:
            prompt: Prompt text
            screenshot_b64: Base64-encoded screenshot

        Returns:
            Response text
        """
        import ollama

        # Create client with explicit timeout (30 seconds)
        client = ollama.Client(timeout=30.0)

        response = client.chat(
            model=self.model,
            messages=[{
                'role': 'user',
                'content': prompt,
                'images': [screenshot_b64]
            }],
            keep_alive=self.keep_alive
        )

        return response['message']['content']

    def _coerce_result(self, text: str, return_type: type) -> Any:
        """
        Coerce text result to specified type

        Args:
            text: Raw text from VLM
            return_type: Target type

        Returns:
            Coerced value

        Raises:
            ValueError: If coercion fails
        """
        text = text.strip()

        try:
            if return_type == bool:
                # Handle various bool representations
                lower = text.lower()
                if lower in ['true', 'yes', '1']:
                    return True
                elif lower in ['false', 'no', '0']:
                    return False
                else:
                    raise ValueError(f"Cannot parse as bool: {text}")

            elif return_type == int:
                return int(text)

            elif return_type == float:
                return float(text)

            elif return_type == str:
                return text

            elif return_type == list:
                return json.loads(text)

            elif return_type == dict:
                return json.loads(text)

            else:
                raise ValueError(f"Unsupported return_type: {return_type}")

        except Exception as e:
            raise ValueError(f"Failed to coerce '{text}' to {return_type.__name__}: {e}")
