#!/usr/bin/env python3
"""
Context Summarizer for analysis management

Summarizes previous analyses when count reaches threshold (e.g., 20)
All analyses are condensed into a single summary entry
"""

import time
from typing import List, Tuple


class ContextSummarizer:
    """
    Summarize agent's previous ANALYSIS sections using LLM
    Triggers when analysis count reaches threshold (default: 20)
    """

    def __init__(self, llm_client, provider: str, model: str):
        """
        Initialize summarizer with LLM client

        Args:
            llm_client: OpenAI/Claude/Gemini client instance
            provider: "openai" | "claude" | "gemini"
            model: Model name (e.g., "gpt-5", "claude-sonnet-4-5-20250929", "gemini-2.5-flash")
        """
        self.client = llm_client
        self.provider = provider
        self.model = model
        self.total_summaries_created = 0

    def summarize_analyses(
        self,
        analyses: List[Tuple]
    ) -> Tuple[str, str]:
        """
        Summarize ANALYSIS sections using LLM

        Args:
            analyses: List of (step, analysis_text) tuples to summarize
                     Can include previous summaries (step="X-Y")

        Returns:
            Tuple of (step_range_label, summary_text)
            e.g., ("1-50", "Player navigated from bedroom to Route 103...")
        """
        if not analyses:
            return ("0-0", "No significant events.")

        # Extract step range
        step_range = self._extract_step_range(analyses)
        step_range_label = f"{step_range[0]}-{step_range[1]}"

        # Build prompt for summarization
        analyses_text = self._format_analyses_for_summary(analyses)
        prompt = self._build_summarization_prompt(analyses_text, step_range)

        # Call LLM
        start = time.time()
        summary = self._call_llm(prompt)
        duration = time.time() - start

        self.total_summaries_created += 1

        print(f"  📝 Summarization took {duration:.1f}s")
        print(f"  📊 Compressed {len(analyses)} entries → 1 summary ({len(summary)} chars)")

        return (step_range_label, summary.strip())

    def _extract_step_range(self, analyses: List[Tuple]) -> Tuple[int, int]:
        """
        Extract step range from analyses

        Handles both regular steps (int) and summary ranges (str like "0-100")
        """
        steps = []
        for step_info, _ in analyses:
            if isinstance(step_info, str) and '-' in step_info:
                # It's a summary range like "0-100"
                start, end = step_info.split('-')
                steps.extend([int(start), int(end)])
            else:
                # Regular step number
                steps.append(int(step_info))

        if not steps:
            return (0, 0)

        return (min(steps), max(steps))

    def _format_analyses_for_summary(
        self,
        analyses: List[Tuple],
        max_chars_per_entry: int = 300
    ) -> str:
        """
        Format analyses for summarization prompt

        Handles both summaries and regular analyses
        """
        lines = []
        for step_info, text in analyses:
            # Check if it's already a summary
            if isinstance(step_info, str) and '-' in step_info:
                lines.append(f"[Steps {step_info} - Previous Summary]")
            else:
                lines.append(f"[Step {step_info}]")

            # Truncate very long entries
            truncated = text[:max_chars_per_entry]
            if len(text) > max_chars_per_entry:
                truncated += "..."
            lines.append(truncated)
            lines.append("")

        return "\n".join(lines)

    def _build_summarization_prompt(
        self,
        analyses_text: str,
        step_range: Tuple[int, int]
    ) -> str:
        """Build prompt for LLM summarization"""
        return f"""You are summarizing an AI agent's observations while playing Pokemon Emerald.

Below are the agent's ANALYSIS sections from steps {step_range[0]} to {step_range[1]}.
Some entries may already be summaries from earlier periods - incorporate them into your summary.

Create a concise summary (2-4 sentences) that captures:

1. Key locations visited and navigation progress
2. Major challenges encountered (stuck situations, NPCs, obstacles)
3. Successful strategies discovered or milestones achieved
4. Important learnings about game mechanics

DO NOT include:
- Step-by-step details
- Repetitive information
- Minor observations

ANALYSIS SECTIONS TO SUMMARIZE:

{analyses_text}

---

Write a concise summary (2-4 sentences) covering steps {step_range[0]}-{step_range[1]}:"""

    def _call_llm(self, prompt: str) -> str:
        """
        Call LLM for summarization (text-only, no image)

        Args:
            prompt: Summarization prompt

        Returns:
            Summary text from LLM
        """
        if self.provider == "openai":
            # Use same model as CodeAgent - gpt-5 uses responses API
            response = self.client.responses.create(
                model=self.model,
                instructions="You are a helpful assistant that creates concise summaries.",
                input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
                reasoning={"effort": "low"},
            )
            return response.output_text

        elif self.provider == "claude":
            # Use same model as CodeAgent
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        elif self.provider == "gemini":
            # Use same model as CodeAgent (already configured in self.client)
            import google.generativeai as genai

            generation_config = genai.types.GenerationConfig(
                max_output_tokens=500,  # Increased from 200 for summaries
                temperature=0.3
            )

            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )

            # Check for safety/recitation issues
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                finish_reason = getattr(candidate, 'finish_reason', None)

                # RECITATION (2), SAFETY (3), or other issues
                if finish_reason in [2, 3, 12]:
                    print(f"  ⚠️ Gemini finish_reason={finish_reason}, using fallback summary")
                    # Return a safe fallback summary
                    return "Agent continued navigation and exploration during this period."

            # Check if we have valid text
            try:
                return response.text
            except (ValueError, AttributeError) as e:
                print(f"  ⚠️ Gemini response.text failed: {e}, using fallback")
                return "Agent continued navigation and exploration during this period."

        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
