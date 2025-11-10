"""
Meta-Agent: Evidence-based Knowledge Validator

주기적으로 knowledge base의 각 entry를 LLM으로 검증하고,
invalid한 것은 삭제, valid한 것은 validation 필드 기록.
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from utils.knowledge_base import KnowledgeBase
from utils.llm_logger import log_llm_interaction, log_llm_error
import logging

logger = logging.getLogger(__name__)


class MetaAgent:
    """
    Meta-Agent that validates knowledge entries with LLM
    """

    def __init__(
        self,
        kb_path: str = ".pokeagent_cache/knowledge.json",
        llm_client=None
    ):
        self.kb = KnowledgeBase(filepath=kb_path)
        self.llm = llm_client  # LLM client for validation (Gemini, OpenAI, etc.)

    def needs_validation(self, entry: Dict) -> bool:
        """
        이 entry를 검증해야 하는가?

        Returns:
            True if needs validation, False if already validated
        """
        # validation 필드가 있으면 스킵
        validation = entry.get('validation')
        if validation:
            return False

        return True

    def build_validation_prompt(self, entry: Dict, evidence: Dict) -> str:
        """
        LLM에게 knowledge entry를 검증하도록 요청하는 프롬프트
        """
        prompt = f"""You are a knowledge validator for a Pokemon Emerald AI agent.

Your task: Validate whether this knowledge claim is TRUE, WELL-SUPPORTED, and USEFUL.

=== KNOWLEDGE CLAIM ===
{entry['content']}

=== EVIDENCE PROVIDED ===
{entry.get('evidence_text', '(No evidence text provided)')}

{f'''Game State:
{json.dumps(evidence.get('state', {}), indent=2)}
''' if evidence.get('state') else '(No game state evidence)'}

=== METADATA ===
Created at: Step {entry['created_step']}, Milestone {entry['created_milestone']}

=== YOUR TASK ===

Evaluate this knowledge based on:

1. **Evidence Quality**
   - Is the evidence sufficient to support this claim?
   - Does the game state match what the claim says?
   - Is the evidence specific enough?

2. **Usefulness**
   - Is this knowledge actionable for the agent?
   - Or is it too vague/obvious/redundant?

3. **Validity**
   - Does this make sense for Pokemon Emerald gameplay?
   - Are there any red flags?

**Respond in this EXACT format:**

STATUS: <verified|suspicious|invalid>
SCORE: <0.0 to 1.0>
REASONING: <your explanation in 1-2 sentences>

**Examples:**

STATUS: verified
SCORE: 0.95
REASONING: The claim "moving down from (12,18) goes to (12,19)" is directly supported by game state evidence showing position changed. Clear and actionable.

STATUS: invalid
SCORE: 0.2
REASONING: The claim is contradictory and lacks concrete evidence. The evidence text is too vague to verify the claim.

STATUS: suspicious
SCORE: 0.6
REASONING: The claim seems plausible but the evidence is incomplete. Needs more verification but might be useful.

**Now validate this entry:**
"""
        return prompt

    def parse_validation_response(self, response: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract status, score, reasoning
        """
        lines = response.strip().split('\n')
        result = {
            'status': 'suspicious',  # default
            'score': 0.5,
            'reasoning': ''
        }

        for line in lines:
            line = line.strip()
            if line.startswith('STATUS:'):
                status = line.split(':', 1)[1].strip().lower()
                if status in ['verified', 'suspicious', 'invalid']:
                    result['status'] = status

            elif line.startswith('SCORE:'):
                try:
                    score = float(line.split(':', 1)[1].strip())
                    result['score'] = max(0.0, min(1.0, score))
                except:
                    pass

            elif line.startswith('REASONING:'):
                result['reasoning'] = line.split(':', 1)[1].strip()

        return result

    def validate_entry(self, entry_dict: Dict) -> Optional[Dict[str, Any]]:
        """
        LLM으로 단일 entry 검증

        Returns:
            validation dict if valid/suspicious, None if invalid (should delete)
        """
        if not self.llm:
            logger.warning("No LLM client provided, skipping validation")
            return None

        entry_id = entry_dict['id']

        # Check if already validated
        if not self.needs_validation(entry_dict):
            logger.debug(f"⏭️  Skipping {entry_id} (already validated)")
            return entry_dict.get('validation')

        logger.info(f"🔍 Validating {entry_id}...")

        # Load evidence
        evidence = self.kb.get_evidence(entry_id)
        if not evidence:
            evidence = {"text": "", "state": None, "screenshot": None}

        # Build prompt
        prompt = self.build_validation_prompt(entry_dict, evidence)

        # Call LLM (no try-except for debugging)
        # Attach screenshot if available
        images = []
        if evidence.get('screenshot'):
            images.append(evidence['screenshot'])

        response = self.llm.query(prompt, images=images)
        result = self.parse_validation_response(response)

        logger.info(f"   Status: {result['status']}, Score: {result['score']:.2f}")
        logger.debug(f"   Reasoning: {result['reasoning']}")

        # 모든 경우 validation record 생성 (verified, suspicious, invalid 모두)
        validation = {
            "status": result['status'],
            "validity_score": result['score'],
            "reasoning": result['reasoning'],
            "validated_timestamp": datetime.now().isoformat()
        }

        return validation

    def validate_all(self, max_validations: int = 100) -> Dict[str, Any]:
        """
        최신 knowledge entries 검증 (아직 검증 안된 것만)
        최신순으로 순회하며, 최신 20개만 확인

        Args:
            max_validations: 최대 LLM 호출 횟수 제한

        Returns:
            검증 결과 통계
        """
        entries = self.kb.get_all()

        # 최신순으로 정렬 (created_timestamp 내림차순)
        entries_sorted = sorted(
            entries,
            key=lambda e: e['created_timestamp'],
            reverse=True
        )

        # 최신 20개만 선택
        entries_sorted = entries_sorted[:max_validations]

        stats = {
            'total': len(entries_sorted),
            'already_validated': 0,
            'newly_validated': 0,
            'skipped': 0
        }

        validated_count = 0

        for entry_dict in entries_sorted:
            # 이미 검증됨
            if not self.needs_validation(entry_dict):
                stats['already_validated'] += 1
                continue

            # LLM 호출 제한 도달
            if validated_count >= max_validations:
                stats['skipped'] += 1
                continue

            # 검증 수행 (에러 발생시 raise)
            validation = self.validate_entry(entry_dict)
            validated_count += 1

            # 정상 검증 완료 (verified/suspicious/invalid 모두 기록)
            success = self.kb.update_validation_by_id(entry_dict['id'], validation)
            if success:
                stats['newly_validated'] += 1

                if validation['status'] == 'verified':
                    status_emoji = "✅"
                elif validation['status'] == 'invalid':
                    status_emoji = "❌"
                    self.kb.delete_by_id(entry_dict['id'])
                else:  # suspicious
                    status_emoji = "⚠️"
                    self.kb.delete_by_id(entry_dict['id'])

                print(f"{status_emoji} Validated {entry_dict['id']}: {validation['status']} ({validation['validity_score']:.2f})")

        return stats

    def run_validation_cycle(self, max_validations: int = 20) -> None:
        """
        검증 사이클 실행
        """
        print("\n" + "="*80)
        print("🔍 META-AGENT VALIDATION CYCLE")
        print("="*80)

        stats = self.validate_all(max_validations=max_validations)

        print(f"\n📊 Validation Results:")
        print(f"   Total entries checked: {stats['total']}")
        print(f"   Already validated: {stats['already_validated']}")
        print(f"   Newly validated: {stats['newly_validated']}")
        print(f"   Skipped (limit): {stats['skipped']}")
        print("="*80)


# ============================================================================
# LLM Client Wrapper (for Gemini/OpenAI/etc)
# ============================================================================

class SimpleLLMClient:
    """
    간단한 LLM client wrapper
    지원: OpenAI GPT-5, Gemini
    """

    def __init__(self, provider: str = "openai", model: str = None):
        self.provider = provider
        self.model = model or self._get_default_model()
        self._init_client()

    def _get_default_model(self):
        if self.provider == "openai":
            return "gpt-5"
        elif self.provider == "gemini":
            return "gemini-2.0-flash-exp"
        return "gpt-5"

    def _init_client(self):
        """Initialize the actual LLM client"""
        if self.provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY not set")
                self.client = openai.OpenAI(api_key=api_key)
                logger.info(f"Initialized OpenAI client with model {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI: {e}")
                self.client = None

        elif self.provider == "gemini":
            try:
                import google.generativeai as genai
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY not set")
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self.model)
                logger.info(f"Initialized Gemini client with model {self.model}")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.client = None
        else:
            raise NotImplementedError(f"Provider {self.provider} not implemented")

    def query(self, prompt: str, images: List = None) -> str:
        """
        Query the LLM

        Args:
            prompt: Text prompt
            images: List of PIL Images (optional)

        Returns:
            LLM response text
        """
        if not self.client:
            raise ValueError("LLM client not initialized")

        start_time = time.time()

        try:
            if self.provider == "openai":
                # OpenAI GPT-5 (uses chat completions API)
                content_parts = [{"type": "text", "text": prompt}]

                # Add images if provided
                if images:
                    for img in images:
                        # Convert PIL Image to base64
                        import io
                        import base64
                        buffer = io.BytesIO()
                        img.save(buffer, format='PNG')
                        img_b64 = base64.b64encode(buffer.getvalue()).decode()

                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                        })

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": content_parts
                    }]
                )
                result = response.choices[0].message.content

                # Extract token usage
                token_usage = {}
                if hasattr(response, 'usage'):
                    token_usage = {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens
                    }

                # Log interaction
                duration = time.time() - start_time
                log_llm_interaction(
                    interaction_type="meta_agent_validation",
                    prompt=prompt,
                    response=result,
                    duration=duration,
                    metadata={"has_image": bool(images), "token_usage": token_usage},
                    model_info={"model": self.model, "provider": self.provider}
                )

                return result

            elif self.provider == "gemini":
                if images:
                    # Multimodal prompt
                    content = [prompt] + images
                    response = self.client.generate_content(content)
                else:
                    response = self.client.generate_content(prompt)

                result = response.text

                # Log interaction
                duration = time.time() - start_time
                log_llm_interaction(
                    interaction_type="meta_agent_validation",
                    prompt=prompt,
                    response=result,
                    duration=duration,
                    metadata={"has_image": bool(images)},
                    model_info={"model": self.model, "provider": self.provider}
                )

                return result

            raise NotImplementedError(f"Provider {self.provider} not implemented")

        except Exception as e:
            duration = time.time() - start_time
            log_llm_error(
                interaction_type="meta_agent_validation",
                prompt=prompt,
                error=str(e),
                metadata={"provider": self.provider, "model": self.model, "duration": duration}
            )
            raise


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="Meta-Agent Knowledge Validator")
    parser.add_argument(
        "--max-validations",
        type=int,
        default=20,
        help="Maximum number of entries to validate (LLM call limit)"
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="openai",
        choices=["openai", "gemini"],
        help="LLM provider to use"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="LLM model name (default: provider's default)"
    )

    args = parser.parse_args()

    # Initialize LLM client
    llm = SimpleLLMClient(provider=args.provider, model=args.model)

    # Initialize meta-agent
    meta = MetaAgent(llm_client=llm)

    # Run validation
    meta.run_validation_cycle(max_validations=args.max_validations)
