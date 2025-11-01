#!/usr/bin/env python3
"""
Code-generating agent that uses OpenAI Vision API to generate Python code
which is then executed to determine game actions.
"""

import openai
import os
import time
import io
import base64
from utils.llm_logger import get_llm_logger
from utils.state_formatter import format_state_for_llm
from utils.milestone_manager import MilestoneManager
from utils.prompt_builder import CodeAgentPromptBuilder, CodePromptConfig
from utils.stuck_detector import StuckDetector


class CodeAgent:
    """Agent that generates and executes Python code to determine actions"""

    def __init__(self):
        """Initialize the CodeAgent with OpenAI client and logger"""
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        self.client = openai.OpenAI(api_key=api_key)
        self.llm_logger = get_llm_logger()
        self.model = "gpt-5-mini"
        self.step_count = 0

        # Milestone manager (like SimpleAgent's objectives)
        self.milestone_manager = MilestoneManager()

        # Stuck detector (threshold=3)
        self.stuck_detector = StuckDetector(threshold=3)

        # Code caching
        self.last_generated_code = None
        self.code_generation_count = 0

        # Prompt builder for structured prompt generation
        self.prompt_builder = CodeAgentPromptBuilder(
            config=CodePromptConfig(
                include_visual_note=True,
                include_milestones=True,
                include_example_code=True
            )
        )

    def step(self, game_state):
        """
        Generate code based on game state and execute it to get action

        Stuck이 아닐 때는 이전 코드 재사용, Stuck일 때만 새 코드 생성

        Args:
            game_state: Dict with keys:
                - 'frame': PIL Image
                - 'player': player info dict
                - 'game': game info dict
                - 'map': map info dict
                - 'visual': visual info dict

        Returns:
            {'action': 'up'} or {'action': ['up', 'a']}
        """
        self.step_count += 1

        # 1. Check for stuck pattern
        is_stuck = self.stuck_detector.check_stuck(game_state)

        # 2. Code selection logic
        try:
            if is_stuck or self.last_generated_code is None:
                # Stuck이거나 첫 실행 -> 새 코드 생성
                code = self._generate_new_code(game_state, is_stuck)
                self.last_generated_code = code
                self.code_generation_count += 1
            else:
                # Not stuck -> 이전 코드 재사용
                code = self.last_generated_code
                print(f"🔄 Reusing previous code (generation #{self.code_generation_count})")

            # 3. Execute code to get action
            action = self._execute_code(code, game_state)

            # 4. Record action and reset if stuck
            self.stuck_detector.record_action(action)
            if is_stuck:
                self.stuck_detector.reset()

            return {'action': action}

        except Exception as e:
            print(f"❌ CodeAgent error: {e}")
            return {'action': 'b'}  # Default action on error

    def _generate_new_code(self, game_state, is_stuck: bool) -> str:
        """
        새로운 코드 생성 (LLM 호출)

        Args:
            game_state: 게임 상태
            is_stuck: Stuck 여부

        Returns:
            생성된 Python 코드
        """
        # 1. Stuck warning 생성
        stuck_warning = self.stuck_detector.get_stuck_warning()

        # 2. 이전 코드 (stuck인 경우만, raw code만 전달)
        previous_code_raw = ""
        if is_stuck and self.last_generated_code:
            previous_code_raw = self.last_generated_code

        # 3. State와 milestone 정보
        state_text = format_state_for_llm(game_state)
        screenshot_base64 = self._get_screenshot_base64(game_state)
        milestones = game_state.get('milestones', {})
        next_milestone_info = self.milestone_manager.get_next_milestone_info(milestones)

        # 4. 프롬프트 생성 (raw code 전달, 포맷팅은 PromptBuilder가 담당)
        prompt = self.prompt_builder.build_prompt(
            formatted_state=state_text,
            next_milestone_info=next_milestone_info,
            stuck_warning=stuck_warning,
            previous_code=previous_code_raw
        )

        # 5. LLM 호출
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Pokemon Emerald AI coding assistant. Generate clean, executable Python code based on visual and text information."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{screenshot_base64}"
                            }
                        }
                    ] if screenshot_base64 else [{"type": "text", "text": prompt}]
                }
            ],
        )
        duration = time.time() - start

        # 6. 코드 추출 및 로깅
        full_response = response.choices[0].message.content
        code = self._extract_code(full_response)

        self.llm_logger.log_interaction(
            interaction_type="code_generation",
            prompt=prompt,
            response=full_response,
            duration=duration,
            model_info={
                "model": self.model,
                "tokens": {
                    "prompt": response.usage.prompt_tokens,
                    "completion": response.usage.completion_tokens
                }
            }
        )

        # 7. Milestone 정보 출력
        if next_milestone_info:
            print(f"📍 Next Milestone: {next_milestone_info['id']}")
        else:
            print(f"🏆 All Milestones Complete!")

        return code

    def _get_screenshot_base64(self, game_state):
        """Get base64 screenshot from game state"""
        frame = game_state.get('frame')
        if frame:
            buffer = io.BytesIO()
            frame.save(buffer, format='PNG')
            return base64.b64encode(buffer.getvalue()).decode()
        return ""

    def _extract_code(self, text):
        """Extract code from structured LLM response"""
        # Try to find CODE: section first (for structured responses)
        if "CODE:" in text:
            code_section = text.split("CODE:")[1]

            # Extract from code block if present
            if "```python" in code_section:
                return code_section.split("```python")[1].split("```")[0].strip()
            elif "```" in code_section:
                return code_section.split("```")[1].split("```")[0].strip()

            # If no code block, take everything after CODE:
            # until we hit another section or end
            lines = code_section.split('\n')
            code_lines = []
            for line in lines:
                # Stop at next section header
                if any(line.strip().startswith(section) for section in ['ANALYSIS:', 'OBJECTIVES:', 'PLAN:', 'REASONING:']):
                    break
                code_lines.append(line)

            extracted = '\n'.join(code_lines).strip()
            if extracted:
                return extracted

        # Fallback to old behavior (for non-structured responses)
        if "```python" in text:
            return text.split("```python")[1].split("```")[0].strip()
        elif "```" in text:
            return text.split("```")[1].split("```")[0].strip()

        return text.strip()

    def _execute_code(self, code, state):
        """
        Execute generated code and extract action

        Args:
            code: Python code string with run(state) function
            state: Game state dict

        Returns:
            action string (e.g., 'up', 'a')
        """
        try:
            # Create execution environment
            exec_globals = {}
            exec(code, exec_globals)

            # Call run function
            if 'run' in exec_globals:
                action = exec_globals['run'](state)

                # Validate action
                valid_actions = ['a', 'b', 'start', 'select', 'up', 'down', 'left', 'right']
                if isinstance(action, str) and action.lower() in valid_actions:
                    return action.lower()
                else:
                    print(f"⚠️ Invalid action returned: {action}, using 'b'")
                    return 'b'
            else:
                print(f"⚠️ No 'run' function found in code, using 'b'")
                return 'b'

        except Exception as e:
            print(f"❌ Code execution error: {e}")
            print(f"Code:\n{code}")
            return 'b'  # Default action on error
