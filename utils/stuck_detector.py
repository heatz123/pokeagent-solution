"""
Simple stuck detection using game state hashing
"""
import hashlib
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class StuckDetector:
    """간단한 state 해싱 기반 stuck 감지기"""

    def __init__(self, threshold: int = 3):
        """
        Args:
            threshold: 같은 state가 N번 반복되면 stuck으로 판정
        """
        self.threshold = threshold
        self.state_hash_counts: Dict[str, int] = {}
        self.recent_actions: list = []

    def hash_game_state(self, game_state: Dict[str, Any]) -> str:
        """
        게임 state를 해싱 (screenshot 제외)

        Returns:
            MD5 해시 문자열
        """
        hash_data = {}

        # Player position
        player = game_state.get("player", {})
        position = player.get("position", {})
        hash_data["position"] = (position.get("x"), position.get("y"))

        # Map ID
        map_info = game_state.get("map", {})
        hash_data["map_id"] = map_info.get("id")

        # Context (battle/dialogue/overworld)
        game_info = game_state.get("game", {})
        hash_data["in_battle"] = game_info.get("is_in_battle", False)

        dialogue = game_info.get("dialogue", {})
        hash_data["dialogue"] = dialogue.get("text", "") if dialogue.get("active") else ""

        # Recent actions (최근 2개)
        if len(self.recent_actions) > 0:
            hash_data["recent_actions"] = tuple(self.recent_actions[-2:])

        # JSON으로 변환 후 MD5 해싱
        hash_string = json.dumps(hash_data, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()

    def check_stuck(self, game_state: Dict[str, Any]) -> bool:
        """
        Stuck 상태인지 확인

        Returns:
            True if stuck, False otherwise
        """
        # State 해싱
        state_hash = self.hash_game_state(game_state)

        # 카운트 증가
        self.state_hash_counts[state_hash] = \
            self.state_hash_counts.get(state_hash, 0) + 1

        # Stuck 판정
        is_stuck = self.state_hash_counts[state_hash] >= self.threshold

        if is_stuck:
            logger.warning(
                f"🔴 STUCK DETECTED! Same state {self.state_hash_counts[state_hash]} times"
            )

        return is_stuck

    def record_action(self, action):
        """액션 기록"""
        if isinstance(action, list):
            self.recent_actions.extend(action)
        else:
            self.recent_actions.append(str(action))

        # 최대 5개만 유지
        if len(self.recent_actions) > 5:
            self.recent_actions = self.recent_actions[-5:]

    def reset(self):
        """Stuck 해결 시 초기화"""
        logger.info("🔄 Stuck detector reset")
        self.state_hash_counts.clear()

    def get_stuck_warning(self) -> str:
        """
        Stuck 경고 메시지 (프롬프트용)

        Returns:
            경고 문자열 (stuck 아니면 빈 문자열)
        """
        # 현재 가장 높은 카운트 확인
        max_count = max(self.state_hash_counts.values()) if self.state_hash_counts else 0

        if max_count >= self.threshold:
            return (
                f"\n⚠️ WARNING: STUCK DETECTED! You are repeating the same game state ({max_count} times).\n"
                "💡 TIP: Try a COMPLETELY DIFFERENT approach:\n"
                "  - Move in a different direction\n"
                "  - Interact with different NPCs or objects\n"
                "  - Explore new areas\n"
            )
        return ""
