#!/usr/bin/env python3
"""
Gymnasium-style Environment for Pokemon Emerald

서버와 동일한 타이밍으로 동작하는 Gym 환경
- 액션 실행: ACTION_HOLD_FRAMES + ACTION_RELEASE_DELAY = 60 frames per step
- State format: LLM이 받는 것과 동일한 comprehensive state
- FPS 조절 가능 (서버는 80 FPS, 여기서는 더 빠르게 설정 가능)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Any, Optional
import logging
import random

from pokemon_env.emulator import EmeraldEmulator
from utils.state_formatter import format_state_for_llm

logger = logging.getLogger(__name__)


class PokemonEnv(gym.Env):
    """
    Gym-style environment for Pokemon Emerald
    서버와 동일한 타이밍/동작, FPS만 조절 가능
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # 서버와 동일한 액션 타이밍 상수 (server/app.py:84-85)
    ACTION_HOLD_FRAMES = 12
    ACTION_RELEASE_DELAY = 48
    FRAMES_PER_STEP = ACTION_HOLD_FRAMES + ACTION_RELEASE_DELAY  # 60 frames

    def __init__(
        self,
        rom_path: str = "Emerald-GBAdvance/rom.gba",
        base_fps: int = 120,  # 서버(80 FPS)보다 빠르게 설정 가능
        render_mode: Optional[str] = None,
        headless: bool = True,
        enable_milestones: bool = True,
        record_video: bool = False,
        video_fps: int = 30,  # 저장할 video FPS
        auto_save_video: bool = True,  # close() 시 자동 저장
        overlay_actions: bool = True,  # 비디오에 action 오버레이
        randomize_release_frames: bool = True,  # Release frame randomization
        release_frames_range: tuple[int, int] = (45, 54),  # Release frames 범위 (기본 48의 ±12.5%)
    ):
        """
        Args:
            rom_path: ROM 파일 경로
            base_fps: 기본 FPS (대화 중에는 자동으로 4배 가속)
            render_mode: 렌더링 모드 ("human", "rgb_array", None)
            headless: 헤드리스 모드 (화면 없이 실행)
            enable_milestones: Milestone 추적 활성화
            record_video: Video recording 활성화
            video_fps: 저장할 video의 FPS (기본 30)
            auto_save_video: close() 시 자동으로 video 저장
            overlay_actions: 비디오에 action 정보 오버레이 표시
            randomize_release_frames: Release frame randomization 활성화 (robustness 향상)
            release_frames_range: Release frames 범위 (min, max)
        """
        super().__init__()

        self.rom_path = rom_path
        self.base_fps = base_fps
        self.render_mode = render_mode
        self.headless = headless
        self.enable_milestones = enable_milestones
        self.record_video = record_video
        self.video_fps = video_fps
        self.auto_save_video = auto_save_video
        self.overlay_actions = overlay_actions
        self.randomize_release_frames = randomize_release_frames
        self.release_frames_range = release_frames_range

        # Emulator 초기화 (아직 initialize는 안함)
        self.emulator = None

        # Action space: 10가지 버튼
        # 0=A, 1=B, 2=START, 3=SELECT, 4=UP, 5=DOWN, 6=LEFT, 7=RIGHT, 8=L, 9=R
        self.action_space = spaces.Discrete(10)
        self.action_map = ["A", "B", "START", "SELECT", "UP", "DOWN", "LEFT", "RIGHT", "L", "R"]

        # Observation space: RGB image (Game Boy Advance resolution: 240x160)
        # obs = PIL Image or numpy array of shape (160, 240, 3)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(160, 240, 3),  # height, width, channels
            dtype=np.uint8,
        )

        # Step tracking
        self.step_count = 0
        self.total_frames = 0
        self.episode_reward = 0.0

        # Previous state for reward calculation
        self.prev_milestones = set()
        self.prev_location = None

        # Video recording
        self.video_frames = []  # 프레임 버퍼
        self.video_frame_skip = max(1, int(self.base_fps / self.video_fps))  # 프레임 스킵 계산
        self.episode_count = 0  # 에피소드 카운터
        self.video_dir = "videos"  # Video 저장 디렉토리
        self.current_action = None  # 현재 실행 중인 action (오버레이용)

        # Action tracking (for expert policies)
        self.last_action = None  # 이전 action 문자열 ("up", "a", "no_op" 등)
        self.last_facing = None  # 이전 facing 방향 ("north", "south", "east", "west")

        # Video 디렉토리 생성
        if record_video:
            import os

            os.makedirs(self.video_dir, exist_ok=True)

        logger.info(f"PokemonEnv initialized (base_fps={base_fps}, headless={headless}, record_video={record_video})")
        if record_video:
            logger.info(f"Video recording enabled: {video_fps} FPS (skip every {self.video_frame_skip} frames)")
            logger.info(f"Videos will be saved to: {self.video_dir}/")
        if randomize_release_frames:
            logger.info(
                f"Release frame randomization enabled: {release_frames_range[0]}-{release_frames_range[1]} frames per step"
            )

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        환경 리셋

        Args:
            seed: Random seed
            options: 추가 옵션 (load_state 등)

        Returns:
            observation, info
        """
        super().reset(seed=seed)

        # Save previous episode video before resetting
        if self.record_video and len(self.video_frames) > 0:
            self._save_episode_video()

        # Emulator가 없으면 새로 생성
        if self.emulator is None:
            self.emulator = EmeraldEmulator(rom_path=self.rom_path, headless=self.headless, sound=False)
            self.emulator.initialize()
            logger.info("Emulator initialized")

        # State 로딩 (options에서 지정 가능)
        if options and "load_state" in options:
            state_path = options["load_state"]
            self.emulator.load_state(path=state_path)
            logger.info(f"Loaded state from {state_path}")

        # Reset tracking
        self.step_count = 0
        self.total_frames = 0
        self.episode_reward = 0.0

        # Reset action tracking
        self.last_action = None
        self.last_facing = None

        # Get state (for info and reward tracking)
        state = self._get_state_dict()

        # Initialize previous state for rewards
        self.prev_milestones = self._get_completed_milestones()
        self.prev_location = state.get("player", {}).get("location")

        # Get observation (screenshot) and info (state dict)
        obs = self._get_obs()
        info = self._get_info(state)

        # Reset video recording for new episode
        if self.record_video:
            self.video_frames = []
            self.episode_count += 1
            logger.info(f"Video recording reset for episode {self.episode_count}")

        logger.info("Environment reset complete")

        return obs, info

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """
        한 스텝 실행 (서버와 동일한 타이밍)

        Args:
            action: 0-9 사이의 정수 (action_map 인덱스)

        Returns:
            observation, reward, terminated, truncated, info
        """
        if self.emulator is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        # Action을 버튼 문자열로 변환
        button = self.action_map[action]

        # 현재 action 저장 (비디오 오버레이용)
        self.current_action = button

        # Release frames 결정 (randomize가 활성화되면 매 step마다 랜덤)
        if self.randomize_release_frames:
            release_frames = random.randint(self.release_frames_range[0], self.release_frames_range[1])
        else:
            release_frames = self.ACTION_RELEASE_DELAY

        # 서버와 동일한 타이밍으로 액션 실행
        # 1. ACTION_HOLD_FRAMES 동안 버튼 누름
        for _ in range(self.ACTION_HOLD_FRAMES):
            self._run_single_frame([button])

        # 2. release_frames 동안 대기 (버튼 해제)
        for _ in range(release_frames):
            self._run_single_frame([])

        self.step_count += 1

        # Get state dict (for reward/termination calculation and info)
        state = self._get_state_dict()

        # Add prev_action to state (for expert policies)
        state["prev_action"] = self.last_action if self.last_action else "no_op"

        # Add facing to state (for expert policies)
        state["facing"] = self.last_facing if self.last_facing else "north"

        # Compute reward, terminated, truncated based on state
        reward = self._compute_reward(state)
        terminated = self._is_terminated(state)
        truncated = self._is_truncated(state)

        self.episode_reward += reward

        # Get observation (screenshot) and info (state dict with metadata)
        obs = self._get_obs()
        info = self._get_info(state)

        # Update last_action for next step (store as lowercase)
        self.last_action = button.lower()

        # Update facing based on direction actions
        if self.last_action in ["up", "down", "left", "right"]:
            facing_map = {"up": "north", "down": "south", "left": "west", "right": "east"}
            self.last_facing = facing_map[self.last_action]

        return obs, reward, terminated, truncated, info

    def _run_single_frame(self, buttons: list[str]):
        """
        단일 프레임 실행 (서버의 step_environment와 동일)

        Args:
            buttons: 누를 버튼 리스트 (빈 리스트면 no-op)
        """
        # Emulator의 run_frame_with_buttons 사용
        # 이미 FPS 동적 조정이 내부에 구현되어 있음 (대화 중 4배)
        self.emulator.run_frame_with_buttons(buttons)
        self.total_frames += 1

        # Video recording (프레임 스킵 적용)
        if self.record_video and self.total_frames % self.video_frame_skip == 0:
            screenshot = self.emulator.get_screenshot()
            if screenshot:
                frame = np.array(screenshot)
                # Action 오버레이 추가
                if self.overlay_actions and self.current_action:
                    frame = self._add_action_overlay(frame, self.current_action)
                self.video_frames.append(frame)

        # Area transition 체크 (서버/app.py:482-498과 동일)
        if hasattr(self.emulator, "memory_reader") and self.emulator.memory_reader:
            try:
                transition_detected = self.emulator.memory_reader._check_area_transition()
                if transition_detected:
                    logger.info("Area transition detected")
                    self.emulator.memory_reader.invalidate_map_cache()

                    # 100프레임 대기 (서버와 동일 - 맵 로딩 시간)
                    for _ in range(100):
                        self.emulator.run_frame_with_buttons([])
                        self.total_frames += 1

                        # Video recording for transition frames
                        if self.record_video and self.total_frames % self.video_frame_skip == 0:
                            screenshot = self.emulator.get_screenshot()
                            if screenshot:
                                frame = np.array(screenshot)
                                # Action 오버레이 추가
                                if self.overlay_actions and self.current_action:
                                    frame = self._add_action_overlay(frame, self.current_action)
                                self.video_frames.append(frame)

                    logger.info("Map loading complete (100 frames)")
            except Exception as e:
                logger.warning(f"Area transition check failed: {e}")

    def _add_action_overlay(self, frame: np.ndarray, action: str) -> np.ndarray:
        """
        프레임에 action 정보 오버레이 추가

        Args:
            frame: RGB numpy array (height, width, 3)
            action: Action 문자열 (예: "UP", "A", "B")

        Returns:
            오버레이가 추가된 프레임
        """
        try:
            import cv2
        except ImportError:
            logger.warning("cv2 not available, skipping action overlay")
            return frame

        # 프레임 복사 (원본 변경 방지)
        frame = frame.copy()
        height, width = frame.shape[:2]

        # 폰트 설정
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Action 표시 (하단 좌측, 큰 글씨)
        action_text = f"Action: {action}"
        action_font_scale = 0.5
        action_thickness = 1
        action_color = (255, 255, 255)  # 흰색
        action_pos = (5, height - 10)

        # 텍스트 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(action_text, font, action_font_scale, action_thickness)

        # 배경 박스 (반투명 검정)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (action_pos[0] - 2, action_pos[1] - text_height - 2),
            (action_pos[0] + text_width + 2, action_pos[1] + baseline + 2),
            (0, 0, 0),
            -1,
        )
        # 블렌딩 (70% 투명도)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # 텍스트 그리기
        cv2.putText(
            frame, action_text, action_pos, font, action_font_scale, action_color, action_thickness, cv2.LINE_AA
        )

        # Step 정보 (상단 우측, 작은 글씨)
        step_text = f"Step: {self.step_count}"
        step_font_scale = 0.35
        step_thickness = 1
        step_color = (255, 255, 255)

        (step_width, step_height), step_baseline = cv2.getTextSize(step_text, font, step_font_scale, step_thickness)
        step_pos = (width - step_width - 5, 15)

        # 배경 박스
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (step_pos[0] - 2, step_pos[1] - step_height - 2),
            (step_pos[0] + step_width + 2, step_pos[1] + step_baseline + 2),
            (0, 0, 0),
            -1,
        )
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, step_text, step_pos, font, step_font_scale, step_color, step_thickness, cv2.LINE_AA)

        return frame

    def _get_state_dict(self) -> dict[str, Any]:
        """
        Get comprehensive state dict (for info and reward calculation)

        Returns:
            State dict with player, game, map, etc.
        """
        state = self.emulator.get_comprehensive_state()
        state["llm_formatted"] = format_state_for_llm(state)
        state["step_count"] = self.step_count
        return state

    def _get_obs(self):
        """
        Get observation (screenshot)

        Returns:
            PIL Image (screenshot)
        """
        screenshot = self.emulator.get_screenshot()
        return screenshot

    def _get_completed_milestones(self) -> set:
        """완료된 milestone ID 세트 반환"""
        if not self.enable_milestones or not hasattr(self.emulator, "milestone_tracker"):
            return set()

        completed = set()
        for milestone_id, data in self.emulator.milestone_tracker.milestones.items():
            if data.get("completed", False):
                completed.add(milestone_id)
        return completed

    def _compute_reward(self, obs: dict[str, Any]) -> float:
        """
        보상 계산 (RL용)

        기본 전략:
        - 새로운 milestone 완료: +1.0
        - 새로운 location 방문: +0.1
        - 시간 패널티: -0.001 (매 스텝마다)
        """
        reward = 0.0

        # 시간 패널티 (매 스텝)
        reward -= 0.001

        # Milestone 보상
        if self.enable_milestones:
            current_milestones = self._get_completed_milestones()
            new_milestones = current_milestones - self.prev_milestones

            if new_milestones:
                milestone_reward = len(new_milestones) * 1.0
                reward += milestone_reward
                logger.info(f"New milestones completed: {new_milestones} (+{milestone_reward})")

            self.prev_milestones = current_milestones

        # Location 변경 보상 (새로운 장소 탐험)
        current_location = obs.get("player", {}).get("location")
        if current_location and current_location != self.prev_location:
            reward += 0.1
            logger.debug(f"New location: {current_location} (+0.1)")
            self.prev_location = current_location

        return reward

    def _is_terminated(self, obs: dict[str, Any]) -> bool:
        """
        에피소드 종료 여부

        예: 특정 milestone 달성 시 (예: 첫 번째 체육관 클리어)
        """
        if self.enable_milestones:
            completed = self._get_completed_milestones()
            # 예: FIRST_GYM_COMPLETE milestone 달성 시 종료
            if "FIRST_GYM_COMPLETE" in completed:
                logger.info("Episode terminated: FIRST_GYM_COMPLETE achieved")
                return True

        return False

    def _is_truncated(self, obs: dict[str, Any]) -> bool:
        """
        에피소드 절단 여부 (시간 초과 등)
        """
        # 최대 스텝 수 제한
        max_steps = 10000
        if self.step_count >= max_steps:
            logger.info(f"Episode truncated: max steps ({max_steps}) reached")
            return True

        return False

    def _get_info(self, state: dict[str, Any]) -> dict[str, Any]:
        """
        Get info dict (state dict + metadata)

        Args:
            state: Comprehensive state dict from _get_state_dict()

        Returns:
            Info dict containing full state + metadata
        """
        # Start with the full state dict
        info = state.copy()

        # Add metadata (these override any existing keys in state)
        info["meta"] = {
            "step_count": self.step_count,
            "total_frames": self.total_frames,
            "episode_reward": self.episode_reward,
        }

        # Add milestone info (if not already in state)
        if self.enable_milestones and hasattr(self.emulator, "get_milestones"):
            try:
                if "milestones" not in info:
                    info["milestones"] = self.emulator.get_milestones()
            except Exception as e:
                logger.warning(f"Failed to get milestone info: {e}")

        return info

    def render(self):
        """렌더링"""
        if self.render_mode == "rgb_array":
            screenshot = self.emulator.get_screenshot()
            return np.array(screenshot) if screenshot else None
        elif self.render_mode == "human":
            # TODO: Pygame 등으로 화면 표시
            logger.warning("Human rendering not implemented yet")
            return None
        return None

    def close(self):
        """환경 종료"""
        # Save final episode video if enabled
        if self.record_video and len(self.video_frames) > 0:
            self._save_episode_video()

        if self.emulator:
            self.emulator.stop()
            logger.info("Emulator stopped")

    def save_state(self, path: str):
        """상태 저장"""
        if self.emulator:
            self.emulator.save_state(path)
            logger.info(f"State saved to {path}")

    def load_state(self, path: str):
        """상태 로드"""
        if self.emulator:
            self.emulator.load_state(path)
            logger.info(f"State loaded from {path}")

    def get_action_meanings(self) -> list[str]:
        """각 액션의 의미 반환 (Atari와 유사)"""
        return self.action_map

    def _save_episode_video(self):
        """현재 에피소드의 video 저장 (내부 헬퍼 메서드)"""
        if not self.record_video or len(self.video_frames) == 0:
            return

        import datetime
        import os

        # 파일명 생성: YYYYMMDD_HHMMSS_episode_XXXX.mp4
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_episode_{self.episode_count:04d}.mp4"
        video_path = os.path.join(self.video_dir, filename)

        # Video 저장
        self.save_video(video_path)

        # 통계 출력
        stats = self.get_video_stats()
        logger.info(f"📹 Episode {self.episode_count} video saved: {video_path}")
        logger.info(
            f"   Frames: {stats['frames']}, Duration: {stats['duration_seconds']:.2f}s, "
            f"Steps: {self.step_count}, Reward: {self.episode_reward:.2f}"
        )

    def save_video(self, path: str, fps: Optional[int] = None):
        """
        녹화된 video를 파일로 저장 (mediapy 사용)

        Args:
            path: 저장할 파일 경로 (확장자 포함, 예: "video.mp4")
            fps: Video FPS (None이면 self.video_fps 사용)
        """
        if not self.record_video:
            logger.warning("Video recording is not enabled")
            return

        if len(self.video_frames) == 0:
            logger.warning("No frames to save")
            return

        try:
            import mediapy as media

            # FPS 설정
            save_fps = fps if fps is not None else self.video_fps

            # Video 저장
            logger.info(f"Saving video: {path} ({len(self.video_frames)} frames @ {save_fps} FPS)")
            media.write_video(path, self.video_frames, fps=save_fps)
            logger.info(f"✅ Video saved: {path}")

            return path

        except ImportError:
            logger.error("mediapy not installed. Install with: pip install mediapy")
            # Fallback to OpenCV
            try:
                import cv2

                logger.info("Falling back to OpenCV for video saving")

                height, width = self.video_frames[0].shape[:2]
                save_fps = fps if fps is not None else self.video_fps
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(path, fourcc, float(save_fps), (width, height))

                for frame in self.video_frames:
                    # Convert RGB to BGR for OpenCV
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    out.write(frame_bgr)

                out.release()
                logger.info(f"✅ Video saved (OpenCV): {path}")
                return path

            except Exception as e:
                logger.error(f"Failed to save video: {e}")
                return None

        except Exception as e:
            logger.error(f"Failed to save video: {e}")
            return None

    def get_video_stats(self) -> dict[str, Any]:
        """Video recording 통계 반환"""
        if not self.record_video:
            return {"enabled": False}

        duration = len(self.video_frames) / self.video_fps if self.video_fps > 0 else 0
        return {
            "enabled": True,
            "frames": len(self.video_frames),
            "fps": self.video_fps,
            "duration_seconds": duration,
            "total_game_frames": self.total_frames,
            "frame_skip": self.video_frame_skip,
        }
