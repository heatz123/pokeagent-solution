#!/usr/bin/env python3
"""
Utility functions for training monitor dashboard
"""

import os
import json
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional


def get_latest_llm_log_file() -> Optional[str]:
    """Get the most recent LLM log file (deprecated, use get_recent_llm_log_files)"""
    log_dir = os.path.join(os.path.dirname(__file__), "..", "llm_logs")
    log_files = glob.glob(os.path.join(log_dir, "llm_log_*.jsonl"))

    if not log_files:
        return None

    # Sort by modification time
    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files[0]


def get_recent_llm_log_files(num_files: int = 10) -> List[str]:
    """
    Get the most recent LLM log files

    Args:
        num_files: Number of recent log files to return

    Returns:
        List of log file paths sorted by modification time (newest first)
    """
    log_dir = os.path.join(os.path.dirname(__file__), "..", "llm_logs")
    log_files = glob.glob(os.path.join(log_dir, "llm_log_*.jsonl"))

    if not log_files:
        return []

    # Sort by modification time (newest first)
    log_files.sort(key=os.path.getmtime, reverse=True)
    return log_files[:num_files]


def parse_llm_logs(log_file: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Parse LLM log file and return list of interactions

    Args:
        log_file: Path to JSONL log file
        limit: Maximum number of interactions to return (most recent)

    Returns:
        List of interaction dictionaries
    """
    if not os.path.exists(log_file):
        return []

    interactions = []

    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    interaction = json.loads(line)
                    interactions.append(interaction)
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        print(f"Error parsing LLM logs: {e}")
        return []

    # Return most recent interactions
    return interactions[-limit:]


def parse_recent_llm_logs(limit: int = 20, num_files: int = 10) -> List[Dict[str, Any]]:
    """
    Parse multiple recent LLM log files and return list of interactions

    Args:
        limit: Maximum number of interactions to return (most recent)
        num_files: Number of recent log files to check

    Returns:
        List of interaction dictionaries sorted by timestamp (newest first)
    """
    log_files = get_recent_llm_log_files(num_files)

    if not log_files:
        return []

    all_interactions = []

    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        interaction = json.loads(line)
                        # Only include actual interactions, not session_start entries
                        if interaction.get('type') == 'interaction':
                            all_interactions.append(interaction)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error parsing LLM log {log_file}: {e}")
            continue

    # Sort by timestamp (newest first)
    all_interactions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)

    # Return most recent interactions
    return all_interactions[:limit]


def get_training_status() -> Dict[str, Any]:
    """
    Get current training status by reading recent logs

    Returns:
        Status dictionary with training info
    """
    status = {
        'running': False,
        'current_milestone': None,
        'attempt': None,
        'total_interactions': 0,
        'last_update': None,
        'recent_logs': []
    }

    # Check LLM logs from recent files
    interactions = parse_recent_llm_logs(limit=100, num_files=10)
    status['total_interactions'] = len(interactions)

    if interactions:
        # interactions are already sorted by timestamp (newest first)
        latest = interactions[0]
        status['last_update'] = latest.get('timestamp')

        # Check if training is recent (within last 5 minutes)
        try:
            timestamp = datetime.fromisoformat(latest['timestamp'].replace('Z', '+00:00'))
            now = datetime.now(timestamp.tzinfo)
            elapsed = (now - timestamp).total_seconds()
            status['running'] = elapsed < 300  # 5 minutes
        except:
            pass

        # Get recent log entries for display (first 5, since sorted newest first)
        for interaction in interactions[:5]:
            log_entry = {
                'timestamp': interaction.get('timestamp'),
                'type': interaction.get('interaction_type'),
                'duration': interaction.get('duration'),
                'model': interaction.get('model_info', {}).get('model')
            }
            status['recent_logs'].append(log_entry)

    return status


def get_video_files() -> List[Dict[str, Any]]:
    """
    Get list of training video files

    Returns:
        List of video file info dicts
    """
    videos_dir = os.path.join(os.path.dirname(__file__), "..", "videos")
    video_files = glob.glob(os.path.join(videos_dir, "*.mp4"))

    videos = []
    for video_path in video_files:
        filename = os.path.basename(video_path)
        stat = os.stat(video_path)

        videos.append({
            'filename': filename,
            'size': stat.st_size,
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'path': video_path
        })

    # Sort by modification time (newest first)
    videos.sort(key=lambda x: x['modified'], reverse=True)

    return videos


def format_llm_interaction(interaction: Dict[str, Any]) -> Dict[str, Any]:
    """
    Format LLM interaction for display

    Args:
        interaction: Raw interaction dict from JSONL

    Returns:
        Formatted dict with display-friendly fields
    """
    formatted = {
        'timestamp': interaction.get('timestamp', 'Unknown'),
        'type': interaction.get('interaction_type', 'unknown'),
        'duration': round(interaction.get('duration', 0), 2),
        'model': interaction.get('model_info', {}).get('model', 'unknown'),
        'prompt_preview': '',
        'response_preview': '',
        'full_prompt': '',
        'full_response': ''
    }

    # Get prompt
    prompt = interaction.get('prompt', '')
    if prompt:
        formatted['full_prompt'] = prompt
        # Preview: first 200 chars
        formatted['prompt_preview'] = prompt[:200] + ('...' if len(prompt) > 200 else '')

    # Get response
    response = interaction.get('response', '')
    if response:
        formatted['full_response'] = response
        # Preview: first 200 chars
        formatted['response_preview'] = response[:200] + ('...' if len(response) > 200 else '')

    return formatted


def get_statistics() -> Dict[str, Any]:
    """
    Calculate training statistics from recent logs

    Returns:
        Statistics dictionary
    """
    stats = {
        'total_llm_calls': 0,
        'total_duration': 0.0,
        'avg_duration': 0.0,
        'total_videos': 0,
        'code_generations': 0,
        'curriculum_code_generations': 0,
        'policy_executions': 0
    }

    # LLM statistics from recent log files
    interactions = parse_recent_llm_logs(limit=1000, num_files=10)
    stats['total_llm_calls'] = len(interactions)

    if interactions:
        total_duration = sum(i.get('duration', 0) for i in interactions)
        stats['total_duration'] = round(total_duration, 2)
        stats['avg_duration'] = round(total_duration / len(interactions), 2)

        # Count by type
        for interaction in interactions:
            interaction_type = interaction.get('interaction_type', '')
            if interaction_type == 'code_generation':
                stats['code_generations'] += 1
            elif interaction_type == 'curriculum_code_generation':
                stats['curriculum_code_generations'] += 1
            elif interaction_type == 'policy_execution':
                stats['policy_executions'] += 1

    # Video count
    videos = get_video_files()
    stats['total_videos'] = len(videos)

    return stats