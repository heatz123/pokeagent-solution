#!/usr/bin/env python3
"""
Flask web server for training monitoring dashboard
"""

import os
import sys
from flask import Flask, render_template, jsonify, send_file, request
from flask_cors import CORS

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from monitor.utils import (
    get_training_status,
    get_video_files,
    parse_llm_logs,
    get_latest_llm_log_file,
    parse_recent_llm_logs,
    format_llm_interaction,
    get_statistics,
    get_dagger_overview,
    get_dagger_milestones,
    get_dagger_milestone_detail,
    get_expert_policy_code
)


app = Flask(__name__)
CORS(app)  # Enable CORS for API access


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')


@app.route('/api/status')
def api_status():
    """
    Get current training status

    Returns:
        JSON with training status
    """
    try:
        status = get_training_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/statistics')
def api_statistics():
    """
    Get training statistics

    Returns:
        JSON with statistics
    """
    try:
        stats = get_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm_logs')
def api_llm_logs():
    """
    Get LLM interaction logs from recent log files

    Query params:
        limit: Number of logs to return (default 20)

    Returns:
        JSON with list of formatted LLM interactions
    """
    try:
        limit = request.args.get('limit', 20, type=int)
        limit = min(limit, 100)  # Cap at 100

        # Parse recent log files to get all recent interactions
        interactions = parse_recent_llm_logs(limit=limit, num_files=10)

        # Format each interaction for display
        formatted_logs = [format_llm_interaction(i) for i in interactions]

        return jsonify({'logs': formatted_logs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/videos')
def api_videos():
    """
    Get list of training videos

    Returns:
        JSON with list of video files
    """
    try:
        videos = get_video_files()
        return jsonify({'videos': videos})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/videos/<filename>')
def serve_video(filename):
    """
    Serve video file

    Args:
        filename: Video filename (episode_*.mp4)

    Returns:
        Video file
    """
    try:
        videos_dir = os.path.join(os.path.dirname(__file__), "..", "videos")
        video_path = os.path.join(videos_dir, filename)

        if not os.path.exists(video_path):
            return jsonify({'error': 'Video not found'}), 404

        # Security check: ensure filename is safe
        if '..' in filename or filename.startswith('/'):
            return jsonify({'error': 'Invalid filename'}), 400

        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/llm_log_raw')
def api_llm_log_raw():
    """
    Get raw LLM log file content

    Returns:
        Raw JSONL content
    """
    try:
        llm_log_file = get_latest_llm_log_file()
        if not llm_log_file:
            return jsonify({'error': 'No log file found'}), 404

        with open(llm_log_file, 'r', encoding='utf-8') as f:
            content = f.read()

        return content, 200, {'Content-Type': 'application/x-ndjson'}
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ============================================================================
# DAgger Pipeline API Endpoints
# ============================================================================

@app.route('/dagger')
def dagger_dashboard():
    """DAgger pipeline dashboard page"""
    return render_template('dagger_dashboard.html')


@app.route('/api/dagger/overview')
def api_dagger_overview():
    """
    Get DAgger pipeline overview

    Returns:
        JSON with pipeline progress
    """
    try:
        overview = get_dagger_overview()
        return jsonify(overview)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dagger/milestones')
def api_dagger_milestones():
    """
    Get list of all milestones with their status

    Returns:
        JSON with list of milestones
    """
    try:
        milestones = get_dagger_milestones()
        return jsonify({'milestones': milestones})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dagger/milestone/<milestone_id>')
def api_dagger_milestone_detail(milestone_id):
    """
    Get detailed information for a specific milestone

    Args:
        milestone_id: Milestone ID

    Returns:
        JSON with milestone details
    """
    try:
        detail = get_dagger_milestone_detail(milestone_id)
        if detail is None:
            return jsonify({'error': 'Milestone not found'}), 404
        return jsonify(detail)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/dagger/milestone/<milestone_id>/code')
def api_dagger_milestone_code(milestone_id):
    """
    Get expert policy Python code for a milestone

    Args:
        milestone_id: Milestone ID

    Returns:
        JSON with code
    """
    try:
        code = get_expert_policy_code(milestone_id)
        if code is None:
            return jsonify({'error': 'Policy code not found'}), 404
        return jsonify({'code': code})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/dagger_videos/<milestone_id>/<filename>')
def serve_dagger_video(milestone_id, filename):
    """
    Serve DAgger video file

    Args:
        milestone_id: Milestone ID
        filename: Video filename

    Returns:
        Video file
    """
    try:
        # Security check: ensure filename is safe
        if '..' in filename or filename.startswith('/') or '..' in milestone_id or milestone_id.startswith('/'):
            return jsonify({'error': 'Invalid filename'}), 400

        # Check in both expert_eval_videos and dagger_videos
        base_dir = os.path.join(os.path.dirname(__file__), "..", "dagger_pipeline_results")

        possible_paths = [
            os.path.join(base_dir, "expert_eval_videos", milestone_id, filename),
            os.path.join(base_dir, "dagger_videos", milestone_id, filename)
        ]

        video_path = None
        for path in possible_paths:
            if os.path.exists(path):
                video_path = path
                break

        if not video_path:
            return jsonify({'error': 'Video not found'}), 404

        return send_file(video_path, mimetype='video/mp4')
    except Exception as e:
        return jsonify({'error': str(e)}), 500


def main():
    """Run Flask server"""
    import argparse

    parser = argparse.ArgumentParser(description='Training Monitor Dashboard')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Training Monitor Dashboard")
    print(f"{'='*70}")
    print(f"URL: http://{args.host}:{args.port}")
    print(f"Debug: {args.debug}")
    print(f"{'='*70}\n")

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug
    )


if __name__ == '__main__':
    main()
