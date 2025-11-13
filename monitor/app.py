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
    get_statistics
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
