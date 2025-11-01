#!/usr/bin/env python3
"""
Code-generating client that uses CodeAgent to play Pokemon Emerald.
Compatible with the existing server/client architecture.
"""

import sys
import os
import argparse
import requests
import time
import base64
import io
from PIL import Image

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from agent.code_agent import CodeAgent


def main():
    """Main entry point for code-generating client"""
    parser = argparse.ArgumentParser(description="Pokemon Code Agent Client")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between steps (seconds)")
    args = parser.parse_args()

    server_url = f"http://localhost:{args.port}"

    print("=" * 60)
    print("🤖 Pokemon Code Agent")
    print("=" * 60)
    print(f"📡 Server: {server_url}")
    print(f"⏱️  Delay: {args.delay}s")
    print()

    # Initialize agent
    try:
        agent = CodeAgent()
        print("✅ CodeAgent initialized")
    except Exception as e:
        print(f"❌ Failed to initialize agent: {e}")
        return 1

    # Check server connection
    try:
        response = requests.get(f"{server_url}/health", timeout=5)
        if response.status_code == 200:
            print(f"✅ Server connected")
        else:
            print(f"❌ Server returned status {response.status_code}")
            return 1
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print(f"   Make sure server is running: python server/app.py --port {args.port}")
        return 1

    print()
    print("🚀 Starting agent loop...")
    print("-" * 60)

    step = 0

    try:
        while True:
            step += 1

            # 1. Get state from server
            try:
                response = requests.get(f"{server_url}/state", timeout=5)
                if response.status_code != 200:
                    print(f"❌ Failed to get state: {response.status_code}")
                    time.sleep(args.delay)
                    continue

                state_data = response.json()
            except requests.exceptions.RequestException as e:
                print(f"❌ Connection error: {e}")
                time.sleep(args.delay)
                continue

            # 2. Convert to game_state format
            screenshot_base64 = state_data.get("visual", {}).get("screenshot_base64", "")
            if screenshot_base64:
                img_data = base64.b64decode(screenshot_base64)
                screenshot = Image.open(io.BytesIO(img_data))
            else:
                screenshot = None

            game_state = {
                'frame': screenshot,
                'player': state_data.get('player', {}),
                'game': state_data.get('game', {}),
                'map': state_data.get('map', {}),
                'visual': state_data.get('visual', {}),
                'milestones': state_data.get('milestones', {}),
                'step_number': state_data.get('step_number', 0),
                'status': state_data.get('status', ''),
                'action_queue_length': state_data.get('action_queue_length', 0)
            }

            # 3. Agent step (generates code, executes, returns action)
            result = agent.step(game_state)
            action = result['action']

            # 4. Send action to server
            try:
                action_list = [action] if isinstance(action, str) else action

                response = requests.post(
                    f"{server_url}/action",
                    json={"buttons": action_list},
                    timeout=5
                )

                if response.status_code == 200:
                    # Get current location for display
                    location = game_state.get('player', {}).get('location', 'Unknown')

                    # Simple output (milestone info is printed by CodeAgent)
                    print(f"🎮 Step {step}: {action} | {location}")
                else:
                    print(f"⚠️ Step {step}: {action} (server error: {response.status_code})")

            except requests.exceptions.RequestException as e:
                print(f"❌ Action send error: {e}")

            # 5. Delay before next step
            time.sleep(args.delay)

    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user")
        print("👋 Goodbye!")
        return 0

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
