# Training Monitor Dashboard

Web-based dashboard for monitoring Pokemon agent training in real-time.

## Features

- **Real-time Status**: Shows if training is currently running
- **Statistics**: Total LLM calls, code generations, duration stats, video count
- **LLM Logs**: Interactive viewer for LLM prompts and responses
- **Video Player**: View training videos directly in browser
- **Auto-refresh**: Updates every 5 seconds

## Installation

Required packages:
```bash
pip install flask flask-cors
```

## Usage

### Start the Dashboard

```bash
# From project root
python3 monitor/app.py

# Or with custom host/port
python3 monitor/app.py --host 0.0.0.0 --port 8080

# Enable debug mode
python3 monitor/app.py --debug
```

### Access the Dashboard

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

### Use with Training

In one terminal, start training:
```bash
python3 milestone_trainer.py --config milestone_config.json --milestone LITTLEROOT_TOWN --agent --video --model gpt-5
```

In another terminal, start the dashboard:
```bash
python3 monitor/app.py
```

The dashboard will automatically detect and display:
- Training status
- LLM interaction logs from `llm_logs/`
- Training videos from `videos/`

## API Endpoints

The dashboard provides the following API endpoints:

- `GET /` - Main dashboard HTML
- `GET /api/status` - Current training status (JSON)
- `GET /api/statistics` - Training statistics (JSON)
- `GET /api/llm_logs?limit=20` - LLM interaction logs (JSON)
- `GET /api/videos` - List of video files (JSON)
- `GET /videos/<filename>` - Serve video file

## Architecture

```
monitor/
├── app.py              # Flask server
├── utils.py            # Log parsing utilities
├── templates/
│   └── dashboard.html  # Main UI
└── README.md          # This file
```

## Notes

- Dashboard is read-only (does not modify logs or training)
- LLM logger format (JSONL) is unchanged
- Dashboard runs independently from training
- No authentication/authorization (intended for local use)
