#!/bin/bash
cd /home/gh_agent/gh_agent
source venv/bin/activate
python -m uvicorn app.main:app --host 0.0.0.0 --port 18234
