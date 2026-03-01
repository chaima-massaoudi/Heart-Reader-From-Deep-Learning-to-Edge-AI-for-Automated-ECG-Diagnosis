"""Launch the Heart Reader web server."""
import os
import sys

# Ensure we're in the heart_reader directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
uvicorn.run("frontend.app:app", host="0.0.0.0", port=8000, reload=False)
