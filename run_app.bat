@echo off
cd /d "%~dp0"
set HF_HUB_DISABLE_SYMLINKS_WARNING=1
set TRANSFORMERS_VERBOSITY=error
echo Starting Ontology Kitchen on http://localhost:8000 ...
".venv\Scripts\python.exe" server.py
pause
