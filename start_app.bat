@echo off
echo ========================================================
echo        CreditPulse Sovereign Risk - Web Console
echo ========================================================
echo.
echo Starting Python FastAPI ML Engine on port 8000...
start cmd /k ".\venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

timeout /t 2 /nobreak > nul

echo Starting React Vite Frontend on port 5173...
cd frontend
start cmd /k "npm run dev -- --host 127.0.0.1"

echo.
echo ========================================================
echo Web App running at:   http://127.0.0.1:5173
echo Backend API Docs at:  http://127.0.0.1:8000/docs
echo ========================================================
pause
