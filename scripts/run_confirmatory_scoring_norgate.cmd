@echo off
setlocal
set "ROOT=%USERPROFILE%\Downloads\video_research"
set "PY=%USERPROFILE%\Downloads\T1forecasting\.venv311\Scripts\python.exe"
set "REPO=%ROOT%\video-quality-reward-modeling"
set "VIDEOS=%USERPROFILE%\Downloads\datasets\ucf101\UCF101\UCF-101"
cd /d "%REPO%"
"%PY%" scripts\score_intervention_pairs.py ^
  --pairs results\confirmatory\intervention_pairs.jsonl ^
  --video-root "%VIDEOS%" ^
  --output results\confirmatory\intervention_scores.jsonl ^
  1> results\confirmatory\scoring.stdout.log ^
  2> results\confirmatory\scoring.stderr.log
exit /b %ERRORLEVEL%
