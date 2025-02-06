@echo off
cd venv\Scripts
call activate
cd ..\..
python -m kohaku_nai.server
