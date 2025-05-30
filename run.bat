@echo off
REM 仮想環境をアクティベート（venvフォルダが同じディレクトリにある前提）
call venv\Scripts\activate.bat

REM Pythonスクリプトを実行（スクリプト名を直接指定）
start "" pythonw.exe main.py

REM pause