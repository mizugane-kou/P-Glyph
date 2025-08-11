@echo off
chcp 65001 > nul 
REM venvディレクトリが存在しなければ作成
if not exist venv (
    python -m venv venv
)

REM 仮想環境を有効化してpipアップグレード、ライブラリインストール
call venv\Scripts\activate
pip install --upgrade pip
pip install PySide6 opencv-python scikit-image scipy Pillow svgwrite fontTools ufoLib2 fontmake fastapi "uvicorn[standard]" shapely


echo --------------------------------------
echo ライブラリのインストールが完了しました
echo --------------------------------------
pause


