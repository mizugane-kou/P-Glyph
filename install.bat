@echo off
chcp 65001 > nul 
REM venvディレクトリが存在しなければ作成
if not exist venv (
    python -m venv venv
)

REM 仮想環境を有効化してpipアップグレード、ライブラリインストール
call venv\Scripts\activate
pip install --upgrade pip
pip install PySide6 Pillow numpy scikit-image scipy svgwrite fonttools ufolib2 fontmake opencv-python shapely


echo --------------------------------------
echo ライブラリのインストールが完了しました
echo --------------------------------------
pause

