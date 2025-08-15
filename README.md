# P-Glyph

P-Glyphは手軽に和文フォントを制作するために開発したソフトです。細かい設定なしにGUIで文字を書いて簡単にフォントとして書き出せます。


## 使い方
### 要件
実行にはPythonが必要です。 Pythonがインストールされていない場合はインストールしてください (Microsoftストアから行うと簡単です。)  
他のバージョンでもおそらく動作すると思いますがPython3.10で開発を行っています。
Window以外のOSでの動作は保証しません。

### インストール
[リリース](https://github.com/mizugane-kou/P-Glyph/releases)から最新版のZIPをDLして保存し、解凍して任意のフォルダに保存してください。  
install.batをダブルクリックで実行すると実行環境の構築とライブラリのインストールが行われます。インストール後run.batの実行によってソフトが起動します。


<img src="スクリーンショット 2025-08-11 223015.png" width="512">

### アップデート
アップデートを行うときはインストールフォルダ内のコードをP-Glyph_VX.XX.zipの中身に完全に置き換え、install.batを再び実行し環境を新規作成してください。

## ライセンス

このソフトウェアのコード部分は MIT ライセンスのもとで提供されます。

ただし、[KanjiVG datasets](https://github.com/KanjiVG/kanjivg/blob/master/COPYING) に基づく [データ](https://github.com/yagays/kanjivg-radical/tree/master/data) が [同梱](https://github.com/mizugane-kou/P-Glyph/tree/main/data) されており、これは **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)** に従います。

また、このソフトウェアは以下の外部ライブラリに依存しています。

| ライブラリ        | ライセンス          |
| ------------ | -------------- |
| PySide6      | LGPL v3        |
| Pillow       | PILライセンス（MIT系） |
| numpy        | BSD            |
| scikit-image | BSD            |
| scipy        | BSD            |
| svgwrite     | MIT            |
| fontTools    | MIT            |
| ufoLib2      | MIT            |

これらのライブラリの詳細なライセンスについては、それぞれの公式リポジトリをご参照ください。


