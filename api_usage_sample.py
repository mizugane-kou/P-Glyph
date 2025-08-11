import requests
import base64
import sys
import os

# P-Glyph側で設定したAPIのURL
BASE_URL = "http://127.0.0.1:8756"

def get_current_glyph():
    """P-Glyphで現在選択されているグリフの情報を取得する"""
    try:
        response = requests.get(f"{BASE_URL}/api/glyph/current")
        response.raise_for_status()  # エラーがあれば例外を発生させる
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error: P-Glyphに接続できません。アプリは起動していますか？\n({e})")
        return None

def send_image_to_glyph(file_path: str, as_reference: bool = False):
    """指定された画像ファイルをP-Glyphに送信する"""
    if not os.path.exists(file_path):
        print(f"Error: ファイルが見つかりません: {file_path}")
        return

    # 画像をバイナリとして読み込み、Base64にエンコード
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    payload = {"image_base64": encoded_string}
    
    # 送信先のエンドポイントを決定
    endpoint = "/api/glyph/reference_image" if as_reference else "/api/glyph/image"
    url = BASE_URL + endpoint
    
    print(f"画像を '{url}' に送信中...")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("送信成功！")
        print("サーバーからの応答:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error: 送信に失敗しました。\n({e})")
        if e.response:
            print("サーバーエラー詳細:", e.response.text)

if __name__ == "__main__":
    # --- 現在のグリフ情報を取得するデモ ---
    print("--- 1. 現在のグリフ情報を取得 ---")
    current_info = get_current_glyph()
    if current_info:
        current_char = current_info.get("character", "N/A")
        if current_char:
            print(f"P-Glyphは現在「{current_char}」を編集中です。")
        else:
            print("P-Glyphでグリフが選択されていません。")
    print("-" * 30 + "\n")


    # --- 画像を送信するデモ ---
    print("--- 2. グリフ画像を送信 ---")
    # コマンドライン引数から画像パスを取得
    # 例: python send_image.py my_awesome_glyph.png
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        
        # コマンドライン引数に "--ref" があれば下書きとして送信
        is_ref = "--ref" in sys.argv or "--reference" in sys.argv

        send_image_to_glyph(image_path, as_reference=is_ref)
    else:
        print("画像を送信するには、コマンドライン引数に画像ファイルのパスを指定してください。")
        print("例: python send_image.py my_glyph.png")
        print("下書きとして送信する場合: python send_image.py my_reference.png --ref")