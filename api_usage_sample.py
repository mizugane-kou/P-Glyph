import requests
import base64
import sys
import os
import argparse

# P-Glyph側で設定したAPIのURL
BASE_URL = "http://127.0.0.1:8756"

def get_current_glyph():
    """P-Glyphで現在選択されているグリフの情報を取得する"""
    print("--- 現在のグリフ情報を取得 ---")
    try:
        response = requests.get(f"{BASE_URL}/api/glyph/current")
        response.raise_for_status()  # エラーがあれば例外を発生させる
        current_info = response.json()
        if current_info:
            current_char = current_info.get("character", "N/A")
            if current_char:
                print(f"P-Glyphは現在「{current_char}」を編集中です。")
                print("詳細:", current_info)
            else:
                print("P-Glyphでグリフが選択されていません。")
        return current_info
    except requests.exceptions.RequestException as e:
        print(f"Error: P-Glyphに接続できません。アプリは起動していますか？\n({e})")
        return None

def send_image_to_glyph(file_path: str, as_reference: bool = False):
    """指定された画像ファイルをP-Glyphに送信する"""
    print(f"--- 画像を {'下書きとして' if as_reference else 'グリフとして'} 送信 ---")
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
    
    print(f"画像 '{file_path}' を '{url}' に送信中...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("送信成功！")
        print("サーバーからの応答:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error: 送信に失敗しました。\n({e})")
        if hasattr(e, 'response') and e.response:
            print("サーバーエラー詳細:", e.response.text)

def select_glyph(character: str, is_vrt2: bool = False, is_pua: bool = False):
    """P-Glyphで編集するグリフを選択する"""
    print(f"--- グリフ選択をリクエスト ---")
    payload = {
        "character": character,
        "is_vrt2": is_vrt2,
        "is_pua": is_pua,
    }
    url = f"{BASE_URL}/api/glyph/select"

    glyph_type = "PUA" if is_pua else ("縦書き" if is_vrt2 else "標準")
    print(f"グリフ '{character}' ({glyph_type}) の選択をリクエスト中...")

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        print("リクエスト成功！")
        print("サーバーからの応答:", response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error: リクエストに失敗しました。\n({e})")
        if hasattr(e, 'response') and e.response:
            print("サーバーエラー詳細:", e.response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="P-Glyph API Test Client. P-Glyphアプリケーションを起動した状態で使用してください。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--info', 
        action='store_true', 
        help='現在P-Glyphで選択されているグリフの情報を取得します。'
    )
    parser.add_argument(
        '--select', 
        type=str, 
        metavar='CHAR',
        help='指定した文字のグリフをP-Glyphで選択させます。\n例: --select あ'
    )
    parser.add_argument(
        '--vrt2', 
        action='store_true', 
        help='--select と一緒に使い、縦書きグリフを選択します。'
    )
    parser.add_argument(
        '--pua', 
        action='store_true', 
        help='--select と一緒に使い、私用領域(PUA)グリフを選択します。'
    )
    parser.add_argument(
        '--send', 
        type=str, 
        metavar='IMAGE_PATH',
        help='指定した画像ファイルを現在のグリフに送信します。\n例: --send my_glyph.png'
    )
    parser.add_argument(
        '--ref', 
        action='store_true', 
        help='--send と一緒に使い、画像を下書きとして送信します。'
    )

    args = parser.parse_args()

    # 引数が何も指定されなかった場合はヘルプを表示
    if not len(sys.argv) > 1:
        parser.print_help()
        sys.exit(0)

    if args.info:
        get_current_glyph()
        print("-" * 30)

    if args.select:
        if args.pua and args.vrt2:
            print("Error: --pua と --vrt2 は同時に指定できません。")
        else:
            select_glyph(args.select, is_vrt2=args.vrt2, is_pua=args.pua)
        print("-" * 30)

    if args.send:
        send_image_to_glyph(args.send, as_reference=args.ref)
        print("-" * 30)
