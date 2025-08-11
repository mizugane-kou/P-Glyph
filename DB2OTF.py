import argparse
import io
import os
import re
import sys
import shutil
import sqlite3
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
import multiprocessing
import traceback

from PIL import Image
import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu
from scipy.signal import savgol_filter
import svgwrite
from shapely.geometry import Polygon

from fontTools.svgLib import SVGPath
from fontTools.pens.pointPen import SegmentToPointPen
from fontTools.misc.transform import Transform
from fontTools.ttLib import TTFont
import ufoLib2

# --- 定数 ---
TMP_DIR_BASE_NAME = "tmp"
IMG_SUBDIR_NAME = "img"
SVG_SUBDIR_NAME = "svg"
UFO_SUBDIR_NAME = "ufo"
SETTING_FONT_NAME = "font_name"
SETTING_FONT_ASCII_NAME = "font_ascii_name"
SETTING_FONT_WEIGHT = "font_weight"
SETTING_ROTATED_VRT2_CHARS = "rotated_vrt2_chars"
SETTING_COPYRIGHT_INFO = "copyright_info"
SETTING_LICENSE_INFO = "license_info"
DEFAULT_FONT_NAME = "MyNewFont"
DEFAULT_FONT_ASCII_NAME = "MyNewFont"
DEFAULT_FONT_WEIGHT = "Regular"
DEFAULT_COPYRIGHT_INFO = ""
DEFAULT_LICENSE_INFO = ""
DEFAULT_ASCENDER_HEIGHT = 900
DEFAULT_ADVANCE_WIDTH = 1000
UNITS_PER_EM = 1000
DEFAULT_DESCENDER = -100

# --- DBヘルパークラス ---
class FontBuildDBHelper:
    def __init__(self, db_path_str: str):
        self.db_path = Path(db_path_str)
        if not self.db_path.exists():
            raise FileNotFoundError(f"データベースファイルが見つかりません: {self.db_path}")

    def _execute_query(self, query, params=None):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()
        return rows

    def get_font_name(self) -> str:
        row = self._execute_query("SELECT value FROM project_settings WHERE key = ?", (SETTING_FONT_NAME,))
        return row[0]['value'] if row and row[0]['value'] else DEFAULT_FONT_NAME

    def get_font_ascii_name(self) -> str:
        ascii_name_row = self._execute_query("SELECT value FROM project_settings WHERE key = ?", (SETTING_FONT_ASCII_NAME,))
        ascii_name = ascii_name_row[0]['value'] if ascii_name_row and ascii_name_row[0]['value'] else None
        if not ascii_name:
            display_name = self.get_font_name()
            generated_name = re.sub(r'[^a-zA-Z0-9_-]', '', display_name)
            return generated_name if generated_name else DEFAULT_FONT_ASCII_NAME
        return ascii_name

    def get_font_weight(self) -> str:
        row = self._execute_query("SELECT value FROM project_settings WHERE key = ?", (SETTING_FONT_WEIGHT,))
        return row[0]['value'] if row and row[0]['value'] else DEFAULT_FONT_WEIGHT

    def get_copyright_info(self) -> str:
        row = self._execute_query("SELECT value FROM project_settings WHERE key = ?", (SETTING_COPYRIGHT_INFO,))
        return row[0]['value'] if row and row[0]['value'] is not None else DEFAULT_COPYRIGHT_INFO

    def get_license_info(self) -> str:
        row = self._execute_query("SELECT value FROM project_settings WHERE key = ?", (SETTING_LICENSE_INFO,))
        return row[0]['value'] if row and row[0]['value'] is not None else DEFAULT_LICENSE_INFO

    def get_ascender_height(self) -> int:
        return DEFAULT_ASCENDER_HEIGHT

    def get_all_standard_glyphs_data(self) -> list:
        return self._execute_query("SELECT character, unicode_val, image_data, advance_width FROM glyphs WHERE image_data IS NOT NULL")

    def get_rotated_vrt2_chars(self) -> list:
        row = self._execute_query("SELECT value FROM project_settings WHERE key = ?", (SETTING_ROTATED_VRT2_CHARS,))
        return list(row[0]['value']) if row and row[0]['value'] else []

    def get_glyph_data_for_char(self, character: str) -> tuple or None:
        row = self._execute_query("SELECT image_data, advance_width FROM glyphs WHERE character = ? AND image_data IS NOT NULL", (character,))
        return (row[0]['image_data'], row[0]['advance_width']) if row else None

    def get_all_non_rotated_vrt2_glyphs_data(self) -> list:
        return self._execute_query("SELECT character, image_data FROM vrt2_glyphs WHERE image_data IS NOT NULL")

    def get_all_pua_glyphs_data(self) -> list:
        try:
            return self._execute_query("SELECT character, unicode_val, image_data, advance_width FROM pua_glyphs WHERE image_data IS NOT NULL")
        except sqlite3.OperationalError:
            print("  情報: pua_glyphs テーブルが見つかりませんでした。PUAグリフのエクスポートをスキップします。")
            return []

    def get_advance_width_for_char(self, character: str) -> int:
        row = self._execute_query("SELECT advance_width FROM glyphs WHERE character = ?", (character,))
        return row[0]['advance_width'] if row and row[0]['advance_width'] is not None else DEFAULT_ADVANCE_WIDTH

# --- 画像からSVGへの変換ロジック ---
def smooth_contour_for_svg(contour, window_length=6, polyorder=3):
    if len(contour) < window_length:
        return contour
    x_coords = savgol_filter(contour[:, 1], window_length, polyorder)
    y_coords = savgol_filter(contour[:, 0], window_length, polyorder)
    return np.column_stack([y_coords, x_coords])

def compute_depth(polygon, all_polygons):
    return sum(1 for other in all_polygons if other != polygon and other.contains(polygon))

def polygon_to_path_data(polygon):
    coords = list(polygon.exterior.coords)
    d = f"M {coords[0][0]:.2f} {coords[0][1]:.2f}"
    for x, y in coords[1:]:
        d += f" L {x:.2f} {y:.2f}"
    d += " Z"
    return d

def image_to_smooth_svg(image_pil: Image.Image, svg_path_str: str, image_width: int, image_height: int):
    if image_pil.mode != 'L':
        image_pil_gray = image_pil.convert('L')
    else:
        image_pil_gray = image_pil
    
    image_np = np.array(image_pil_gray)
    
    binary = np.zeros_like(image_np, dtype=bool) 
    try:
        inverted_image_np = 255 - image_np 
        thresh_val = threshold_otsu(inverted_image_np)
        binary = image_np < (255 - thresh_val)
    except Exception:
        unique_colors = np.unique(image_np)
        if len(unique_colors) == 1 and unique_colors[0] < 128:
            binary = np.ones_like(image_np, dtype=bool) 
            
    raw_contours = measure.find_contours(binary, 0.5)

    dwg = svgwrite.Drawing(svg_path_str, size=(f"{image_width}px", f"{image_height}px"), 
                           viewBox=f"0 0 {image_width} {image_height}")

    if not raw_contours:
        dwg.save() 
        return

    simplified_contours = []
    for contour in raw_contours:
        if len(contour) < 5: continue
        smoothed = smooth_contour_for_svg(contour, window_length=13, polyorder=3)
        simplified = measure.approximate_polygon(smoothed, tolerance=0.5)
        if len(simplified) >= 3:
            simplified_contours.append(simplified[:, ::-1])

    if not simplified_contours:
        dwg.save()
        return

    polygons = [Polygon(c) for c in simplified_contours]
    
    corrected_polygons = []
    for poly in polygons:
        depth = compute_depth(poly, polygons)
        should_be_ccw = (depth % 2 == 0)
        
        if poly.exterior.is_ccw != should_be_ccw:
            corrected_poly = Polygon(list(poly.exterior.coords)[::-1])
            corrected_polygons.append(corrected_poly)
        else:
            corrected_polygons.append(poly)

    path_data_list = [polygon_to_path_data(p) for p in corrected_polygons]
    
    if path_data_list:
        final_path_d = " ".join(path_data_list)
        dwg.add(dwg.path(d=final_path_d, fill='black', stroke='none'))
    
    dwg.save()

# --- フォントビルドステップ ---
def step1_export_images_from_db(db_helper: FontBuildDBHelper, img_output_dir: Path):
    print("\nステップ1: DBからグリフ画像をエクスポート中...")
    if img_output_dir.exists():
        shutil.rmtree(img_output_dir)
    img_output_dir.mkdir(parents=True, exist_ok=True)

    print("  標準グリフをエクスポート...")
    std_glyphs = db_helper.get_all_standard_glyphs_data()
    for char_str, unicode_val_db, image_data, _ in std_glyphs:
        if char_str == ".notdef":
            img_name = "_notdef.png"
        elif unicode_val_db is not None and unicode_val_db != -1 :
            img_name = f"uni{unicode_val_db:04X}.png"
        else:
            safe_char_name = re.sub(r'[^\w-]', '_', char_str)
            img_name = f"{safe_char_name}.png"
            print(f"    警告: '{char_str}' は .notdef でも有効なUnicode値でもありません。'{img_name}'として保存します。")

        img_path = img_output_dir / img_name
        try:
            with open(img_path, "wb") as f:
                f.write(image_data)
        except IOError as e:
            print(f"    エラー: {img_name} の保存に失敗 - {e}")
    print(f"  {len(std_glyphs)}個の標準グリフ画像をエクスポートしました。")

    print("  回転vrt2グリフをエクスポート...")
    rotated_chars = db_helper.get_rotated_vrt2_chars()
    count_rotated = 0
    skipped_rotated_no_base = 0
    for char_str in rotated_chars:
        glyph_data = db_helper.get_glyph_data_for_char(char_str) 
        if glyph_data:
            image_data_rot, _ = glyph_data
            unicode_val_rot = ord(char_str)
            img_name_rot = f"uni{unicode_val_rot:04X}.vert.png"
            img_path_rot = img_output_dir / img_name_rot
            try:
                img = Image.open(io.BytesIO(image_data_rot))
                img_rotated = img.rotate(-90, expand=True) 
                img_rotated.save(img_path_rot, "PNG")
                count_rotated +=1
            except Exception as e:
                print(f"    エラー: '{char_str}' (rot) の処理中にエラー - {e}")
        else:
            print(f"    警告: 回転vrt2文字 '{char_str}' のベースグリフがglyphsテーブルに見つかりません。スキップします。")
            skipped_rotated_no_base += 1
    print(f"  {count_rotated}個の回転vrt2グリフ画像をエクスポートしました。")
    if skipped_rotated_no_base > 0:
        print(f"  {skipped_rotated_no_base}個の回転vrt2文字はベースグリフがなかったためスキップされました。")

    print("  非回転vrt2グリフをエクスポート...")
    nr_vrt2_glyphs = db_helper.get_all_non_rotated_vrt2_glyphs_data()
    for char_str_nr, image_data_nr in nr_vrt2_glyphs:
        unicode_val_nr = ord(char_str_nr)
        img_name_nr = f"uni{unicode_val_nr:04X}.vert.png"
        img_path_nr = img_output_dir / img_name_nr
        try:
            with open(img_path_nr, "wb") as f:
                f.write(image_data_nr)
        except IOError as e:
            print(f"    エラー: {img_name_nr} (nrot) の保存に失敗 - {e}")
    print(f"  {len(nr_vrt2_glyphs)}個の非回転vrt2グリフ画像をエクスポートしました。")

    print("  私用領域(PUA)グリフをエクスポート...")
    pua_glyphs = db_helper.get_all_pua_glyphs_data()
    for char_str_pua, unicode_val_pua, image_data_pua, _ in pua_glyphs:
        if unicode_val_pua is not None:
            img_name_pua = f"uni{unicode_val_pua:04X}.png"
            img_path_pua = img_output_dir / img_name_pua
            try:
                with open(img_path_pua, "wb") as f:
                    f.write(image_data_pua)
            except IOError as e:
                print(f"    エラー: {img_name_pua} (PUA) の保存に失敗 - {e}")
        else:
            print(f"    警告: '{char_str_pua}' (PUA) は有効なUnicode値を持っていません。スキップします。")
    print(f"  {len(pua_glyphs)}個のPUAグリフ画像をエクスポートしました。")

def process_single_image_to_svg(png_path: Path):
    svg_output_dir = png_path.parent.parent / SVG_SUBDIR_NAME
    svg_name = png_path.stem + ".svg"
    svg_path = svg_output_dir / svg_name
    
    try:
        img_pil = Image.open(png_path)
        image_to_smooth_svg(img_pil, str(svg_path), 500, 500)
        return (str(png_path), True, None)
    except Exception as e:
        error_info = f"エラー: {png_path.name} のSVG変換に失敗 - {e}\n{traceback.format_exc()}"
        return (str(png_path), False, error_info)

def step2_convert_images_to_svg(img_source_dir: Path, svg_output_dir: Path):
    print("\nステップ2: 画像をSVGに変換中...")
    if svg_output_dir.exists():
        shutil.rmtree(svg_output_dir)
    svg_output_dir.mkdir(parents=True, exist_ok=True)

    png_files = list(img_source_dir.glob("*.png"))
    if not png_files:
        print("  変換対象のPNG画像が見つかりません。")
        return

    print(f"  {len(png_files)}個のPNG画像を並列処理します。")

    with multiprocessing.Pool(processes=None) as pool:
        results = pool.map(process_single_image_to_svg, png_files)

    success_count = 0
    for path, success, error in results:
        if success:
            success_count += 1
        else:
            print(f"      {error}")

    print(f"  {success_count}/{len(png_files)} 個のSVG変換処理完了。")

def preprocess_svg_content_for_font(svg_content_raw): 
    return re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", svg_content_raw, count=1).strip()

def has_meaningful_drawing_elements_for_font(svg_root): 
    if svg_root is None: return False
    for elem in svg_root.iter():
        tag_name = elem.tag
        if '}' in tag_name: tag_name = tag_name.split('}', 1)[1] 
        if tag_name == "path":
            if elem.get("d", "").strip(): return True
        elif tag_name in ["circle", "ellipse", "line", "polyline", "polygon", "rect"]:
            return True 
    return False

def set_font_names(font_path_str: str, family_name_display: str, family_name_ascii: str, style_name: str):
    """
    ビルド後のOTFファイルを開き、内部のnameテーブルに日本語名と英語（ASCII）名の両方を設定する。
    """
    print(f"  フォント内部の名称を '{family_name_display}' (日本語) と '{family_name_ascii}' (英語) に設定中...")
    try:
        font = TTFont(font_path_str)
        name_table = font['name']

        # 日本語名とASCII名の両方のフルネームを生成
        full_name_display = f"{family_name_display} {style_name}"
        full_name_ascii = f"{family_name_ascii} {style_name}"
        
        # 設定する名前IDと、(日本語文字列, ASCII文字列)のタプルをマッピング
        names_to_set = {
            1: (family_name_display, family_name_ascii),
            2: (style_name, style_name),  # スタイル名は通常ASCIIなので共通
            4: (full_name_display, full_name_ascii),
            16: (family_name_display, family_name_ascii),
            17: (style_name, style_name),
        }
        
        # 言語IDの定数
        LANG_ID_EN_WIN, LANG_ID_EN_MAC = 1033, 0
        LANG_ID_JA_WIN, LANG_ID_JA_MAC = 1041, 11
        
        for name_id, (display_str, ascii_str) in names_to_set.items():
            # setName(string, nameID, platformID, platEncID, langID) を使用
            
            # 英語名 (ASCII) を設定
            name_table.setName(ascii_str, name_id, 3, 1, LANG_ID_EN_WIN) # Windows, Unicode BMP, English (US)
            name_table.setName(ascii_str, name_id, 1, 0, LANG_ID_EN_MAC) # Mac, Roman, English
            
            # 日本語名 (表示用) を設定
            if display_str != ascii_str: # 日本語名がASCII名と異なる場合のみ追加
                name_table.setName(display_str, name_id, 3, 1, LANG_ID_JA_WIN) # Windows, Unicode BMP, Japanese
                name_table.setName(display_str, name_id, 1, 1, LANG_ID_JA_MAC) # Mac, Unicode 2.0+, Japanese
        
        font.save(font_path_str)
        font.close()
        print("  フォント内部の名称設定完了。")
        return True
    except Exception as e:
        print(f"  [ERROR] フォント内部の名称設定中にエラーが発生しました: {e}\n{traceback.format_exc()}")
        return False


def step3_build_otf_from_svgs(svg_source_dir: Path, db_helper: FontBuildDBHelper, 
                              ufo_output_dir_base: Path, otf_final_dir: Path):
    print("\nステップ3: SVGからOTFフォントをビルド中...")

    font_family_name_display = db_helper.get_font_name()
    font_family_name_ascii = db_helper.get_font_ascii_name()
    font_style_name = db_helper.get_font_weight()
    copyright_text = db_helper.get_copyright_info()
    license_text = db_helper.get_license_info()
    ascender = db_helper.get_ascender_height()
    descender = UNITS_PER_EM - ascender
    if descender > 0: descender = -descender

    safe_ascii_name = re.sub(r'[^\w-]', '', font_family_name_ascii)
    if not safe_ascii_name: safe_ascii_name = "UntitledFont"
    safe_style_name = re.sub(r'[^\w-]', '', font_style_name)
    if not safe_style_name: safe_style_name = "Regular"
    
    ufo_dir_name = f"{safe_ascii_name}-{safe_style_name}.ufo"
    intermediate_otf_file_name = f"{safe_ascii_name}-{safe_style_name}.otf"

    safe_display_family_name = re.sub(r'[\\/*?:"<>|]', '_', font_family_name_display)
    if not safe_display_family_name: safe_display_family_name = "UntitledFont"
    final_otf_file_name = f"{safe_display_family_name}-{safe_style_name}.otf"

    ufo_path = ufo_output_dir_base / ufo_dir_name
    intermediate_otf_path = otf_final_dir / intermediate_otf_file_name
    final_otf_path = otf_final_dir / final_otf_file_name

    if ufo_path.exists(): shutil.rmtree(ufo_path)
    if intermediate_otf_path.exists(): intermediate_otf_path.unlink()
    if final_otf_path.exists() and final_otf_path.resolve() != intermediate_otf_path.resolve():
        final_otf_path.unlink()

    print(f"  UFO構造を初期化 ({ufo_path})...")
    font = ufoLib2.Font()

    font.info.familyName = font_family_name_ascii
    font.info.styleName = font_style_name
    
    if copyright_text: font.info.copyright = copyright_text
    if license_text: font.info.openTypeNameLicense = license_text
    
    font.info.styleMapFamilyName = font_family_name_ascii
    font.info.styleMapStyleName = font_style_name.lower()

    font.info.unitsPerEm = UNITS_PER_EM
    font.info.ascender = ascender
    font.info.descender = descender 
    font.info.capHeight = ascender 
    font.info.xHeight = int(ascender * 0.66) 

    font.info.versionMajor = 1
    font.info.versionMinor = 0 

    font.info.openTypeOS2TypoAscender = ascender
    font.info.openTypeOS2TypoDescender = descender
    font.info.openTypeOS2TypoLineGap = 0 
    font.info.openTypeOS2WinAscent = ascender
    font.info.openTypeOS2WinDescent = abs(descender)
    font.info.openTypeOS2Panose = [0,0,0,0,0,0,0,0,0,0] 
    font.info.openTypeOS2UnicodeRanges = [0, 17, 18, 19, 48] 
    font.info.openTypeOS2CodePageRanges = [17] 
    font.info.openTypeOS2VendorID = "MYMD" 

    ps_family_name = "".join(font_family_name_ascii.split())
    ps_style_name = "".join(font_style_name.split())
    font.info.postscriptFontName = f"{ps_family_name}-{ps_style_name}"
    if len(font.info.postscriptFontName) > 63:
        font.info.postscriptFontName = font.info.postscriptFontName[:63]
    font.info.postscriptIsFixedPitch = True 

    font.info.openTypeHheaAscender = ascender
    font.info.openTypeHheaDescender = descender
    font.info.openTypeHheaLineGap = 0

    font.info.openTypeVheaVertTypoAscender = ascender 
    font.info.openTypeVheaVertTypoDescender = ascender - DEFAULT_ADVANCE_WIDTH 
    font.info.openTypeVheaVertTypoLineGap = 0

    default_layer = font.layers.defaultLayer
    glyph_order = []
    vertical_substitutions = []
    
    standard_glyph_transform = Transform(1, 0, 0, -1, 0, ascender)

    svg_files = sorted(list(svg_source_dir.glob("*.svg")))
    print(f"  {len(svg_files)}個のSVGファイルを処理します...")

    notdef_svg_processed = False

    for svg_file_path in svg_files:
        glyph_name_from_file = svg_file_path.stem 
        try:
            svg_content_raw = svg_file_path.read_text(encoding="utf-8-sig")
            svg_content_for_parsing = preprocess_svg_content_for_font(svg_content_raw)
            parsed_svg_root = None
            final_svg_viewbox_transform = Transform() 

            if svg_content_for_parsing.strip():
                try:
                    parsed_svg_root = ET.fromstring(svg_content_for_parsing)
                    if vb_str := parsed_svg_root.get("viewBox"):
                        vb_p = [float(p) for p in vb_str.strip().replace(',', ' ').split()]
                        if len(vb_p) == 4:
                            vb_min_x, vb_min_y, vb_w, vb_h = vb_p
                            scale = UNITS_PER_EM / (vb_h if vb_h != 0 else UNITS_PER_EM)
                            final_svg_viewbox_transform = Transform(scale, 0, 0, scale, -vb_min_x * scale, -vb_min_y * scale)
                except Exception as e: print(f"      警告: viewBox処理エラー ({glyph_name_from_file}): {e}")

            ufo_glyph_name, unicode_val, char_for_adv = "", None, None
            if glyph_name_from_file == "_notdef": ufo_glyph_name = ".notdef"
            elif (match := re.match(r"uni([0-9a-fA-F]{4,6})(\.vert)?", glyph_name_from_file)):
                unicode_val = int(match.group(1), 16); char_for_adv = chr(unicode_val)
                ufo_glyph_name = glyph_name_from_file if match.group(2) else f"uni{unicode_val:04X}"
            else: continue
            
            glyph = default_layer.newGlyph(ufo_glyph_name)
            adv_width = db_helper.get_advance_width_for_char(char_for_adv) if char_for_adv else DEFAULT_ADVANCE_WIDTH
            glyph.width = adv_width
            glyph.height = adv_width
            if unicode_val and not ufo_glyph_name.endswith(".vert"): glyph.unicodes = [unicode_val]
            
            if parsed_svg_root and has_meaningful_drawing_elements_for_font(parsed_svg_root):
                combined_transform = standard_glyph_transform.transform(final_svg_viewbox_transform)
                SVGPath.fromstring(svg_content_for_parsing, transform=combined_transform).draw(SegmentToPointPen(glyph.getPointPen()))
            
            if ufo_glyph_name == ".notdef": notdef_svg_processed = True
            glyph.lib["public.verticalOrigin"] = ascender
            if ufo_glyph_name not in glyph_order: glyph_order.append(ufo_glyph_name)
            if ufo_glyph_name.endswith(".vert"): vertical_substitutions.append((ufo_glyph_name[:-5], ufo_glyph_name))
        
        except Exception as e: 
            print(f"    致命的エラー: {svg_file_path.name} の処理に失敗 - {e}\n{traceback.format_exc()}")

    if not notdef_svg_processed or ".notdef" not in default_layer or not default_layer[".notdef"].contours:
        if ".notdef" in default_layer: del default_layer[".notdef"]
        notdef_glyph = default_layer.newGlyph(".notdef"); notdef_glyph.width = DEFAULT_ADVANCE_WIDTH
        pen = notdef_glyph.getPen(); margin = UNITS_PER_EM * 0.05
        rect_y_start = descender + margin; rect_height = (ascender - margin) - rect_y_start
        if (rect_width := DEFAULT_ADVANCE_WIDTH - 2 * margin) > 0 and rect_height > 0:
            pen.moveTo((margin, rect_y_start)); pen.lineTo((margin + rect_width, rect_y_start))
            pen.lineTo((margin + rect_width, rect_y_start + rect_height)); pen.lineTo((margin, rect_y_start + rect_height)); pen.closePath()
        notdef_glyph.height = DEFAULT_ADVANCE_WIDTH; notdef_glyph.lib["public.verticalOrigin"] = ascender
    
    if ".notdef" in glyph_order:
        if glyph_order[0] != ".notdef": glyph_order.remove(".notdef"); glyph_order.insert(0, ".notdef")
    elif ".notdef" in default_layer: glyph_order.insert(0, ".notdef")

    font.glyphOrder = glyph_order
    
    if vertical_substitutions:
        print("  縦書き用feature (vert, vrt2) を生成中...")
        subs = "\n".join([f"    sub {b} by {v};" for b,v in vertical_substitutions if b in font and v in font])
        if subs: font.features.text = f"feature vert {{\n{subs}\n}} vert;\n\nfeature vrt2 {{\n{subs}\n}} vrt2;"
    
    print(f"\n  UFOを保存中: {ufo_path}");font.save(ufo_path, overwrite=True)
    print(f"  fontmakeでOTFをコンパイル中 (中間ファイル: {intermediate_otf_path})")
    try:
        cmd = ["fontmake", "-u", str(ufo_path), "-o", "otf", "--output-path", str(intermediate_otf_path)]
        env = os.environ.copy(); env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', errors='replace', env=env)
        
        if result.stderr:
            print(f"    --- fontmake LOG ---\n{result.stderr}\n    --- End of LOG ---")
            
        if result.returncode == 0 and intermediate_otf_path.exists():
            print(f"  OTFコンパイル成功: {intermediate_otf_path} ({intermediate_otf_path.stat().st_size} bytes)")
            
            if set_font_names(str(intermediate_otf_path), font_family_name_display, font_family_name_ascii, font_style_name):
                if intermediate_otf_path.resolve() != final_otf_path.resolve():
                    print(f"  最終ファイル名にリネーム: {final_otf_path}")
                    shutil.move(intermediate_otf_path, final_otf_path)
                print(f"  ビルド成功: {final_otf_path}")
            else:
                print(f"    [ERROR] フォント内部名の設定に失敗したため、ビルドを中止します。")
                if intermediate_otf_path.exists(): intermediate_otf_path.unlink()
        else:
            error_msg = f"fontmakeコンパイル失敗 (終了コード {result.returncode})。"
            if not intermediate_otf_path.exists():
                error_msg = "fontmakeは完了しましたがOTFファイルが作成されませんでした。"
            print(f"    [ERROR] {error_msg}")
            if intermediate_otf_path.exists():
                intermediate_otf_path.unlink()

    except FileNotFoundError:
        print("    [ERROR] 'fontmake'コマンドが見つかりませんでした。fontmakeがインストールされ、PATHに含まれていることを確認してください。")
    except Exception as e:
        print(f"    [ERROR] fontmake実行中に予期せぬエラーが発生しました - {e}")
        if intermediate_otf_path.exists():
            intermediate_otf_path.unlink()

def main():
    parser = argparse.ArgumentParser(description="フォントデータベースからOTFフォントをビルドします。")
    parser.add_argument("--db_path", required=True, help="入力となる.fontprojデータベースファイルのパス")
    args = parser.parse_args()
    base_dir = Path(".").resolve(); tmp_dir = base_dir / TMP_DIR_BASE_NAME
    img_tmp = tmp_dir / IMG_SUBDIR_NAME; svg_tmp = tmp_dir / SVG_SUBDIR_NAME; ufo_tmp = tmp_dir / UFO_SUBDIR_NAME
    if tmp_dir.exists(): shutil.rmtree(tmp_dir)
    for p in [img_tmp, svg_tmp, ufo_tmp]: p.mkdir(parents=True, exist_ok=True)
    try:
        db_helper = FontBuildDBHelper(args.db_path)
        step1_export_images_from_db(db_helper, img_tmp)
        step2_convert_images_to_svg(img_tmp, svg_tmp)
        step3_build_otf_from_svgs(svg_tmp, db_helper, ufo_tmp, base_dir)
        print("\nビルド処理が完了しました。")
    except Exception as e:
        print(f"\nエラーが発生したため処理を中断しました: {e}\n{traceback.format_exc()}"); sys.exit(1)

if __name__ == "__main__":
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except (TypeError, AttributeError):
            pass
    multiprocessing.freeze_support()
    main()
