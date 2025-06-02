#!/usr/bin/env python3
import argparse
import io
import os
import re
import sys
import glob
import shutil
import sqlite3
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image, ImageOps, ImageChops
import numpy as np
from skimage import measure
from skimage.filters import threshold_otsu
from scipy.signal import savgol_filter
import svgwrite
from fontTools.svgLib import SVGPath
from fontTools.pens.pointPen import SegmentToPointPen
from fontTools.pens.transformPen import TransformPointPen
from fontTools.misc.transform import Transform
import ufoLib2




# --- 定数 (main.pyから一部持ってくるが、DBクエリで直接キーを使うので多くは不要) ---
TMP_DIR_BASE_NAME = "tmp"
IMG_SUBDIR_NAME = "img"
SVG_SUBDIR_NAME = "svg"
UFO_SUBDIR_NAME = "ufo"

# main.py の定数と合わせる
SETTING_FONT_NAME = "font_name"
SETTING_FONT_WEIGHT = "font_weight"
SETTING_ASCENDER_HEIGHT = "ascender_height" # Note: main.pyではこのキーは現在保存されていない
SETTING_ROTATED_VRT2_CHARS = "rotated_vrt2_chars"
# main.py のデフォルト値
DEFAULT_FONT_NAME = "MyNewFont"
DEFAULT_FONT_WEIGHT = "Regular"
DEFAULT_ASCENDER_HEIGHT = 900 # main.pyのDEFAULT_ASCENDER_HEIGHTに合わせる
DEFAULT_ADVANCE_WIDTH = 1000
UNITS_PER_EM = 1000
DEFAULT_DESCENDER = -100 # (UNITS_PER_EM - DEFAULT_ASCENDER_HEIGHT) = 1000 - 900 = 100, so -100.

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

    def get_font_weight(self) -> str:
        row = self._execute_query("SELECT value FROM project_settings WHERE key = ?", (SETTING_FONT_WEIGHT,))
        return row[0]['value'] if row and row[0]['value'] else DEFAULT_FONT_WEIGHT
    
    def get_ascender_height(self) -> int:
        # 固定値としたため、デフォルト値を使用
        return DEFAULT_ASCENDER_HEIGHT


    def get_all_standard_glyphs_data(self) -> list: # [(char, image_data, advance_width), ...]
        # character, unicode_val, image_data, advance_width
        return self._execute_query("SELECT character, unicode_val, image_data, advance_width FROM glyphs WHERE image_data IS NOT NULL")

    def get_rotated_vrt2_chars(self) -> list: # [char, ...]
        row = self._execute_query("SELECT value FROM project_settings WHERE key = ?", (SETTING_ROTATED_VRT2_CHARS,))
        return list(row[0]['value']) if row and row[0]['value'] else []

    def get_glyph_data_for_char(self, character: str) -> tuple or None: # (image_data, advance_width)
        row = self._execute_query("SELECT image_data, advance_width FROM glyphs WHERE character = ? AND image_data IS NOT NULL", (character,))
        return (row[0]['image_data'], row[0]['advance_width']) if row else None

    def get_all_non_rotated_vrt2_glyphs_data(self) -> list: # [(char, image_data), ...]
        # character, image_data
        return self._execute_query("SELECT character, image_data FROM vrt2_glyphs WHERE image_data IS NOT NULL")

    def get_advance_width_for_char(self, character: str) -> int:
        row = self._execute_query("SELECT advance_width FROM glyphs WHERE character = ?", (character,))
        return row[0]['advance_width'] if row and row[0]['advance_width'] is not None else DEFAULT_ADVANCE_WIDTH

# --- 画像からSVGへの変換ロジック (test1.pyベース) ---
def smooth_contour_for_svg(contour, window_length=9, polyorder=3):
    if len(contour) < window_length:
        return contour
    x_coords = savgol_filter(contour[:, 1], window_length, polyorder)
    y_coords = savgol_filter(contour[:, 0], window_length, polyorder)
    return np.column_stack([y_coords, x_coords])

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
    except Exception as e: 
        print(f"  Warning: threshold_otsu failed for {Path(svg_path_str).name} ({e}). Checking if image is all one color.")
        unique_colors = np.unique(image_np)
        if len(unique_colors) == 1: 
             if unique_colors[0] < 128 : 
                 binary = np.ones_like(image_np, dtype=bool) 
    contours = measure.find_contours(binary, 0.5)

    dwg = svgwrite.Drawing(svg_path_str, size=(f"{image_width}px", f"{image_height}px"), 
                           viewBox=f"0 0 {image_width} {image_height}")

    if not contours:
        dwg.save() 
        return

    path_data_list = []
    for contour in contours:
        if len(contour) < 5: 
            continue
        smoothed = smooth_contour_for_svg(contour, window_length=13, polyorder=3) 
        simplified = measure.approximate_polygon(smoothed, tolerance=0.7) 
        if len(simplified) < 3: 
            continue
        
        d = f"M {simplified[0][1]:.2f} {simplified[0][0]:.2f}"
        for point in simplified[1:]:
            d += f" L {point[1]:.2f} {point[0]:.2f}"
        d += " Z"
        path_data_list.append(d)

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
    for char_str, unicode_val_db, image_data, _ in std_glyphs: # unicode_val_db is from DB
        if char_str == ".notdef":
            img_name = "_notdef.png" # Use underscore prefix for .notdef image
        elif unicode_val_db is not None and unicode_val_db != -1 : # Regular unicode char
            img_name = f"uni{unicode_val_db:04X}.png"
        else: # Should not happen if DB is consistent, but as a fallback
            safe_char_name = re.sub(r'[^\w-]', '_', char_str) # Sanitize name
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

def step2_convert_images_to_svg(img_source_dir: Path, svg_output_dir: Path):
    print("\nステップ2: 画像をSVGに変換中...")
    if svg_output_dir.exists():
        shutil.rmtree(svg_output_dir)
    svg_output_dir.mkdir(parents=True, exist_ok=True)

    png_files = list(img_source_dir.glob("*.png"))
    if not png_files:
        print("  変換対象のPNG画像が見つかりません。")
        return
        
    print(f"  {len(png_files)}個のPNG画像を処理します。")
    for png_path in png_files:
        svg_name = png_path.stem + ".svg" # Handles "_notdef.png" -> "_notdef.svg"
        svg_path = svg_output_dir / svg_name
        print(f"    変換中: {png_path.name} -> {svg_name}")
        try:
            img_pil = Image.open(png_path)
            canvas_width, canvas_height = 500, 500 
            image_to_smooth_svg(img_pil, str(svg_path), canvas_width, canvas_height)
        except Exception as e:
            import traceback
            print(f"      エラー: {png_path.name} のSVG変換に失敗 - {e}")
            print(traceback.format_exc())
    print(f"  SVG変換処理完了。")


# --- UFO/OTFビルドロジック (test5.pyベース) ---
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

def step3_build_otf_from_svgs(svg_source_dir: Path, db_helper: FontBuildDBHelper, 
                              ufo_output_dir_base: Path, otf_final_dir: Path):
    print("\nステップ3: SVGからOTFフォントをビルド中...")

    font_family_name = db_helper.get_font_name()
    font_style_name = db_helper.get_font_weight()
    ascender = db_helper.get_ascender_height()
    descender = UNITS_PER_EM - ascender
    if descender > 0: descender = -descender # Ensure descender is negative or zero

    safe_family_name = re.sub(r'[^\w-]', '', font_family_name)
    safe_style_name = re.sub(r'[^\w-]', '', font_style_name)
    if not safe_family_name: safe_family_name = "UntitledFont"
    if not safe_style_name: safe_style_name = "Regular"

    ufo_dir_name = f"{safe_family_name}-{safe_style_name}.ufo"
    otf_file_name = f"{safe_family_name}-{safe_style_name}.otf"

    ufo_path = ufo_output_dir_base / ufo_dir_name
    otf_path = otf_final_dir / otf_file_name 

    if ufo_path.exists():
        shutil.rmtree(ufo_path)
    if otf_path.exists():
        try:
            otf_path.unlink()
        except OSError as e:
            print(f"  警告: 既存のOTFファイル {otf_path} の削除に失敗しました: {e}")


    print(f"  UFO構造を初期化 ({ufo_path})...")
    font = ufoLib2.Font()
    font.info.familyName = font_family_name
    font.info.styleName = font_style_name
    
    font.info.styleMapFamilyName = font_family_name
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

    ps_family_name = "".join(font_family_name.split()) 
    ps_style_name = "".join(font_style_name.split())
    font.info.postscriptFontName = f"{ps_family_name}-{ps_style_name}"
    if len(font.info.postscriptFontName) > 63:
        font.info.postscriptFontName = font.info.postscriptFontName[:63]
        print(f"  警告: postscriptFontNameが63文字を超過したため切り詰めました: {font.info.postscriptFontName}")
    font.info.postscriptIsFixedPitch = True 
    font.info.postscriptUnderlineThickness = int(UNITS_PER_EM * 0.05)
    font.info.postscriptUnderlinePosition = int(descender * 0.75)

    font.info.openTypeHheaAscender = ascender
    font.info.openTypeHheaDescender = descender
    font.info.openTypeHheaLineGap = 0

    font.info.openTypeVheaVertTypoAscender = ascender 
    font.info.openTypeVheaVertTypoDescender = ascender - DEFAULT_ADVANCE_WIDTH 
    font.info.openTypeVheaVertTypoLineGap = 0

    default_layer = font.layers.defaultLayer
    glyph_order_set = set() # To ensure .notdef is only added once to the list
    glyph_order = []
    vertical_substitutions = []
    
    standard_glyph_transform = Transform(1, 0, 0, -1, 0, ascender)

    svg_files = sorted(list(svg_source_dir.glob("*.svg"))) # Includes _notdef.svg if present
    if not svg_files:
        print(f"  SVGファイルが {svg_source_dir} に見つかりません。.notdefのみのフォントを作成します。")
    else:
        print(f"  {len(svg_files)}個のSVGファイルを処理します...")

    processed_unicodes = set()
    notdef_svg_processed = False # Flag to track if .notdef was processed from SVG

    for svg_file_path in svg_files:
        glyph_name_from_file = svg_file_path.stem 
        is_notdef_glyph_from_svg = (glyph_name_from_file == "_notdef")
        
        print(f"    処理中: {svg_file_path.name} -> グリフ名: {glyph_name_from_file}")

        try:
            with open(svg_file_path, "r", encoding="utf-8-sig") as f: 
                svg_content_raw = f.read()
            
            svg_content_for_parsing = preprocess_svg_content_for_font(svg_content_raw)
            parsed_svg_root = None
            final_svg_viewbox_transform = Transform() 

            if svg_content_for_parsing.strip():
                try:
                    parsed_svg_root = ET.fromstring(svg_content_for_parsing)
                    vb_str = parsed_svg_root.get("viewBox")
                    if vb_str:
                        vb_p_str = vb_str.strip().replace(',', ' ').split()
                        if len(vb_p_str) == 4:
                            vb_p = [float(p) for p in vb_p_str]
                            vb_min_x, vb_min_y, vb_width, vb_height = vb_p
                            svg_source_height_for_scale = vb_height if vb_height != 0 else UNITS_PER_EM
                            scale_factor = UNITS_PER_EM / svg_source_height_for_scale
                            translate_x = -vb_min_x * scale_factor
                            translate_y = -vb_min_y * scale_factor 
                            final_svg_viewbox_transform = Transform(scale_factor, 0, 0, scale_factor, translate_x, translate_y)
                        else: print(f"      警告: 不正なviewBox形式 ({glyph_name_from_file})。")
                    else: print(f"      警告: viewBoxがありません ({glyph_name_from_file})。")
                except ET.ParseError as pe: print(f"      警告: XMLパースエラー ({glyph_name_from_file}): {pe}"); parsed_svg_root = None
                except ValueError as ve: print(f"      警告: 不正なviewBox値 ({glyph_name_from_file}): {ve}"); parsed_svg_root = None
            else: print(f"      SVGコンテンツが空です ({glyph_name_from_file})。")

            unicode_val = None
            char_for_adv_width = ""
            current_glyph_obj_name = ""

            if is_notdef_glyph_from_svg:
                current_glyph_obj_name = ".notdef"
                # .notdef doesn't have a char_for_adv_width from DB typically
            elif glyph_name_from_file.endswith(".vert"):
                base_glyph_name = glyph_name_from_file[:-5] 
                match_unicode_base = re.match(r"uni([0-9a-fA-F]{4,6})", base_glyph_name)
                if match_unicode_base:
                    unicode_val = int(match_unicode_base.group(1), 16)
                    char_for_adv_width = chr(unicode_val)
                    current_glyph_obj_name = glyph_name_from_file # e.g., uni3042.vert
                else:
                    print(f"      警告: '{glyph_name_from_file}' (vert) からUnicodeを解析できませんでした。スキップします。")
                    continue
            else: 
                match_unicode_std = re.match(r"uni([0-9a-fA-F]{4,6})", glyph_name_from_file)
                if match_unicode_std:
                    unicode_val = int(match_unicode_std.group(1), 16)
                    char_for_adv_width = chr(unicode_val)
                    processed_unicodes.add(unicode_val) 
                    current_glyph_obj_name = f"uni{unicode_val:04X}" 
                else:
                     print(f"      警告: '{glyph_name_from_file}' からUnicodeを解析できませんでした。スキップします。")
                     continue
            
            if not current_glyph_obj_name: # Should have been set or skipped
                print(f"      エラー: グリフオブジェクト名が設定されませんでした ({glyph_name_from_file})。スキップします。")
                continue

            glyph = default_layer.newGlyph(current_glyph_obj_name)
            
            if current_glyph_obj_name == ".notdef":
                adv_width = DEFAULT_ADVANCE_WIDTH # .notdef specific advance width
                notdef_svg_processed = True # Mark that we attempted to process .notdef from SVG
            elif char_for_adv_width:
                adv_width = db_helper.get_advance_width_for_char(char_for_adv_width)
            else: # Should not happen for non-.notdef glyphs if parsing was correct
                adv_width = DEFAULT_ADVANCE_WIDTH
            
            glyph.width = adv_width 
            glyph.height = adv_width 

            if unicode_val is not None and not current_glyph_obj_name.endswith(".vert") and current_glyph_obj_name != ".notdef":
                glyph.unicodes = [unicode_val]

            should_draw_outlines = False
            if parsed_svg_root and has_meaningful_drawing_elements_for_font(parsed_svg_root):
                should_draw_outlines = True
            
            if should_draw_outlines:
                try:
                    combined_transform = standard_glyph_transform.transform(final_svg_viewbox_transform)
                    svg_path_parser = SVGPath.fromstring(svg_content_for_parsing, transform=combined_transform)
                    point_pen = glyph.getPointPen()
                    segment_to_point_pen = SegmentToPointPen(point_pen) 
                    svg_path_parser.draw(segment_to_point_pen)
                except Exception as e_draw:
                    print(f"      エラー: SVGの解析/描画エラー ({current_glyph_obj_name}): {e_draw}")
                    # If .notdef drawing from SVG fails, we'll fall back to auto-generated rect later
                    if current_glyph_obj_name == ".notdef":
                        notdef_svg_processed = False # Allow fallback by unsetting the flag
                        if ".notdef" in default_layer: del default_layer[".notdef"] # Remove partially drawn
            elif current_glyph_obj_name == ".notdef": # If .notdef SVG was empty or meaningless
                notdef_svg_processed = False # Allow fallback
                if ".notdef" in default_layer: del default_layer[".notdef"] # Remove empty glyph

            glyph.lib["public.verticalOrigin"] = ascender 

            if current_glyph_obj_name not in glyph_order_set:
                glyph_order.append(current_glyph_obj_name)
                glyph_order_set.add(current_glyph_obj_name)

            if current_glyph_obj_name.endswith(".vert"):
                original_glyph_name_for_feat = current_glyph_obj_name[:-5] 
                if re.match(r"uni([0-9a-fA-F]{4,6})", original_glyph_name_for_feat):
                    vertical_substitutions.append((original_glyph_name_for_feat, current_glyph_obj_name))
        
        except Exception as e_outer:
            import traceback
            print(f"    致命的エラー: {svg_file_path.name} の処理に失敗 - {e_outer}")
            print(traceback.format_exc())
            if 'current_glyph_obj_name' in locals() and current_glyph_obj_name and current_glyph_obj_name in default_layer:
                 del default_layer[current_glyph_obj_name]


    # .notdef グリフ処理 (Fallback if not processed from SVG or SVG was empty/bad)
    if not notdef_svg_processed or ".notdef" not in default_layer or not default_layer[".notdef"].contours:
        if ".notdef" in default_layer and not default_layer[".notdef"].contours:
            print("  .notdef グリフはSVGから読み込まれましたが空でした。自動生成された四角形を使用します。")
            del default_layer[".notdef"] # Remove the empty one
        elif not notdef_svg_processed and ".notdef" not in default_layer:
             print("  _notdef.svg が見つからなかったか、処理できませんでした。.notdef グリフを自動生成します。")
        # Else, it might exist but failed drawing from SVG (already deleted above in that case)

        notdef_glyph = default_layer.newGlyph(".notdef")
        notdef_glyph.width = DEFAULT_ADVANCE_WIDTH 
        pen = notdef_glyph.getPen()
        margin = UNITS_PER_EM * 0.05 
        rect_width = DEFAULT_ADVANCE_WIDTH - 2 * margin
        # Use full ascender-descender height for the box
        rect_y_start = descender + margin 
        rect_y_end = ascender - margin
        rect_height_auto = rect_y_end - rect_y_start

        if rect_height_auto > 0 and rect_width > 0 :
            pen.moveTo((margin, rect_y_start))
            pen.lineTo((margin + rect_width, rect_y_start))
            pen.lineTo((margin + rect_width, rect_y_start + rect_height_auto))
            pen.lineTo((margin, rect_y_start + rect_height_auto))
            pen.closePath()
        else:
            print(f"    警告: .notdef の自動生成四角形の寸法が無効です (w:{rect_width}, h:{rect_height_auto})。空の .notdef になります。")

        notdef_glyph.height = DEFAULT_ADVANCE_WIDTH 
        notdef_glyph.lib["public.verticalOrigin"] = ascender
    
    # Ensure .notdef is first in glyphOrder
    if ".notdef" in glyph_order_set:
        if glyph_order[0] != ".notdef":
            glyph_order.remove(".notdef")
            glyph_order.insert(0, ".notdef")
    elif ".notdef" in default_layer: # If it was auto-generated and not in initial svg list
        glyph_order.insert(0, ".notdef")
        glyph_order_set.add(".notdef") # Should be covered by newGlyph
    else: # Should not happen if .notdef creation logic is correct
        print("    致命的エラー: .notdef グリフが最終的に作成されませんでした。")


    font.glyphOrder = glyph_order
    
    feature_text_parts = []
    if vertical_substitutions:
        print("  縦書き用feature (vert, vrt2) を生成中...")
        vert_feature_content = "feature vert {\n"
        for base, vert_glyph in vertical_substitutions:
            vert_feature_content += f"    sub {base} by {vert_glyph};\n"
        vert_feature_content += "} vert;\n"
        feature_text_parts.append(vert_feature_content)

        vrt2_feature_content = "feature vrt2 {\n" 
        for base, vert_glyph in vertical_substitutions:
            vrt2_feature_content += f"    sub {base} by {vert_glyph};\n"
        vrt2_feature_content += "} vrt2;\n"
        feature_text_parts.append(vrt2_feature_content)
    
    if feature_text_parts:
        font.features.text = "\n".join(feature_text_parts)
        print(f"  Feature text set:\n{font.features.text}")

    print(f"  UFOを保存中: {ufo_path}")
    try:
        ufo_output_dir_base.mkdir(parents=True, exist_ok=True) 
        font.save(ufo_path, overwrite=True)
    except Exception as e_save_ufo:
        print(f"    エラー: UFOの保存に失敗 - {e_save_ufo}")
        return

    print(f"  fontmakeでOTFをコンパイル中 (出力先: {otf_path})...")
    try:
        cmd = ["fontmake", "-u", str(ufo_path), "-o", "otf", "--output-path", str(otf_path)]
        print(f"    実行コマンド: {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, encoding='utf-8', env=env)
        
        if result.stdout: print(f"    fontmake STDOUT:\n{result.stdout}")
        if result.stderr: print(f"    fontmake STDERR:\n{result.stderr}")
        
        if result.returncode != 0:
            print(f"    エラー: fontmakeコンパイル失敗 (終了コード {result.returncode})。")
        elif not otf_path.exists():
            print(f"    エラー: fontmakeは完了しましたがOTFファイル '{otf_path}' が作成されませんでした。")
        else:
            print(f"  OTFコンパイル成功: {otf_path} ({otf_path.stat().st_size} bytes)")
    except FileNotFoundError:
        print("    エラー: fontmakeコマンドが見つかりません。fontToolsが正しくインストールされているか確認してください。")
    except Exception as e_fontmake:
        print(f"    エラー: fontmake実行中にエラー - {e_fontmake}")


# --- メイン処理 ---
def main():
    parser = argparse.ArgumentParser(description="フォントデータベースからOTFフォントをビルドします。")
    parser.add_argument("--db_path", required=True, help="入力となる.fontprojデータベースファイルのパス")
    args = parser.parse_args()


    base_dir = Path(".").resolve() 
    tmp_dir = base_dir / TMP_DIR_BASE_NAME
    
    img_tmp = tmp_dir / IMG_SUBDIR_NAME
    svg_tmp = tmp_dir / SVG_SUBDIR_NAME
    ufo_tmp = tmp_dir / UFO_SUBDIR_NAME 

    tmp_dir.mkdir(parents=True, exist_ok=True)

    for p in [img_tmp, svg_tmp, ufo_tmp]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True) 
    
    otf_output_dir = base_dir 

    try:
        db_helper = FontBuildDBHelper(args.db_path)

        step1_export_images_from_db(db_helper, img_tmp)
        step2_convert_images_to_svg(img_tmp, svg_tmp)
        step3_build_otf_from_svgs(svg_tmp, db_helper, ufo_tmp, otf_output_dir)



    except Exception as e:

        sys.exit(1) 

if __name__ == "__main__":
    try:
        if sys.stdout.encoding is None or sys.stdout.encoding.lower() != 'utf-8':
            sys.stdout.reconfigure(encoding='utf-8')
        if sys.stderr.encoding is None or sys.stderr.encoding.lower() != 'utf-8':
            sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
    main()
