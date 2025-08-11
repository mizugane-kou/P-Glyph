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

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import find_contours
from PIL import Image
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

# --- ContourProcessorクラス (角文字用の独自処理) ---
class ContourProcessor:
    LINEARITY_THICKNESS_THRESHOLD = 2.0
    FAT_GROUP_SPLIT_ANGLE_THRESHOLD = 20.0
    SKIMAGE_CONTOUR_LEVEL = 128.0
    HV_TOLERANCE = 6.0
    GROUPING_ANGLE_TOLERANCE = 15
    NOISE_LENGTH_THRESHOLD = 12.0
    THIN_FEATURE_THRESHOLD = 7.0
    IMG_WIDTH = 500
    IMG_HEIGHT = 500

    def __init__(self, image_np_bgr: np.ndarray):
        self.original_image = self._prepare_image(image_np_bgr)
    def _prepare_image(self, image_bgr: np.ndarray):
        if len(image_bgr.shape) == 2 or image_bgr.shape[2] == 1: image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)
        if np.mean(image_bgr[0, 0]) < 128: image_bgr = cv2.bitwise_not(image_bgr)
        return cv2.resize(image_bgr, (self.IMG_WIDTH, self.IMG_HEIGHT))
    def run(self):
        initial_contours = self._find_contours()
        if not initial_contours: return []
        processed_contours = []
        for contour in initial_contours:
            all_groups, all_contour_points = self._create_and_group_segments([contour])
            if not all_groups: continue
            groups, contour_points = all_groups[0], all_contour_points[0]
            groups = self._merge_noisy_groups([groups], [contour_points])[0];groups = self._merge_groups_by_thickness([groups])[0]
            groups = self._split_fat_groups([groups])[0];groups = self._recalculate_and_snap_groups([groups])[0]
            groups = self._merge_consecutive_hv_groups([groups])[0];corrected = self._correct_contours([groups])
            if not corrected: continue
            aligned = self._finalize_contour_alignment(corrected)
            if not aligned: continue
            final_contour = self._apply_final_shift(aligned)
            if final_contour: processed_contours.append(final_contour[0])
        return processed_contours
    def _find_contours(self):
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY);gray_inverted = cv2.bitwise_not(gray)
        contours_sk = find_contours(gray_inverted, level=self.SKIMAGE_CONTOUR_LEVEL)
        return [c[:, [1, 0]].astype(np.int32).reshape(-1, 1, 2) for c in contours_sk]
    def _create_and_group_segments(self, contours):
        all_groups, all_contour_points = [], []
        for contour in contours:
            if len(contour) < 2: continue
            contour_points = contour.squeeze(axis=1);all_contour_points.append(contour_points)
            point_list = contour_points.tolist(); point_list.append(point_list[0])
            segments = [{'p1': p1, 'p2': p2, 'original_angle': self._calculate_angle(p1,p2), 'len': np.linalg.norm(np.array(p2)-np.array(p1))} for i in range(len(point_list)-1) if not np.array_equal(p1:=np.array(point_list[i]), p2:=np.array(point_list[i+1]))]
            if not segments: continue
            groups = []
            if segments:
                current_group = [segments[0]]
                for seg in segments[1:]:
                    if self._angle_diff(seg['original_angle'], current_group[-1]['original_angle']) < self.GROUPING_ANGLE_TOLERANCE: current_group.append(seg)
                    else: groups.append(current_group); current_group = [seg]
                groups.append(current_group)
            if len(groups) > 1 and self._angle_diff(groups[0][0]['original_angle'], groups[-1][0]['original_angle']) < self.GROUPING_ANGLE_TOLERANCE: groups[0].extend(groups.pop(-1))
            for g in groups: self._recalculate_single_group_angle(g, snap=False)
            all_groups.append(groups)
        return all_groups, all_contour_points
    def _merge_noisy_groups(self, all_groups, all_contour_points):
        merged_all_groups = []
        for i, groups in enumerate(all_groups):
            if not groups or len(groups) <= 2: merged_all_groups.append(groups); continue
            kdtree = cKDTree(all_contour_points[i]) if len(all_contour_points[i]) > 0 else None
            merged_in_pass = True
            while merged_in_pass:
                merged_in_pass = False
                if len(groups) <= 2: break
                sorted_indices = sorted(range(len(groups)), key=lambda k: sum(s['len'] for s in groups[k]))
                for idx in sorted_indices:
                    if len(groups) <= 2: break
                    group_to_check = groups[idx]
                    if sum(s['len'] for s in group_to_check) >= self.NOISE_LENGTH_THRESHOLD or (kdtree and self._get_local_thickness(group_to_check, kdtree) < self.THIN_FEATURE_THRESHOLD): continue
                    num_groups = len(groups); prev_i, next_i = (idx - 1 + num_groups) % num_groups, (idx + 1) % num_groups
                    prev_g, next_g = groups[prev_i], groups[next_i]
                    if self._angle_diff(prev_g[0].get('angle', 0), next_g[0].get('angle', 0)) < 15.0:
                        to_merge = groups.pop(idx)
                        try:
                            target_idx = groups.index(prev_g) if sum(s['len'] for s in prev_g) >= sum(s['len'] for s in next_g) else groups.index(next_g)
                            groups[target_idx].extend(to_merge); self._recalculate_single_group_angle(groups[target_idx], snap=False); merged_in_pass = True; break
                        except ValueError:
                            if groups: groups[0].extend(to_merge); self._recalculate_single_group_angle(groups[0], snap=False); merged_in_pass = True; break
            merged_all_groups.append(groups)
        return merged_all_groups
    def _merge_groups_by_thickness(self, all_groups):
        merged_all_groups = [];
        for groups in all_groups:
            if len(groups) < 2: merged_all_groups.append(groups); continue
            merged_in_pass = True
            while merged_in_pass:
                merged_in_pass = False; i = 0
                while i < len(groups):
                    if len(groups) < 2: break
                    next_idx = (i + 1) % len(groups); current, next_g = groups[i], groups[next_idx]
                    combined_points = np.array([s['p1'] for s in current] + [current[-1]['p2']] + [s['p1'] for s in next_g] + [next_g[-1]['p2']], dtype=np.float32)
                    if len(combined_points) >= 3 and min(cv2.minAreaRect(combined_points)[1]) < self.LINEARITY_THICKNESS_THRESHOLD:
                        merged_group = groups.pop(next_idx); target_idx = i if next_idx > i else i - 1
                        groups[target_idx].extend(merged_group); self._recalculate_single_group_angle(groups[target_idx], snap=False); merged_in_pass = True; break
                    else: i += 1
                if not merged_in_pass: break
            merged_all_groups.append(groups)
        return merged_all_groups
    def _split_fat_groups(self, all_groups):
        new_all_groups = []
        for groups in all_groups:
            if not groups: new_all_groups.append(groups); continue
            final, queue = [], list(groups)
            while queue:
                group = queue.pop(0)
                if len(group) <= 1 or self._get_group_thickness(group) <= self.LINEARITY_THICKNESS_THRESHOLD: final.append(group); continue
                max_d, split_i = -1, -1
                for i in range(len(group) - 1):
                    d = self._angle_diff(group[i]['original_angle'], group[i+1]['original_angle'])
                    if d > max_d: max_d, split_i = d, i + 1
                if split_i != -1 and max_d > self.FAT_GROUP_SPLIT_ANGLE_THRESHOLD:
                    g1, g2 = group[:split_i], group[split_i:];
                    if g1: queue.append(g1)
                    if g2: queue.append(g2)
                else: final.append(group)
            for g in final: self._recalculate_single_group_angle(g, snap=False)
            new_all_groups.append(final)
        return new_all_groups
    def _get_local_thickness(self, group, kdtree):
        if not group or kdtree is None: return float('inf')
        mid = np.mean([s['p1'] for s in group], axis=0); dist, _ = kdtree.query(mid); return dist
    def _get_group_thickness(self, group):
        if not group: return 0
        points = np.array([s['p1'] for s in group] + [group[-1]['p2']], dtype=np.float32)
        if len(points) < 3: return 0
        return min(cv2.minAreaRect(points)[1])
    def _recalculate_and_snap_groups(self, all_groups):
        for groups in all_groups:
            for g in groups:
                if g: self._recalculate_single_group_angle(g, snap=True)
        return all_groups
    def _merge_consecutive_hv_groups(self, all_groups):
        final = []
        for groups in all_groups:
            if len(groups) < 2: final.append(groups); continue
            merged = []
            if groups:
                merged.append(groups[0])
                for g in groups[1:]:
                    last = merged[-1]
                    if g and last and g[0]['angle'] in {0.0, 90.0} and g[0]['angle'] == last[0]['angle']: last.extend(g)
                    else: merged.append(g)
                if len(merged) > 1 and merged[-1] and merged[0] and merged[-1][0]['angle'] in {0.0, 90.0} and merged[-1][0]['angle'] == merged[0][0]['angle']: merged[0].extend(merged.pop(-1))
            final.append(merged)
        return final
    def _correct_contours(self, all_groups):
        corrected = []
        for groups in all_groups:
            if not groups or len(groups) < 2: continue
            lines = []
            for g in groups:
                if not g: continue
                rad = np.deg2rad(g[0]['angle']); points = np.array([s['p1'] for s in g] + [g[-1]['p2']])
                c = np.mean(points, axis=0); d = np.array([np.cos(rad), np.sin(rad)])
                lines.append((c - d * 1e4, c + d * 1e4))
            if len(lines) < 2: continue
            new_points = []
            for i in range(len(lines)):
                l1, l2 = lines[i], lines[(i + 1) % len(lines)]; inter = self._line_intersection(l1, l2)
                if inter is None: p1_end, p2_start = groups[i][-1]['p2'], groups[(i + 1) % len(lines)][0]['p1']; inter = tuple(np.mean([p1_end, p2_start], axis=0).astype(int))
                new_points.append([inter])
            if new_points: corrected.append(np.array(new_points, dtype=np.int32))
        return corrected
    def _finalize_contour_alignment(self, contours):
        final = []; ALIGN_TOL = 2.5
        for contour in contours:
            num = len(contour)
            if num < 2: final.append(contour); continue
            points = new_points = contour.copy().squeeze(axis=1); edges = []
            for i in range(num):
                p1, p2 = points[i], points[(i + 1) % num]; a = self._calculate_angle(p1, p2)
                abs_a = abs((a + 180) % 360 - 180); e_type = -1
                if abs_a <= ALIGN_TOL or abs_a >= (180 - ALIGN_TOL): e_type = 0
                elif abs(abs_a - 90) <= ALIGN_TOL: e_type = 1
                edges.append({'type': e_type, 'p1': i, 'p2': (i + 1) % num})
            for type_p in [0, 1]:
                processed = [False] * num
                for i in range(num):
                    if processed[i] or edges[i]['type'] != type_p: continue
                    q, visited, seq_e = [i], {i}, []
                    while q:
                        idx = q.pop(0); seq_e.append(edges[idx])
                        for p_idx in [edges[idx]['p1'], edges[idx]['p2']]:
                            for n_idx in [(p_idx - 1 + num) % num, p_idx]:
                                if n_idx not in visited and edges[n_idx]['type'] == type_p: q.append(n_idx); visited.add(n_idx)
                    if not seq_e: continue
                    p_indices = {p for e in seq_e for p in (e['p1'], e['p2'])}
                    coord = int(round(np.mean([new_points[idx][1 - type_p] for idx in p_indices])))
                    for idx in p_indices: new_points[idx][1 - type_p] = coord; processed[idx] = True
            final.append(new_points.reshape(-1, 1, 2))
        return final
    def _apply_final_shift(self, contours):
        shifted = []; is_top = lambda a: -45 <= a <= 45; is_left = lambda a: -135 <= a < -45
        for contour in contours:
            points = contour.squeeze(axis=1); num = len(points)
            if num < 2: shifted.append(contour); continue
            shifts = [np.array([1, 0] if is_left(self._calculate_angle(points[i], points[(i + 1) % num])) else [0, 1] if is_top(self._calculate_angle(points[i], points[(i + 1) % num])) else [0, 0]) for i in range(num)]
            new_points = []
            for i in range(num):
                prev_i = (i - 1 + num) % num
                l1 = (points[prev_i] + shifts[prev_i], points[i] + shifts[prev_i]); l2 = (points[i] + shifts[i], points[(i + 1) % num] + shifts[i])
                inter = self._line_intersection(l1, l2)
                new_points.append(inter if inter is not None else tuple((points[i] + shifts[i]).astype(int)))
            shifted.append(np.array(new_points, dtype=np.int32).reshape(-1, 1, 2))
        return shifted
    def _recalculate_single_group_angle(self, group, snap=True):
        if not group: return
        total_len = sum(s['len'] for s in group)
        if total_len == 0: angle = group[0].get('original_angle', 0.0)
        else:
            sum_x = sum(s['len'] * np.cos(np.deg2rad(s['original_angle'])) for s in group)
            sum_y = sum(s['len'] * np.sin(np.deg2rad(s['original_angle'])) for s in group)
            angle = np.rad2deg(np.arctan2(sum_y, sum_x))
        final = self._snap_angle(angle) if snap else angle
        for s in group: s['angle'] = final
    def _calculate_angle(self, p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]; return np.rad2deg(np.arctan2(dy, dx)) if dx or dy else 0.0
    def _snap_angle(self, angle):
        a = abs((angle + 180) % 360 - 180)
        if a <= self.HV_TOLERANCE or a >= (180 - self.HV_TOLERANCE): return 0.0
        if abs(a - 90) <= self.HV_TOLERANCE: return 90.0
        return (angle + 180) % 360 - 180
    def _angle_diff(self, a1, a2): d = abs(a1 - a2); return min(d, 360 - d)
    def _line_intersection(self, l1, l2):
        (x1, y1), (x2, y2) = l1; (x3, y3), (x4, y4) = l2
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(den) < 1e-6: return None
        t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4); t = t_num / den
        return (int(round(x1 + t * (x2 - x1))), int(round(y1 + t * (y2 - y1))))

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

# --- SVG生成ヘルパー ---
def compute_depth(polygon, all_polygons):
    return sum(1 for other in all_polygons if other != polygon and other.contains(polygon))

def polygon_to_path_data(polygon):
    coords = list(polygon.exterior.coords)
    d = f"M {coords[0][0]:.2f} {coords[0][1]:.2f}"
    for x, y in coords[1:]:
        d += f" L {x:.2f} {y:.2f}"
    d += " Z"
    return d

def image_to_rectified_svg(image_pil: Image.Image, svg_path_str: str, image_width: int, image_height: int):
    image_np_bgr = cv2.cvtColor(np.array(image_pil.convert('RGB')), cv2.COLOR_RGB2BGR)
    processor = ContourProcessor(image_np_bgr)
    final_contours = processor.run()

    dwg = svgwrite.Drawing(svg_path_str, size=(f"{image_width}px", f"{image_height}px"), viewBox=f"0 0 {image_width} {image_height}")

    if not final_contours:
        dwg.save()
        return

    polygons = [Polygon(c.squeeze(axis=1)) for c in final_contours if len(c.squeeze(axis=1)) >= 3]
    corrected_polygons = []
    for poly in polygons:
        depth = compute_depth(poly, polygons)
        should_be_ccw = (depth % 2 == 0)
        if poly.exterior.is_ccw != should_be_ccw:
            corrected_polygons.append(Polygon(list(poly.exterior.coords)[::-1]))
        else:
            corrected_polygons.append(poly)
    
    path_data_list = [polygon_to_path_data(p) for p in corrected_polygons]
    if path_data_list:
        dwg.add(dwg.path(d=" ".join(path_data_list), fill='black', stroke='none'))
    dwg.save()

# --- ステップ1, 2 ---
def step1_export_images_from_db(db_helper: FontBuildDBHelper, img_output_dir: Path):
    print("\nステップ1: DBからグリフ画像をエクスポート中...")
    if img_output_dir.exists(): shutil.rmtree(img_output_dir)
    img_output_dir.mkdir(parents=True, exist_ok=True)
    std_glyphs = db_helper.get_all_standard_glyphs_data()
    for char, unicode_val, img_data, _ in std_glyphs:
        img_name = "_notdef.png" if char==".notdef" else f"uni{unicode_val:04X}.png" if unicode_val is not None and unicode_val!=-1 else re.sub(r'[^\w-]', '_', char)+".png"
        (img_output_dir/img_name).write_bytes(img_data)
    print(f"  {len(std_glyphs)}個の標準グリフ画像をエクスポートしました。")
    rotated_chars = db_helper.get_rotated_vrt2_chars()
    count_rotated = 0
    for char in rotated_chars:
        if data := db_helper.get_glyph_data_for_char(char):
            Image.open(io.BytesIO(data[0])).rotate(-90,expand=True).save(img_output_dir/f"uni{ord(char):04X}.vert.png", "PNG"); count_rotated += 1
    print(f"  {count_rotated}個の回転vrt2グリフ画像をエクスポートしました。")
    nr_vrt2_glyphs = db_helper.get_all_non_rotated_vrt2_glyphs_data()
    for char, img_data in nr_vrt2_glyphs: (img_output_dir/f"uni{ord(char):04X}.vert.png").write_bytes(img_data)
    print(f"  {len(nr_vrt2_glyphs)}個の非回転vrt2グリフ画像をエクスポートしました。")
    pua_glyphs = db_helper.get_all_pua_glyphs_data()
    for _, unicode_val, img_data, _ in pua_glyphs:
        if unicode_val: (img_output_dir/f"uni{unicode_val:04X}.png").write_bytes(img_data)
    print(f"  {len(pua_glyphs)}個のPUAグリフ画像をエクスポートしました。")

def process_single_image_to_svg(png_path: Path):
    svg_path = png_path.parent.parent / SVG_SUBDIR_NAME / (png_path.stem + ".svg")
    try:
        img_pil = Image.open(png_path)
        image_to_rectified_svg(img_pil, str(svg_path), ContourProcessor.IMG_WIDTH, ContourProcessor.IMG_HEIGHT)
        return (str(png_path), True, None)
    except Exception as e:
        return (str(png_path), False, f"Error: SVG conversion failed for {png_path.name}\n{traceback.format_exc()}")

def step2_convert_images_to_svg(img_source_dir: Path, svg_output_dir: Path):
    print("\nステップ2: 画像をSVGに変換中 (角文字処理)...")
    if svg_output_dir.exists(): shutil.rmtree(svg_output_dir)
    svg_output_dir.mkdir(parents=True, exist_ok=True)
    png_files = list(img_source_dir.glob("*.png"))
    if not png_files: print("  変換対象のPNG画像が見つかりません。"); return
    print(f"  {len(png_files)}個のPNG画像を並列処理します。")
    with multiprocessing.Pool() as pool: results = pool.map(process_single_image_to_svg, png_files)
    success_count = sum(1 for _, s, _ in results if s)
    for _, s, err in results:
        if not s: print(f"      {err}")
    print(f"  {success_count}/{len(png_files)} 個のSVG変換処理完了。")

# --- ステップ3 (移植と修正) ---
def set_font_names(font_path_str: str, family_name_display: str, family_name_ascii: str, style_name: str):
    print(f"  フォント内部の名称を '{family_name_display}' (日本語) と '{family_name_ascii}' (英語) に設定中...")
    try:
        font = TTFont(font_path_str)
        name_table = font['name']
        full_name_display = f"{family_name_display} {style_name}"
        full_name_ascii = f"{family_name_ascii} {style_name}"
        names_to_set = {
            1: (family_name_display, family_name_ascii), 2: (style_name, style_name),
            4: (full_name_display, full_name_ascii), 16: (family_name_display, family_name_ascii),
            17: (style_name, style_name),
        }
        LANG_ID_EN_WIN, LANG_ID_EN_MAC = 1033, 0
        LANG_ID_JA_WIN, LANG_ID_JA_MAC = 1041, 11
        for name_id, (display_str, ascii_str) in names_to_set.items():
            name_table.setName(ascii_str, name_id, 3, 1, LANG_ID_EN_WIN)
            name_table.setName(ascii_str, name_id, 1, 0, LANG_ID_EN_MAC)
            if display_str != ascii_str:
                name_table.setName(display_str, name_id, 3, 1, LANG_ID_JA_WIN)
                name_table.setName(display_str, name_id, 1, 1, LANG_ID_JA_MAC)
        font.save(font_path_str)
        font.close()
        print("  フォント内部の名称設定完了。")
        return True
    except Exception as e:
        print(f"  [ERROR] フォント内部の名称設定中にエラーが発生しました: {e}\n{traceback.format_exc()}")
        return False

# ★★★ ここからが修正の核心 ★★★
# step3_build_otf_from_svgs 内で欠落していたヘルパー関数を再定義
def preprocess_svg_content_for_font(svg_content_raw): 
    return re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", svg_content_raw, count=1).strip()

def has_meaningful_drawing_elements_for_font(svg_root): 
    if svg_root is None: return False
    for elem in svg_root.iter():
        tag_name = elem.tag.split('}')[-1]
        if tag_name == "path" and elem.get("d", "").strip(): return True
        if tag_name in ["circle", "ellipse", "line", "polyline", "polygon", "rect"]: return True 
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

    safe_ascii_name = re.sub(r'[^\w-]', '', font_family_name_ascii) or "UntitledFont"
    safe_style_name = re.sub(r'[^\w-]', '', font_style_name) or "Regular"
    
    ufo_dir_name = f"{safe_ascii_name}-{safe_style_name}.ufo"
    intermediate_otf_file_name = f"{safe_ascii_name}-{safe_style_name}.otf"
    safe_display_family_name = re.sub(r'[\\/*?:"<>|]', '_', font_family_name_display) or "UntitledFont"
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
    font.info.unitsPerEm = UNITS_PER_EM; font.info.ascender = ascender; font.info.descender = descender
    font.info.capHeight = ascender; font.info.xHeight = int(ascender * 0.66)
    font.info.versionMajor = 1; font.info.versionMinor = 0
    font.info.openTypeOS2TypoAscender = ascender; font.info.openTypeOS2TypoDescender = descender
    font.info.openTypeOS2TypoLineGap = 0; font.info.openTypeOS2WinAscent = ascender
    font.info.openTypeOS2WinDescent = abs(descender); font.info.openTypeOS2Panose = [0]*10
    font.info.openTypeOS2UnicodeRanges = [0, 17, 18, 19, 48]; font.info.openTypeOS2CodePageRanges = [17]
    font.info.openTypeOS2VendorID = "MYMD"
    ps_family_name = "".join(font_family_name_ascii.split())
    ps_style_name = "".join(font_style_name.split())
    font.info.postscriptFontName = f"{ps_family_name}-{ps_style_name}"[:63]
    font.info.postscriptIsFixedPitch = True
    font.info.openTypeHheaAscender = ascender; font.info.openTypeHheaDescender = descender
    font.info.openTypeHheaLineGap = 0; font.info.openTypeVheaVertTypoAscender = ascender
    font.info.openTypeVheaVertTypoDescender = ascender - DEFAULT_ADVANCE_WIDTH
    font.info.openTypeVheaVertTypoLineGap = 0

    default_layer = font.layers.defaultLayer; glyph_order, vertical_substitutions = [], []
    standard_glyph_transform = Transform(1, 0, 0, -1, 0, ascender)

    svg_files = sorted(list(svg_source_dir.glob("*.svg")))
    print(f"  {len(svg_files)}個のSVGファイルを処理します...")
    notdef_svg_processed = False
    for svg_file_path in svg_files:
        glyph_name_from_file = svg_file_path.stem
        try:
            svg_content_raw = svg_file_path.read_text(encoding="utf-8-sig")
            svg_content_for_parsing = preprocess_svg_content_for_font(svg_content_raw)
            parsed_svg_root, final_svg_viewbox_transform = None, Transform()
            if svg_content_for_parsing.strip():
                try:
                    parsed_svg_root = ET.fromstring(svg_content_for_parsing)
                    if vb_str := parsed_svg_root.get("viewBox"):
                        vb_p = [float(p) for p in vb_str.strip().replace(',', ' ').split()]
                        if len(vb_p) == 4:
                            scale = UNITS_PER_EM / (vb_p[3] if vb_p[3] != 0 else UNITS_PER_EM)
                            final_svg_viewbox_transform = Transform(scale, 0, 0, scale, -vb_p[0] * scale, -vb_p[1] * scale)
                except Exception as e: print(f"      警告: viewBox処理エラー ({glyph_name_from_file}): {e}")
            
            ufo_glyph_name, unicode_val, char_for_adv = "", None, None
            if glyph_name_from_file == "_notdef": ufo_glyph_name = ".notdef"
            elif (match := re.match(r"uni([0-9a-fA-F]{4,6})(\.vert)?", glyph_name_from_file)):
                unicode_val = int(match.group(1), 16); char_for_adv = chr(unicode_val)
                ufo_glyph_name = glyph_name_from_file if match.group(2) else f"uni{unicode_val:04X}"
            else: continue
            
            glyph = default_layer.newGlyph(ufo_glyph_name)
            adv_width = db_helper.get_advance_width_for_char(char_for_adv) if char_for_adv else DEFAULT_ADVANCE_WIDTH
            glyph.width = adv_width; glyph.height = adv_width
            if unicode_val and not ufo_glyph_name.endswith(".vert"): glyph.unicodes = [unicode_val]
            
            if parsed_svg_root and has_meaningful_drawing_elements_for_font(parsed_svg_root):
                SVGPath.fromstring(svg_content_for_parsing, transform=standard_glyph_transform.transform(final_svg_viewbox_transform)).draw(SegmentToPointPen(glyph.getPointPen()))
            
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
    
    if ".notdef" in glyph_order and glyph_order[0] != ".notdef":
        glyph_order.remove(".notdef"); glyph_order.insert(0, ".notdef")
    elif ".notdef" not in glyph_order and ".notdef" in default_layer:
        glyph_order.insert(0, ".notdef")
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
        
        if result.stderr: print(f"    --- fontmake LOG ---\n{result.stderr}\n    --- End of LOG ---")
            
        if result.returncode == 0 and intermediate_otf_path.exists():
            print(f"  OTFコンパイル成功: {intermediate_otf_path} ({intermediate_otf_path.stat().st_size} bytes)")
            if set_font_names(str(intermediate_otf_path), font_family_name_display, font_family_name_ascii, font_style_name):
                if intermediate_otf_path.resolve() != final_otf_path.resolve():
                    print(f"  最終ファイル名にリネーム: {final_otf_path}")
                    shutil.move(intermediate_otf_path, final_otf_path)
                print(f"  ビルド成功: {final_otf_path}")
            else:
                print(f"    [ERROR] フォント内部名の設定に失敗。ビルドを中止します。")
                if intermediate_otf_path.exists(): intermediate_otf_path.unlink()
        else:
            error_msg = f"fontmakeコンパイル失敗 (終了コード {result.returncode})。"
            if not intermediate_otf_path.exists(): error_msg = "fontmakeは完了しましたがOTFファイルが作成されませんでした。"
            print(f"    [ERROR] {error_msg}")
            if intermediate_otf_path.exists(): intermediate_otf_path.unlink()
    except FileNotFoundError:
        print("    [ERROR] 'fontmake'コマンドが見つかりません。")
    except Exception as e:
        print(f"    [ERROR] fontmake実行中にエラー - {e}")
        if intermediate_otf_path.exists(): intermediate_otf_path.unlink()

def main():
    parser = argparse.ArgumentParser(description="フォントデータベースからOTFフォントをビルドします（角文字処理）。")
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
        try: sys.stdout.reconfigure(encoding='utf-8'); sys.stderr.reconfigure(encoding='utf-8')
        except (TypeError, AttributeError): pass
    multiprocessing.freeze_support()
    main()
