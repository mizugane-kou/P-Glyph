

import sys
import os
import sqlite3
import functools
import re 
import json 
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Set

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox,
    QCheckBox, QGridLayout, QButtonGroup, QFrame,
    QScrollArea, QTextEdit, QSizePolicy, QFileDialog,
    QDialog, QMessageBox, QLineEdit, QSpinBox, QDialogButtonBox,
    QStatusBar,
    QTabBar, QStyle, QStyleOptionTab, QSpacerItem, 
    QTabWidget, QAbstractButton, QMenu
)
from PySide6.QtGui import (
    QPainter, QPen, QMouseEvent, QColor, QPixmap,
    QPainterPath, QKeySequence, QKeyEvent, QPaintEvent,
    QImage, QPalette, QDragEnterEvent, QDropEvent, 
    QFont, QFontDatabase, QFontMetrics, QTransform, QCursor, QResizeEvent, QPaintEngine, QIcon,
    QAction
)
from PySide6.QtCore import (
    Qt, QPoint, QPointF, Signal, QRectF, QSize, QBuffer,
    QIODevice, QByteArray, QRunnable, QThreadPool, Slot, QObject, QProcess, QTimer, 
    QRect, QMutex, QMutexLocker, QEvent, QThread 
)


# --- Constants ---
MAX_HISTORY_SIZE = 20
VIRTUAL_MARGIN = 30
CANVAS_IMAGE_WIDTH = 500
CANVAS_IMAGE_HEIGHT = 500
DEFAULT_GLYPH_PREVIEW_SIZE = QSize(64, 64)
GLYPH_GRID_WIDTH = 400
PROPERTIES_WIDTH = 250
GRID_COLUMNS = 4
GLYPH_ITEM_WIDTH = 80
GLYPH_ITEM_HEIGHT = 100
DEFAULT_CHAR_SET = "ぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎわゐゑをん"
FONT_SETTINGS_FILENAME = "font_settings.txt"
R_VERT_FILENAME = "r_vert.txt"
VERT_FILENAME = "vert.txt"
DEFAULT_R_VERT_CHARS = "≠≦≧〈〉《》「」『』【】〔〕〜゠ー（）：；＜＝＞［］＿｛｜｝～￣"
DEFAULT_VERT_CHARS = "‥…‘’“”、。，．"
SETTING_PEN_WIDTH = "pen_width"
SETTING_PEN_SHAPE = "pen_shape"
SETTING_CURRENT_TOOL = "current_tool"
SETTING_MIRROR_MODE = "mirror_mode"
SETTING_GLYPH_MARGIN_WIDTH = "glyph_margin_width"
SETTING_LAST_ACTIVE_GLYPH = "last_active_glyph"
SETTING_ROTATED_VRT2_CHARS = "rotated_vrt2_chars"
SETTING_NON_ROTATED_VRT2_CHARS = "non_rotated_vrt2_chars"
DEFAULT_PEN_WIDTH = 2
DEFAULT_PEN_SHAPE = "丸"
DEFAULT_CURRENT_TOOL = "brush"
DEFAULT_MIRROR_MODE = False
DEFAULT_GLYPH_MARGIN_WIDTH = 0
DEFAULT_ASCENDER_HEIGHT = 880
DEFAULT_ADVANCE_WIDTH = 1000

SETTING_FONT_NAME = "font_name"
SETTING_FONT_WEIGHT = "font_weight"
DEFAULT_FONT_NAME = "MyNewFont"
FONT_WEIGHT_OPTIONS = ["Thin", "ExtraLight", "Light", "Regular", "Medium", "SemiBold", "Bold", "ExtraBold", "Black"]
DEFAULT_FONT_WEIGHT = "Regular" 


SETTING_REFERENCE_IMAGE_OPACITY = "reference_image_opacity"
DEFAULT_REFERENCE_IMAGE_OPACITY = 0.5
VRT2_PREVIEW_BACKGROUND_TINT = QColor("#6E51E4")
REFERENCE_GLYPH_DISPLAY_DELAY=450
FONT_BOOKMARKS_FILENAME = "font_bookmarks.json"

# Settings for Kanji Viewer state
SETTING_KV_CURRENT_FONT = "kv_current_font" 
SETTING_KV_DISPLAY_MODE = "kv_display_mode" 
DEFAULT_KV_MODE_FOR_SETTINGS = 1 

GLYPH_LOAD_BATCH_SIZE = 100 # グリフ読み込みのバッチサイズ
GLYPH_UI_UPDATE_BATCH_SIZE = 100 # Batch size for UI updates in GlyphGridWidget


# --- kanzi2.py の関数群 (変更なし) ---
def get_data_file_path(filename: str) -> str | None:
    script_dir = os.path.dirname(os.path.abspath(__file__)) 
    data_path_local = os.path.join(script_dir, "data", filename)
    if os.path.exists(data_path_local):
        return data_path_local
    parent_dir_script = os.path.dirname(script_dir)
    data_path_parent_script = os.path.join(parent_dir_script, "data", filename)
    if os.path.exists(data_path_parent_script) and \
       (os.path.exists(os.path.join(parent_dir_script, "README.md")) or
        os.path.exists(os.path.join(parent_dir_script, ".git"))):
        return data_path_parent_script
    cwd_data_path = os.path.join(os.getcwd(), "data", filename)
    if os.path.exists(cwd_data_path):
        return cwd_data_path
    grandparent_dir = os.path.dirname(parent_dir_script)
    data_path_grandparent = os.path.join(grandparent_dir, "data", filename)
    if os.path.exists(data_path_grandparent) and \
       (os.path.exists(os.path.join(grandparent_dir, "README.md")) or
        os.path.exists(os.path.join(grandparent_dir, ".git"))):
        return data_path_grandparent
    return None

def load_json_data(file_path: str) -> dict | None:
    if file_path is None: return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f: data = json.load(f)
        return data
    except FileNotFoundError: return None
    except json.JSONDecodeError: return None

def find_kanji_by_shared_radical_per_radical(
    input_kanji: str,
    kanji_to_radicals_map: dict | None,
    radical_to_kanji_map: dict | None
) -> dict[str, list[str]]:
    if kanji_to_radicals_map is None or radical_to_kanji_map is None: return {}
    if not isinstance(input_kanji, str) or len(input_kanji) != 1: return {}
    if input_kanji not in kanji_to_radicals_map: return {}
    target_radicals = kanji_to_radicals_map.get(input_kanji, [])
    if not target_radicals: return {}
    results_per_radical = {}
    for radical in target_radicals:
        if radical in radical_to_kanji_map:
            kanji_list_for_radical = radical_to_kanji_map.get(radical, [])
            filtered_kanji_list = sorted(list(set(
                k for k in kanji_list_for_radical if k != input_kanji
            )))
            if filtered_kanji_list: results_per_radical[radical] = filtered_kanji_list
    return results_per_radical

# --- カスタム縦書きタブバー (変更なし) ---
class VerticalTabBar(QTabBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setElideMode(Qt.TextElideMode.ElideNone)
        self.font_metrics = QFontMetrics(self.font())
        self.vertical_padding = 8
        self.horizontal_padding = 6
        self._update_font_dependent_metrics()

    def _update_font_dependent_metrics(self):
        self.font_metrics = QFontMetrics(self.font())
        self.char_height = self.font_metrics.height()
        self.tab_fixed_width = int(self.font_metrics.averageCharWidth() * 2.5) + \
                               2 * self.horizontal_padding
        self.tab_fixed_width = max(self.tab_fixed_width, 40)

    def setFont(self, font: QFont):
        super().setFont(font)
        self._update_font_dependent_metrics()
        self.updateGeometry()
        self.update()

    def tabSizeHint(self, index: int) -> QSize:
        text = self.tabText(index)
        if not text:
            return QSize(self.tab_fixed_width, self.char_height + 2 * self.vertical_padding)
        text_visual_height = self.char_height * len(text) + 2 * self.vertical_padding
        return QSize(self.tab_fixed_width, text_visual_height)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        if not painter.paintEngine():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        opt = QStyleOptionTab()
        for i in range(self.count()):
            self.initStyleOption(opt, i)
            tab_rect = self.tabRect(i)
            is_selected = (self.currentIndex() == i)
            is_under_mouse = tab_rect.contains(self.mapFromGlobal(QCursor.pos()))
            palette = self.palette()
            if is_selected:
                bg_color = palette.color(QPalette.ColorRole.Highlight)
                text_color = palette.color(QPalette.ColorRole.HighlightedText)
                border_color = bg_color.darker(110)
            elif is_under_mouse:
                bg_color = palette.color(QPalette.ColorRole.Button).lighter(115)
                text_color = palette.color(QPalette.ColorRole.ButtonText)
                border_color = palette.color(QPalette.ColorRole.Midlight)
            else:
                bg_color = palette.color(QPalette.ColorRole.Button)
                text_color = palette.color(QPalette.ColorRole.ButtonText)
                border_color = palette.color(QPalette.ColorRole.Mid)
            painter.save()
            corner_radius = 4
            painter.setBrush(bg_color)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(tab_rect.adjusted(0, 0, -1, -1), corner_radius, corner_radius)
            pen = QPen(border_color); pen.setWidth(1); painter.setPen(pen)
            painter.drawRoundedRect(tab_rect.adjusted(0, 0, -1, -1), corner_radius, corner_radius)
            text_to_draw = self.tabText(i)
            if text_to_draw:
                painter.setPen(text_color); painter.setFont(self.font())
                total_text_block_height = len(text_to_draw) * self.char_height
                drawable_tab_height = tab_rect.height() - 2 * self.vertical_padding
                start_y_offset = self.vertical_padding + self.font_metrics.ascent()
                if total_text_block_height < drawable_tab_height:
                    start_y_offset = (drawable_tab_height - total_text_block_height) / 2 + \
                                     self.vertical_padding + self.font_metrics.ascent()
                current_y = start_y_offset
                for char_code in text_to_draw:
                    char_bound_rect = self.font_metrics.boundingRect(char_code)
                    x_pos = (tab_rect.width() - char_bound_rect.width()) / 2 + tab_rect.left()
                    draw_point_y = tab_rect.top() + current_y
                    if draw_point_y < tab_rect.bottom() - self.font_metrics.descent():
                        painter.drawText(int(x_pos), int(draw_point_y), char_code)
                    current_y += self.char_height
            painter.restore()

# --- カスタムタブウィジェット (変更なし) ---
class VerticalTabWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        custom_tab_bar = VerticalTabBar(self)
        self.setTabBar(custom_tab_bar)
        self.setTabPosition(QTabWidget.TabPosition.West)

# --- 関連漢字データを準備するワーカースレッド (変更なし) ---
class RelatedKanjiWorker(QThread):
    result_ready = Signal(int, dict, str, int)
    error_occurred = Signal(int, str)

    def __init__(self, process_id, input_char, kanji_data, radical_data, font_family, font_pixel_size, parent=None):
        super().__init__(parent)
        self.process_id = process_id
        self.input_char = input_char
        self.kanji_data = kanji_data
        self.radical_data = radical_data
        self.font_family = font_family
        self.font_pixel_size = font_pixel_size
        self._is_cancelled_flag = False
        self.mutex = QMutex()

    def run(self):
        try:
            results_dict = find_kanji_by_shared_radical_per_radical(
                self.input_char, self.kanji_data, self.radical_data
            )
            with QMutexLocker(self.mutex):
                if self._is_cancelled_flag:
                    return
            self.result_ready.emit(self.process_id, results_dict, self.font_family, self.font_pixel_size)
        except Exception as e:
            with QMutexLocker(self.mutex):
                if self._is_cancelled_flag:
                    return
            self.error_occurred.emit(self.process_id, str(e))

    def cancel(self):
        with QMutexLocker(self.mutex):
            self._is_cancelled_flag = True


# --- Asynchronous Workers (SaveGlyphWorker, SaveGuiStateWorker, SaveAdvanceWidthWorker, SaveReferenceImageWorker は変更なし) ---
class WorkerBaseSignals(QObject):
    finished = Signal()
    error = Signal(str)

class SaveGlyphWorkerSignals(WorkerBaseSignals):
    result = Signal(str, QPixmap, bool)

class SaveGlyphWorker(QRunnable):
    def __init__(self, db_path: str, character: str, pixmap: QPixmap, is_vrt2_glyph: bool = False):
        super().__init__()
        self.db_path = db_path
        self.character = character
        self.pixmap = pixmap.copy()
        self.is_vrt2_glyph = is_vrt2_glyph
        self.signals = SaveGlyphWorkerSignals()

    @Slot()
    def run(self):
        try:
            byte_array = QByteArray()
            buffer = QBuffer(byte_array)
            buffer.open(QIODevice.WriteOnly)
            qimage = self.pixmap.toImage()
            qimage.save(buffer, "PNG")
            image_data = byte_array.data()

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if self.is_vrt2_glyph:
                cursor.execute("""
                    INSERT OR REPLACE INTO vrt2_glyphs (character, image_data, last_modified)
                    VALUES (?, ?, CURRENT_TIMESTAMP)
                """, (self.character, image_data))
            else:
                cursor.execute("""
                    UPDATE glyphs
                    SET image_data = ?, last_modified = CURRENT_TIMESTAMP
                    WHERE character = ?
                """, (image_data, self.character))

                if cursor.rowcount == 0:
                    # Character might not exist if char set was modified and this is a delayed save.
                    # Or if .notdef was somehow targeted for standard update without existing.
                    # For now, we'll let it silently fail to insert if not found, to avoid creating
                    # new glyph entries outside of the designated character set management.
                    pass 

            conn.commit()
            conn.close()
            self.signals.result.emit(self.character, self.pixmap, self.is_vrt2_glyph)
        except Exception as e:
            import traceback
            err_type = "vrt2 glyph" if self.is_vrt2_glyph else "glyph"
            self.signals.error.emit(f"Error saving {err_type} '{self.character}': {e}\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()

class SaveGuiStateWorker(QRunnable):
    def __init__(self, db_path: str, key: str, value: str):
        super().__init__()
        self.db_path = db_path
        self.key = key
        self.value = str(value) # Ensure value is string
        self.signals = WorkerBaseSignals()

    @Slot()
    def run(self):
        try:
            if not self.db_path:
                self.signals.error.emit(f"Cannot save GUI setting '{self.key}': DB path not set.")
                return
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT OR REPLACE INTO project_settings (key, value) VALUES (?, ?)", (self.key, self.value))
            conn.commit()
            conn.close()
        except Exception as e:
            import traceback
            self.signals.error.emit(f"Error saving GUI setting '{self.key}': {e}\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()

class SaveAdvanceWidthWorker(QRunnable):
    def __init__(self, db_path: str, character: str, advance_width: int):
        super().__init__()
        self.db_path = db_path
        self.character = character
        self.advance_width = advance_width
        self.signals = WorkerBaseSignals()

    @Slot()
    def run(self):
        try:
            if not self.db_path:
                self.signals.error.emit(f"Cannot save advance width for '{self.character}': DB path not set.")
                return
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE glyphs SET advance_width = ? WHERE character = ?", (self.advance_width, self.character))
            conn.commit()
            conn.close()
        except Exception as e:
            import traceback
            self.signals.error.emit(f"Error saving advance width for '{self.character}': {e}\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()

class SaveReferenceImageWorkerSignals(WorkerBaseSignals):
    result = Signal(str, QPixmap, bool) # char, pixmap_or_None, is_vrt2

class SaveReferenceImageWorker(QRunnable):
    def __init__(self, db_path: str, character: str, pixmap: Optional[QPixmap], is_vrt2_glyph: bool = False):
        super().__init__()
        self.db_path = db_path
        self.character = character
        self.pixmap = pixmap.copy() if pixmap else None # Store copy or None
        self.is_vrt2_glyph = is_vrt2_glyph
        self.signals = SaveReferenceImageWorkerSignals()

    @Slot()
    def run(self):
        try:
            image_data: Optional[bytes] = None
            if self.pixmap:
                byte_array = QByteArray()
                buffer = QBuffer(byte_array)
                buffer.open(QIODevice.WriteOnly)
                qimage = self.pixmap.toImage()
                qimage.save(buffer, "PNG")
                image_data = byte_array.data()
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            if self.is_vrt2_glyph:
                cursor.execute("""
                    UPDATE vrt2_glyphs SET reference_image_data = ? WHERE character = ?
                """, (image_data, self.character))
            else:
                cursor.execute("""
                    UPDATE glyphs SET reference_image_data = ? WHERE character = ?
                """, (image_data, self.character))

            if cursor.rowcount == 0:
                # This can happen if the character was removed from the set,
                # or if it's a new VRT2 glyph not yet in vrt2_glyphs table.
                # For VRT2, if it's not there, it might be an error unless creation is implicit.
                # For now, let's treat it as an error if no rows were updated.
                table_name = "vrt2_glyphs" if self.is_vrt2_glyph else "glyphs"
                self.signals.error.emit(f"Error saving reference image for '{self.character}' in {table_name}: Character not found or no update occurred.")
                conn.close()
                return

            conn.commit()
            conn.close()
            self.signals.result.emit(self.character, self.pixmap, self.is_vrt2_glyph) 
        except Exception as e:
            import traceback
            err_type = "vrt2 reference" if self.is_vrt2_glyph else "reference"
            self.signals.error.emit(f"Error saving {err_type} image for '{self.character}': {e}\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()


# --- Worker for loading project data (Modified for batch loading) ---
class LoadProjectWorkerSignals(QObject):
    # Basic project info (non-glyph data)
    basic_info_loaded = Signal(dict) 
    # Batch of glyph data: list of (char, QPixmap), is_vrt2_batch, is_last_batch_for_type
    glyph_batch_loaded = Signal(list, bool, bool) 
    load_progress = Signal(int, str) # progress (0-100), message
    error = Signal(str)
    finished = Signal()

class LoadProjectWorker(QRunnable):
    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self.signals = LoadProjectWorkerSignals()

    def _load_and_emit_glyph_data_in_batches(self, db_manager: "DatabaseManager", is_vrt2_glyphs: bool):
        # Load raw data (char, image_bytes)
        if is_vrt2_glyphs:
            raw_glyph_data = db_manager.get_all_defined_nrvg_with_preview_data()
            progress_msg_prefix = "特別VRT2グリフ"
        else:
            raw_glyph_data = db_manager.get_all_glyphs_with_preview_data()
            progress_msg_prefix = "標準グリフ"
        
        total_glyphs = len(raw_glyph_data)
        if total_glyphs == 0:
            self.signals.glyph_batch_loaded.emit([], is_vrt2_glyphs, True) # Empty batch, but it's the last
            self.signals.load_progress.emit(100, f"{progress_msg_prefix} 読み込み完了 (0件)")
            return

        for i in range(0, total_glyphs, GLYPH_LOAD_BATCH_SIZE):
            batch_raw = raw_glyph_data[i:i + GLYPH_LOAD_BATCH_SIZE]
            batch_pixmaps = []
            for char, img_bytes in batch_raw:
                pixmap = None
                if img_bytes:
                    pixmap = QPixmap()
                    pixmap.loadFromData(img_bytes)
                batch_pixmaps.append((char, pixmap))
            
            is_last_batch = (i + GLYPH_LOAD_BATCH_SIZE >= total_glyphs)
            self.signals.glyph_batch_loaded.emit(batch_pixmaps, is_vrt2_glyphs, is_last_batch)
            
            progress_percent = int(((i + len(batch_pixmaps)) / total_glyphs) * 100)
            self.signals.load_progress.emit(progress_percent, f"{progress_msg_prefix} 読み込み中 ({i+len(batch_pixmaps)}/{total_glyphs})")
            QThread.msleep(10) # Give main thread a chance to process events, small delay

    @Slot()
    def run(self):
        try:
            db_manager = DatabaseManager(self.db_path) # Create a local instance for this thread

            self.signals.load_progress.emit(0, "基本設定を読み込み中...")
            char_set_list = db_manager.get_project_character_set()
            r_vrt2_list = db_manager.get_rotated_vrt2_character_set()
            nr_vrt2_list = db_manager.get_non_rotated_vrt2_character_set()
            
            font_name = db_manager.load_gui_setting(SETTING_FONT_NAME, DEFAULT_FONT_NAME)
            font_weight = db_manager.load_gui_setting(SETTING_FONT_WEIGHT, DEFAULT_FONT_WEIGHT)
            
            ref_opacity_str = db_manager.load_gui_setting(SETTING_REFERENCE_IMAGE_OPACITY, str(DEFAULT_REFERENCE_IMAGE_OPACITY))
            try: ref_opacity_val = float(ref_opacity_str if ref_opacity_str else DEFAULT_REFERENCE_IMAGE_OPACITY)
            except ValueError: ref_opacity_val = DEFAULT_REFERENCE_IMAGE_OPACITY

            gui_settings = {
                SETTING_PEN_WIDTH: db_manager.load_gui_setting(SETTING_PEN_WIDTH, str(DEFAULT_PEN_WIDTH)),
                SETTING_PEN_SHAPE: db_manager.load_gui_setting(SETTING_PEN_SHAPE, DEFAULT_PEN_SHAPE),
                SETTING_CURRENT_TOOL: db_manager.load_gui_setting(SETTING_CURRENT_TOOL, DEFAULT_CURRENT_TOOL),
                SETTING_MIRROR_MODE: db_manager.load_gui_setting(SETTING_MIRROR_MODE, str(DEFAULT_MIRROR_MODE)),
                SETTING_GLYPH_MARGIN_WIDTH: db_manager.load_gui_setting(SETTING_GLYPH_MARGIN_WIDTH, str(DEFAULT_GLYPH_MARGIN_WIDTH)),
                SETTING_REFERENCE_IMAGE_OPACITY: str(ref_opacity_val)
            }

            kv_font_actual_name = db_manager.load_gui_setting(SETTING_KV_CURRENT_FONT, "")
            kv_display_mode_str = db_manager.load_gui_setting(SETTING_KV_DISPLAY_MODE, str(MainWindow.KV_MODE_WRITTEN_GLYPHS)) # Default to written if not found
            try:
                kv_display_mode_val = int(kv_display_mode_str)
                if kv_display_mode_val not in [MainWindow.KV_MODE_FONT_DISPLAY, MainWindow.KV_MODE_WRITTEN_GLYPHS, MainWindow.KV_MODE_HIDDEN]:
                    kv_display_mode_val = MainWindow.KV_MODE_WRITTEN_GLYPHS
            except ValueError:
                kv_display_mode_val = MainWindow.KV_MODE_WRITTEN_GLYPHS # Fallback
            
            last_active_glyph_char = db_manager.load_gui_setting(SETTING_LAST_ACTIVE_GLYPH)

            basic_loaded_data = {
                'char_set_list': char_set_list,
                'r_vrt2_list': r_vrt2_list,
                'nr_vrt2_list': nr_vrt2_list,
                'font_name': font_name,
                'font_weight': font_weight,
                'gui_settings': gui_settings,
                'kv_font_actual_name': kv_font_actual_name,
                'kv_display_mode_val': kv_display_mode_val,
                'last_active_glyph_char': last_active_glyph_char,
            }
            self.signals.basic_info_loaded.emit(basic_loaded_data)
            self.signals.load_progress.emit(10, "基本設定 読み込み完了") # Arbitrary progress update

            # Load standard glyphs in batches
            self._load_and_emit_glyph_data_in_batches(db_manager, is_vrt2_glyphs=False)
            
            # Load special VRT2 glyphs in batches
            self._load_and_emit_glyph_data_in_batches(db_manager, is_vrt2_glyphs=True)

            self.signals.load_progress.emit(100, "すべてのデータ読み込み完了")

        except Exception as e:
            import traceback
            self.signals.error.emit(f"プロジェクトデータの読み込み中にエラーが発生しました: {e}\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()


# --- Worker for creating project DB (変更なし) ---
class CreateProjectWorkerSignals(QObject):
    finished = Signal()
    error = Signal(str)
    project_created = Signal(str)

class CreateProjectWorker(QRunnable):
    def __init__(self, filepath: str, characters: List[str],
                 r_vrt2_chars: List[str], nr_vrt2_chars: List[str]):
        super().__init__()
        self.filepath = filepath
        self.characters = characters
        self.r_vrt2_chars = r_vrt2_chars
        self.nr_vrt2_chars = nr_vrt2_chars
        self.signals = CreateProjectWorkerSignals()

    @Slot()
    def run(self):
        try:
            temp_db_manager = DatabaseManager()
            temp_db_manager.create_project_db(
                self.filepath,
                self.characters,
                self.r_vrt2_chars,
                self.nr_vrt2_chars
            )
            self.signals.project_created.emit(self.filepath)
        except Exception as e:
            import traceback
            self.signals.error.emit(f"プロジェクトデータベースの作成に失敗しました: {e}\n{traceback.format_exc()}")
        finally:
            self.signals.finished.emit()


# --- Database Manager (変更なし) ---
class DatabaseManager:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path

    def _get_connection(self):
        if not self.db_path:
            raise ConnectionError("Database path not set.")
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def connect_db(self, db_path: str):
        self.db_path = db_path

    def _create_empty_image_data(self) -> bytes:
        image = QImage(QSize(CANVAS_IMAGE_WIDTH, CANVAS_IMAGE_HEIGHT), QImage.Format_ARGB32_Premultiplied)
        image.fill(QColor(Qt.white)) # Fill with white
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.WriteOnly)
        image.save(buffer, "PNG")
        return byte_array.data()

    def create_project_db(self, filepath: str, characters: List[str],
                          r_vrt2_chars: List[str], nr_vrt2_chars: List[str]):
        self.db_path = filepath
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS glyphs (
                character TEXT PRIMARY KEY,
                unicode_val INTEGER, 
                image_data BLOB,
                reference_image_data BLOB DEFAULT NULL, 
                advance_width INTEGER DEFAULT 1000,
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS project_settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS vrt2_glyphs (
                character TEXT PRIMARY KEY, 
                image_data BLOB,
                reference_image_data BLOB DEFAULT NULL, 
                last_modified TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        initial_char_string = "".join(characters)
        cursor.execute("INSERT OR REPLACE INTO project_settings (key, value) VALUES (?, ?)",
                       ('character_set', initial_char_string))
        r_vrt2_string = "".join(r_vrt2_chars)
        cursor.execute("INSERT OR REPLACE INTO project_settings (key, value) VALUES (?, ?)",
                       (SETTING_ROTATED_VRT2_CHARS, r_vrt2_string))
        nr_vrt2_string = "".join(nr_vrt2_chars)
        cursor.execute("INSERT OR REPLACE INTO project_settings (key, value) VALUES (?, ?)",
                       (SETTING_NON_ROTATED_VRT2_CHARS, nr_vrt2_string))
        
        empty_image_bytes = self._create_empty_image_data()
        cursor.execute("""
            INSERT OR IGNORE INTO glyphs (character, unicode_val, image_data, advance_width)
            VALUES (?, ?, ?, ?)
        """, ('.notdef', -1, empty_image_bytes, DEFAULT_ADVANCE_WIDTH)) # Ensure .notdef always has image data

        whitespace_chars_to_initialize = [' ', '　', '\t'] # Common whitespace that might need an empty image
        for char_val in characters:
            if len(char_val) != 1:
                # Log or handle multi-character strings if necessary, for now skip
                continue
            
            image_data_for_char = None # Default to NULL (no image data written)
            if char_val in whitespace_chars_to_initialize:
                 image_data_for_char = empty_image_bytes # Provide empty image for specific whitespace

            try:
                unicode_val = ord(char_val)
                cursor.execute("""
                    INSERT OR IGNORE INTO glyphs (character, unicode_val, image_data, advance_width)
                    VALUES (?, ?, ?, ?)
                """, (char_val, unicode_val, image_data_for_char, DEFAULT_ADVANCE_WIDTH))
            except TypeError: # ord() on non-char
                pass # Skip if character is not a single valid char for ord()

        # Initialize vrt2_glyphs table entries based on nr_vrt2_chars
        # These start without image_data, which will be added by user or transfer
        for char_val in nr_vrt2_chars:
            if len(char_val) == 1: # Ensure single character
                 cursor.execute("INSERT OR IGNORE INTO vrt2_glyphs (character) VALUES (?)", (char_val,))
        
        self._save_default_gui_settings(cursor)

        conn.commit()
        conn.close()

    def _save_default_gui_settings(self, cursor: sqlite3.Cursor):
        defaults = {
            SETTING_PEN_WIDTH: str(DEFAULT_PEN_WIDTH),
            SETTING_PEN_SHAPE: DEFAULT_PEN_SHAPE,
            SETTING_CURRENT_TOOL: DEFAULT_CURRENT_TOOL,
            SETTING_MIRROR_MODE: str(DEFAULT_MIRROR_MODE),
            SETTING_GLYPH_MARGIN_WIDTH: str(DEFAULT_GLYPH_MARGIN_WIDTH),
            SETTING_LAST_ACTIVE_GLYPH: "", # No last active glyph initially
            SETTING_FONT_NAME: DEFAULT_FONT_NAME,
            SETTING_FONT_WEIGHT: DEFAULT_FONT_WEIGHT,
            SETTING_REFERENCE_IMAGE_OPACITY: str(DEFAULT_REFERENCE_IMAGE_OPACITY),
            SETTING_KV_CURRENT_FONT: "", # No Kanji Viewer font initially
            SETTING_KV_DISPLAY_MODE: str(DEFAULT_KV_MODE_FOR_SETTINGS), # Default KV mode
        }
        for key, value in defaults.items():
            cursor.execute("INSERT OR IGNORE INTO project_settings (key, value) VALUES (?, ?)", (key, value))

    def get_project_character_set(self) -> List[str]: return self._get_char_set_from_settings('character_set')
    def get_rotated_vrt2_character_set(self) -> List[str]: return self._get_char_set_from_settings(SETTING_ROTATED_VRT2_CHARS)
    def get_non_rotated_vrt2_character_set(self) -> List[str]: return self._get_char_set_from_settings(SETTING_NON_ROTATED_VRT2_CHARS)

    def _get_char_set_from_settings(self, setting_key: str) -> List[str]:
        if not self.db_path: return []
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("SELECT value FROM project_settings WHERE key = ?", (setting_key,))
        row = cursor.fetchone(); conn.close()
        if row and row['value']:
            # Ensure uniqueness and sort by Unicode value for consistency
            seen = set(); unique_ordered_chars = []
            for c in row['value']:
                if len(c) == 1 and c not in seen: unique_ordered_chars.append(c); seen.add(c)
            return sorted(unique_ordered_chars, key=ord) # Sort by Unicode codepoint
        return []

    def update_project_character_set(self, characters: List[str]): self._update_char_set_in_settings('character_set', characters, is_main_set=True)
    def update_rotated_vrt2_character_set(self, characters: List[str]): self._update_char_set_in_settings(SETTING_ROTATED_VRT2_CHARS, characters)
    def update_non_rotated_vrt2_character_set(self, characters: List[str]): self._update_char_set_in_settings(SETTING_NON_ROTATED_VRT2_CHARS, characters, updates_vrt2_table=True)

    def _update_char_set_in_settings(self, setting_key: str, characters: List[str],
                                     is_main_set: bool = False, updates_vrt2_table: bool = False):
        if not self.db_path: return
        conn = self._get_connection(); cursor = conn.cursor()
        valid_characters = [c for c in characters if len(c) == 1]
        new_char_string = "".join(valid_characters)
        cursor.execute("INSERT OR REPLACE INTO project_settings (key, value) VALUES (?, ?)", (setting_key, new_char_string))

        table_to_check = "glyphs" if is_main_set else ("vrt2_glyphs" if updates_vrt2_table else None)
        if table_to_check:
            cursor.execute(f"SELECT character FROM {table_to_check}")
            db_chars_query_result = cursor.fetchall()
            # Exclude .notdef from main set for addition/deletion logic
            db_chars = {row['character'] for row in db_chars_query_result if not (is_main_set and row['character'] == '.notdef')}

            new_chars_set = set(valid_characters)
            empty_image_bytes_for_new_main_glyphs = self._create_empty_image_data() if is_main_set else None

            for char_to_add in new_chars_set - db_chars:
                if is_main_set:
                    cursor.execute("INSERT OR IGNORE INTO glyphs (character, unicode_val, image_data, advance_width) VALUES (?, ?, ?, ?)",
                                   (char_to_add, ord(char_to_add), empty_image_bytes_for_new_main_glyphs, DEFAULT_ADVANCE_WIDTH))
                elif updates_vrt2_table: # This is for non-rotated VRT2 set
                    # Ensure character exists in glyphs table first, if not, add it (without image data initially)
                    cursor.execute("SELECT 1 FROM glyphs WHERE character = ?", (char_to_add,))
                    if not cursor.fetchone(): # If not in main glyphs table, add it
                         cursor.execute("INSERT OR IGNORE INTO glyphs (character, unicode_val, advance_width) VALUES (?, ?, ?)",
                                   (char_to_add, ord(char_to_add), DEFAULT_ADVANCE_WIDTH))
                    cursor.execute("INSERT OR IGNORE INTO vrt2_glyphs (character) VALUES (?)", (char_to_add,))

            for char_to_remove in db_chars - new_chars_set:
                if char_to_remove == '.notdef' and is_main_set: continue # Never remove .notdef
                cursor.execute(f"DELETE FROM {table_to_check} WHERE character = ?", (char_to_remove,))
        
        conn.commit(); conn.close()

    def load_glyph_image(self, character: str, is_vrt2: bool = False) -> Optional[QPixmap]:
        if not self.db_path: return None
        conn = self._get_connection(); cursor = conn.cursor()
        table = "vrt2_glyphs" if is_vrt2 else "glyphs"
        cursor.execute(f"SELECT image_data FROM {table} WHERE character = ?", (character,))
        row = cursor.fetchone(); conn.close()
        if row and row['image_data']: pixmap = QPixmap(); pixmap.loadFromData(row['image_data']); return pixmap
        return None

    def load_reference_image(self, character: str) -> Optional[QPixmap]:
        if not self.db_path: return None
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute(f"SELECT reference_image_data FROM glyphs WHERE character = ?", (character,))
        row = cursor.fetchone(); conn.close()
        if row and row['reference_image_data']: pixmap = QPixmap(); pixmap.loadFromData(row['reference_image_data']); return pixmap
        return None
    
    def load_vrt2_glyph_reference_image(self, character: str) -> Optional[QPixmap]:
        if not self.db_path: return None
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("SELECT reference_image_data FROM vrt2_glyphs WHERE character = ?", (character,))
        row = cursor.fetchone(); conn.close()
        if row and row['reference_image_data']:
            pixmap = QPixmap(); pixmap.loadFromData(row['reference_image_data']); return pixmap
        return None

    def save_vrt2_glyph_image(self, character: str, pixmap: QPixmap): # Typically handled by SaveGlyphWorker
        if not self.db_path:
            # Handle error: DB path not set
            return
        byte_array = QByteArray(); buffer = QBuffer(byte_array); buffer.open(QIODevice.WriteOnly)
        qimage = pixmap.toImage(); qimage.save(buffer, "PNG"); image_data = byte_array.data()
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO vrt2_glyphs (character, image_data, last_modified) VALUES (?, ?, CURRENT_TIMESTAMP)", (character, image_data))
        conn.commit(); conn.close()

    def load_glyph_advance_width(self, character: str) -> int:
        if not self.db_path: return DEFAULT_ADVANCE_WIDTH
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("SELECT advance_width FROM glyphs WHERE character = ?", (character,))
        row = cursor.fetchone(); conn.close()
        return row['advance_width'] if row and row['advance_width'] is not None else DEFAULT_ADVANCE_WIDTH

    def save_glyph_advance_width(self, character: str, advance_width: int): # Typically handled by SaveAdvanceWidthWorker
        if not self.db_path: return
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("UPDATE glyphs SET advance_width = ? WHERE character = ?", (advance_width, character))
        conn.commit(); conn.close()

    def get_all_glyphs_with_preview_data(self) -> List[Tuple[str, Optional[bytes]]]:
        if not self.db_path: return []
        conn = self._get_connection(); cursor = conn.cursor()
        # Order by unicode_val then character for consistent ordering, especially for .notdef
        cursor.execute("SELECT character, unicode_val, image_data FROM glyphs ORDER BY unicode_val ASC, character ASC")
        results = []
        all_db_rows = cursor.fetchall()
        conn.close()
        for row in all_db_rows:
            char_val = row['character']
            image_bytes = row['image_data'] if row['image_data'] else None
            results.append((char_val, image_bytes))
        return results

    def get_all_defined_nrvg_with_preview_data(self) -> List[Tuple[str, Optional[bytes]]]:
        if not self.db_path: return []
        nrvg_chars_from_settings = self.get_non_rotated_vrt2_character_set() # Already sorted by unicode
        if not nrvg_chars_from_settings: return []
        
        conn = self._get_connection(); cursor = conn.cursor(); results = []
        for char_val in nrvg_chars_from_settings: # Iterate in sorted order
            cursor.execute("SELECT image_data FROM vrt2_glyphs WHERE character = ?", (char_val,))
            row = cursor.fetchone()
            image_bytes = None
            if row and row['image_data']: # Glyph exists in vrt2_glyphs and has image data
                image_bytes = row['image_data']
            # We include the char even if no image_data, as it's defined in the set
            results.append((char_val, image_bytes)) 
        conn.close(); return results


    def load_gui_setting(self, key: str, default_value: Optional[str] = None) -> Optional[str]:
        if not self.db_path: return default_value
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("SELECT value FROM project_settings WHERE key = ?", (key,))
        row = cursor.fetchone(); conn.close()
        return row['value'] if row else default_value

    # Methods for batch import, direct DB interaction for performance
    def update_glyph_image_data_bytes(self, character: str, image_data: bytes) -> bool:
        if not self.db_path: return False
        conn = self._get_connection(); cursor = conn.cursor()
        # Ensure character exists before updating
        cursor.execute("SELECT 1 FROM glyphs WHERE character = ?", (character,)); exists = cursor.fetchone()
        if not exists: conn.close(); return False # Character not in project
        try:
            cursor.execute("UPDATE glyphs SET image_data = ?, last_modified = CURRENT_TIMESTAMP WHERE character = ?", (image_data, character))
            updated_rows = cursor.rowcount; conn.commit(); return updated_rows > 0
        except Exception: conn.rollback(); return False
        finally: conn.close()

    def update_glyph_reference_image_data_bytes(self, character: str, image_data: Optional[bytes]) -> bool:
        if not self.db_path: return False
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM glyphs WHERE character = ?", (character,)); exists = cursor.fetchone()
        if not exists: conn.close(); return False
        try:
            cursor.execute("UPDATE glyphs SET reference_image_data = ? WHERE character = ?", (image_data, character))
            updated_rows = cursor.rowcount; conn.commit(); return updated_rows > 0
        except Exception: conn.rollback(); return False
        finally: conn.close()

    def update_vrt2_glyph_reference_image_data_bytes(self, character: str, image_data: Optional[bytes]) -> bool:
        if not self.db_path: return False
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM vrt2_glyphs WHERE character = ?", (character,)); exists = cursor.fetchone()
        if not exists: conn.close(); return False
        try:
            cursor.execute("UPDATE vrt2_glyphs SET reference_image_data = ? WHERE character = ?", (image_data, character))
            updated_rows = cursor.rowcount; conn.commit(); return updated_rows > 0
        except Exception: conn.rollback(); return False
        finally: conn.close()



# --- Canvas Widget (変更なし) ---
class Canvas(QWidget):
    pen_width_changed = Signal(int)
    tool_changed = Signal(str)
    undo_redo_state_changed = Signal(bool, bool)
    glyph_modified_signal = Signal(str, QPixmap, bool) # char, pixmap, is_vrt2
    glyph_margin_width_changed = Signal(int)
    glyph_advance_width_changed = Signal(int) # advance_width (shared for H/V)

    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.ClickFocus); self.setAcceptDrops(True)
        self.ascender_height_for_baseline = DEFAULT_ASCENDER_HEIGHT # Example, not fully used yet
        self.glyph_margin_width: int = DEFAULT_GLYPH_MARGIN_WIDTH
        self.current_glyph_character: Optional[str] = None
        self.editing_vrt2_glyph: bool = False
        self.current_glyph_advance_width: int = DEFAULT_ADVANCE_WIDTH
        self.image_size = QSize(CANVAS_IMAGE_WIDTH, CANVAS_IMAGE_HEIGHT)
        self.setFixedSize(self.image_size.width() + 2 * VIRTUAL_MARGIN, self.image_size.height() + 2 * VIRTUAL_MARGIN)
        self.image = QPixmap(self.image_size); self.image.fill(QColor(Qt.white))
        self.reference_image: Optional[QPixmap] = None
        self.reference_image_opacity: float = DEFAULT_REFERENCE_IMAGE_OPACITY
        self.pen_width = DEFAULT_PEN_WIDTH
        self.pen_shape = Qt.RoundCap if DEFAULT_PEN_SHAPE == "丸" else Qt.SquareCap
        self.current_pen_color = QColor(Qt.black); self.last_brush_color = QColor(Qt.black)
        self.drawing = False; self.stroke_points: list[QPointF] = []; self.current_path = QPainterPath()
        self.mirror_mode = DEFAULT_MIRROR_MODE; self.move_mode = False; self.moving_image = False
        self.move_start_pos = QPointF(); self.move_offset = QPointF()
        self.undo_stack: List[QPixmap] = []; self.redo_stack: List[QPixmap] = []
        self.current_tool = DEFAULT_CURRENT_TOOL

    def set_glyph_margin_width(self, width: int):
        max_margin = min(self.image_size.width(), self.image_size.height()) // 4
        new_width = max(0, min(width, max_margin))
        if self.glyph_margin_width != new_width:
            self.glyph_margin_width = new_width
            self.glyph_margin_width_changed.emit(self.glyph_margin_width); self.update()

    def set_current_glyph_advance_width(self, width: int):
        if width >= 0: # Allow 0 for advance width
            self.current_glyph_advance_width = width
            self.glyph_advance_width_changed.emit(width); self.update()

    def load_glyph(self, character: str, pixmap: Optional[QPixmap], reference_pixmap: Optional[QPixmap], advance_width: int, is_vrt2: bool = False):
        self.current_glyph_character = character; self.editing_vrt2_glyph = is_vrt2
        self.current_glyph_advance_width = advance_width; self.reference_image = reference_pixmap
        if pixmap:
            self.image = pixmap.copy()
            if self.image.size() != self.image_size: # Ensure image is correct canvas size
                temp_target_pixmap = QPixmap(self.image_size); temp_target_pixmap.fill(Qt.white)
                painter = QPainter(temp_target_pixmap); painter.drawPixmap(0, 0, self.image); painter.end()
                self.image = temp_target_pixmap
        else: self.image = QPixmap(self.image_size); self.image.fill(QColor(Qt.white))
        self.undo_stack = []; self.redo_stack = []
        self._save_state_to_undo_stack(is_initial_load=True); self.update()

    def set_reference_image_opacity(self, opacity: float): self.reference_image_opacity = max(0.0, min(1.0, opacity)); self.update()
    def get_current_image_and_type(self) -> Tuple[QPixmap, bool]: return self.image.copy(), self.editing_vrt2_glyph
    def get_current_image(self) -> QPixmap: return self.image.copy() # Convenience
    def _emit_undo_redo_state(self): self.undo_redo_state_changed.emit(len(self.undo_stack) > 1, bool(self.redo_stack))

    def _save_state_to_undo_stack(self, is_initial_load: bool = False):
        if len(self.undo_stack) >= MAX_HISTORY_SIZE: self.undo_stack.pop(0)
        self.undo_stack.append(self.image.copy())
        if not is_initial_load: self.redo_stack.clear() # Clear redo stack on new action
        self._emit_undo_redo_state()

    def undo(self):
        if len(self.undo_stack) > 1: # Need at least one state to revert to
            popped_state = self.undo_stack.pop(); self.redo_stack.append(popped_state)
            self.image = self.undo_stack[-1].copy(); self.update(); self._emit_undo_redo_state()
            if self.current_glyph_character: self.glyph_modified_signal.emit(self.current_glyph_character, self.image.copy(), self.editing_vrt2_glyph)

    def redo(self):
        if self.redo_stack:
            popped_state = self.redo_stack.pop(); self.undo_stack.append(popped_state)
            self.image = popped_state.copy(); self.update(); self._emit_undo_redo_state()
            if self.current_glyph_character: self.glyph_modified_signal.emit(self.current_glyph_character, self.image.copy(), self.editing_vrt2_glyph)

    def set_pen_width(self, width: int):
        self.pen_width = width; self.pen_width_changed.emit(self.pen_width)
        if self.drawing and self.stroke_points: self._rebuild_current_path_from_stroke_points(finalize=False)

    def set_pen_shape(self, shape_name: str): 
        if shape_name == "丸": self.pen_shape = Qt.RoundCap
        elif shape_name == "四角": self.pen_shape = Qt.SquareCap
        if self.drawing and self.stroke_points: self._rebuild_current_path_from_stroke_points(finalize=False)

    def set_current_pen_color(self, color: QColor): 
        self.current_pen_color = color
        if self.drawing and self.stroke_points: self._rebuild_current_path_from_stroke_points(finalize=False)

    def set_eraser_mode(self):
        if self.move_mode: self.move_mode = False; self.unsetCursor() # Ensure move mode is off
        if self.current_pen_color != Qt.white: self.last_brush_color = self.current_pen_color # Save last brush color
        self.set_current_pen_color(QColor(Qt.white)); self.drawing = False; self.current_tool = "eraser"
        self.tool_changed.emit("eraser"); self.update()

    def set_brush_mode(self):
        if self.move_mode: self.move_mode = False; self.unsetCursor() # Ensure move mode is off
        self.set_current_pen_color(self.last_brush_color); self.drawing = False; self.current_tool = "brush"
        self.tool_changed.emit("brush"); self.update()

    def set_mirror_mode(self, enabled: bool):
        if self.mirror_mode == enabled: return
        self.mirror_mode = enabled
        # If drawing was in progress, finalize or cancel it to avoid visual artifacts
        if self.drawing: self.drawing = False; self.stroke_points = []; self.current_path = QPainterPath() 
        self.update()

    def set_move_mode(self, enabled: bool):
        if enabled and self.current_tool == "move": return # Already in move mode
        if not enabled and self.current_tool != "move": return # Already not in move mode
        
        self.move_mode = enabled; self.moving_image = False; self.move_offset = QPointF()
        if enabled:
            self.drawing = False # Can't draw and move simultaneously
            self.setCursor(Qt.OpenHandCursor); self.current_tool = "move"; self.tool_changed.emit("move")
        else: # Disabling move mode
            self.unsetCursor()
            # Revert to brush or eraser based on current pen color
            if self.current_tool == "move": # Only if we were actually in move mode
                if self.current_pen_color == QColor(Qt.white): self.set_eraser_mode()
                else: self.set_brush_mode()
        self.update()

    def _map_view_point_to_logical_point(self, view_point: QPointF) -> QPointF:
        image_local_view_point = view_point - QPointF(VIRTUAL_MARGIN, VIRTUAL_MARGIN)
        if self.mirror_mode: # Apply mirroring transformation
            return QPointF(self.image_size.width() - image_local_view_point.x(), image_local_view_point.y())
        return image_local_view_point

    def _generate_path_from_points(self, points: list[QPointF], finalize=False) -> QPainterPath:
        path = QPainterPath(); n = len(points)
        if n == 0: return path
        path.moveTo(points[0])
        if n == 1: path.lineTo(points[0]); return path # Single point draws a dot/square
        
        # Catmull-Rom to Cubic Bezier approximation for smoother strokes
        for i in range(n - 1):
            p1 = points[i]; p2 = points[i+1]
            # Control points (p0 and p3 are for tangent calculation)
            p0 = points[i-1] if i > 0 else p1 # Previous or current if first segment
            p3 = points[i+2] if i + 2 < n else p2 # Next or current if last segment

            # Tension parameter (typically 0.5 for Catmull-Rom)
            alpha = 0.5 
            # Calculate Bezier control points from Catmull-Rom points
            cp1_x = p1.x() + (p2.x() - p0.x()) * alpha / 3.0
            cp1_y = p1.y() + (p2.y() - p0.y()) * alpha / 3.0
            cp2_x = p2.x() - (p3.x() - p1.x()) * alpha / 3.0
            cp2_y = p2.y() - (p3.y() - p1.y()) * alpha / 3.0
            path.cubicTo(QPointF(cp1_x, cp1_y), QPointF(cp2_x, cp2_y), p2)
        return path

    def _rebuild_current_path_from_stroke_points(self, finalize=False): self.current_path = self._generate_path_from_points(self.stroke_points, finalize); self.update()

    def _draw_path_on_painter(self, painter: QPainter, path_to_draw: QPainterPath):
        if path_to_draw.isEmpty(): return
        painter.setRenderHint(QPainter.Antialiasing, True)

        if self.pen_shape == Qt.RoundCap:
            pen = QPen(self.current_pen_color, float(self.pen_width), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen); painter.drawPath(path_to_draw)
        elif self.pen_shape == Qt.SquareCap:
            painter.setPen(Qt.NoPen); painter.setBrush(self.current_pen_color)
            path_length = path_to_draw.length()

            if path_length == 0 and path_to_draw.elementCount() > 0 and \
               path_to_draw.elementAt(0).isMoveTo(): # Single point click for square cap
                first_point = QPointF(path_to_draw.elementAt(0).x, path_to_draw.elementAt(0).y)
                rect_size = float(self.pen_width)
                rect = QRectF(first_point.x() - rect_size / 2.0, first_point.y() - rect_size / 2.0, rect_size, rect_size)
                painter.drawRect(rect)
                return

            # For square cap lines, draw series of overlapping squares along the path
            sampling_step_length = max(1.0, float(self.pen_width) / 10.0) # Sample frequently for dense coverage
            num_samples = 0
            if sampling_step_length > 0: num_samples = max(1, int(path_length / sampling_step_length))
            else: num_samples = 1 # Avoid division by zero, at least one sample

            for i in range(num_samples + 1): # +1 to include the end point
                percent = 0.0
                if num_samples > 0 : percent = float(i) / num_samples
                else: percent = 0.0 # For single point or zero length path

                point_on_path = path_to_draw.pointAtPercent(percent)
                if point_on_path.isNull(): continue # Should not happen if path is valid

                rect_size = float(self.pen_width)
                rect = QRectF(point_on_path.x() - rect_size / 2.0, point_on_path.y() - rect_size / 2.0, rect_size, rect_size)
                painter.drawRect(rect)

    def _apply_image_move(self, view_drag_delta: QPointF):
        # Convert view delta to logical delta considering mirror mode
        logical_offset_x = view_drag_delta.x()
        if self.mirror_mode: logical_offset_x = -view_drag_delta.x() # Mirrored horizontal movement
        logical_offset_y = view_drag_delta.y()

        # Round to nearest pixel to avoid subpixel rendering issues if not desired
        int_logical_offset = QPointF(round(logical_offset_x), round(logical_offset_y)).toPoint()
        if int_logical_offset.x() == 0 and int_logical_offset.y() == 0: return # No actual movement

        moved_image = QPixmap(self.image_size); moved_image.fill(QColor(Qt.white)) # Start with a fresh white pixmap
        painter = QPainter(moved_image)
        painter.drawPixmap(int_logical_offset, self.image) # Draw original image at offset
        painter.end()
        self.image = moved_image # Replace current image

    def paintEvent(self, event: QPaintEvent):
        canvas_painter = QPainter(self)
        canvas_painter.fillRect(self.rect(), QColor(200, 200, 200)) # Background of the widget itself

        image_area_rect_in_widget = QRectF(VIRTUAL_MARGIN, VIRTUAL_MARGIN, self.image_size.width(), self.image_size.height())
        
        # Base drawing area (white background for the glyph)
        canvas_painter.fillRect(image_area_rect_in_widget, Qt.white)
        canvas_painter.save(); canvas_painter.translate(image_area_rect_in_widget.topLeft()) # Origin to image top-left

        # --- Draw glyph content (current image + live stroke if drawing) ---
        temp_pixmap_for_drawing_content = QPixmap(self.image_size)
        temp_pixmap_for_drawing_content.fill(Qt.transparent) # Start with transparent for compositing
        image_content_painter = QPainter(temp_pixmap_for_drawing_content)
        image_content_painter.drawPixmap(0,0, self.image) # Draw committed strokes
        if not self.move_mode and self.drawing and not self.current_path.isEmpty(): # Draw live stroke preview
            self._draw_path_on_painter(image_content_painter, self.current_path)
        image_content_painter.end()

        preview_draw_offset = QPointF(0,0)
        if self.move_mode and self.moving_image:
            preview_draw_offset = self.move_offset # Apply live move preview offset

        # Draw the (potentially moved) content, handling mirror mode
        if self.mirror_mode:
            canvas_painter.save()
            canvas_painter.translate(self.image_size.width(), 0); canvas_painter.scale(-1, 1) # Mirror transform
            mirrored_preview_offset = QPointF(-preview_draw_offset.x(), preview_draw_offset.y()) # Mirror the offset too
            canvas_painter.drawPixmap(mirrored_preview_offset.toPoint(), temp_pixmap_for_drawing_content)
            canvas_painter.restore()
        else:
            canvas_painter.drawPixmap(preview_draw_offset.toPoint(), temp_pixmap_for_drawing_content)

        # --- Draw reference image (if exists and opaque) ---
        if self.reference_image and self.reference_image_opacity > 0.0 and \
           self.reference_image.size() == self.image_size: # Ensure ref image is valid
            canvas_painter.save()
            canvas_painter.setOpacity(self.reference_image_opacity)
            # Using Multiply for better visibility of underlying drawing
            canvas_painter.setCompositionMode(QPainter.CompositionMode_Multiply) 
            if self.mirror_mode: # Reference image should also be mirrored if view is mirrored
                canvas_painter.save()
                canvas_painter.translate(self.image_size.width(), 0); canvas_painter.scale(-1, 1)
                canvas_painter.drawPixmap(QPoint(0,0), self.reference_image)
                canvas_painter.restore()
            else:
                canvas_painter.drawPixmap(QPoint(0,0), self.reference_image)
            canvas_painter.restore()

        # --- Draw Guidelines ---
        img_height = float(self.image_size.height()); img_width = float(self.image_size.width())

        # Advance Width/Height Line (Red, Dotted)
        adv_pen = QPen(QColor(255, 0, 0, 180), 1, Qt.DotLine); adv_pen.setCosmetic(True)
        canvas_painter.setPen(adv_pen)
        if self.editing_vrt2_glyph: # Vertical writing mode, advance is height
            line_y_canvas_adv = (float(self.current_glyph_advance_width) / 1000.0) * img_height
            if line_y_canvas_adv >= 0 : canvas_painter.drawLine(QPointF(0, line_y_canvas_adv), QPointF(img_width, line_y_canvas_adv))
        else: # Horizontal writing mode, advance is width
            line_x_canvas_adv = (float(self.current_glyph_advance_width) / 1000.0) * img_width
            if line_x_canvas_adv >= 0: canvas_painter.drawLine(QPointF(line_x_canvas_adv, 0), QPointF(line_x_canvas_adv, img_height))

        # Thirds Guidelines (Gray, Dashed)
        guideline_pen = QPen(QColor(180, 180, 180), 1, Qt.DashLine); guideline_pen.setCosmetic(True)
        canvas_painter.setPen(guideline_pen)
        for i in range(1, 3): # Draws lines at 1/3 and 2/3
            y_pos = (img_height * i / 3.0); canvas_painter.drawLine(QPointF(0, y_pos), QPointF(img_width, y_pos))
            x_pos = (img_width * i / 3.0); canvas_painter.drawLine(QPointF(x_pos, 0), QPointF(x_pos, img_height))
        
        # Center Crosshair (Light Gray, Solid)
        crosshair_pen = QPen(QColor(150, 150, 150), 1, Qt.SolidLine); crosshair_pen.setCosmetic(True)
        canvas_painter.setPen(crosshair_pen)
        center_x, center_y = img_width / 2.0, img_height / 2.0; CROSSHAIR_ARM_LENGTH = 25 # Length of each arm
        canvas_painter.drawLine(QPointF(center_x - CROSSHAIR_ARM_LENGTH, center_y), QPointF(center_x + CROSSHAIR_ARM_LENGTH, center_y))
        canvas_painter.drawLine(QPointF(center_x, center_y - CROSSHAIR_ARM_LENGTH), QPointF(center_x, center_y + CROSSHAIR_ARM_LENGTH))

        # Glyph Margin (Blue, Dotted)
        base_margin_px = float(self.glyph_margin_width) # Already in pixels relative to canvas
        if base_margin_px > 0:
            margin_pen = QPen(QColor(100, 100, 200), 1, Qt.DotLine); margin_pen.setCosmetic(True)
            canvas_painter.setPen(margin_pen)
            
            # Define margin box based on writing direction and advance width
            left_x, top_y, right_x, bottom_y = 0.0,0.0,0.0,0.0
            if self.editing_vrt2_glyph:
                left_x = base_margin_px
                right_x = img_width - base_margin_px
                top_y = base_margin_px
                advance_edge_y = (float(self.current_glyph_advance_width) / 1000.0) * img_height
                bottom_y = advance_edge_y - base_margin_px
            else: # Horizontal
                left_x = base_margin_px
                advance_edge_x = (float(self.current_glyph_advance_width) / 1000.0) * img_width
                right_x = advance_edge_x - base_margin_px
                top_y = base_margin_px
                bottom_y = img_height - base_margin_px
            
            margin_rect = QRectF(left_x, top_y, right_x - left_x, bottom_y - top_y)
            if margin_rect.isValid() and margin_rect.width() > 0 and margin_rect.height() > 0 :
                canvas_painter.drawRect(margin_rect)
        
        canvas_painter.restore() # Restore from translate to image top-left

        # --- Draw widget border and fill outer margin ---
        # Fill area outside the image_area_rect_in_widget (the VIRTUAL_MARGIN area)
        outer_margin_path = QPainterPath()
        outer_margin_path.setFillRule(Qt.OddEvenFill) # Key for filling space between two rects
        outer_margin_path.addRect(QRectF(self.rect())) # Outer rect (whole widget)
        outer_margin_path.addRect(image_area_rect_in_widget) # Inner rect (image area)
        
        canvas_painter.save()
        canvas_painter.setBrush(QColor(200, 200, 200)) # Match widget background
        canvas_painter.setPen(Qt.NoPen)
        canvas_painter.drawPath(outer_margin_path)
        canvas_painter.restore()

        # Draw border around the image area
        canvas_painter.save()
        border_pen = QPen(Qt.darkGray); border_pen.setWidth(1) # Cosmetic pen might be better if scaling
        canvas_painter.setPen(border_pen); canvas_painter.setBrush(Qt.NoBrush)
        canvas_painter.drawRect(image_area_rect_in_widget.adjusted(0,0,-1,-1)) # adjusted for pixel-perfect border
        canvas_painter.restore()


    def mousePressEvent(self, event: QMouseEvent):
        if not self.current_glyph_character: return # No drawing if no glyph selected
        if self.move_mode:
            if event.button() == Qt.LeftButton:
                # Check if press is within the image area (excluding virtual margin)
                image_rect_in_widget = QRectF(VIRTUAL_MARGIN, VIRTUAL_MARGIN, self.image_size.width(), self.image_size.height())
                if image_rect_in_widget.contains(event.position()):
                    self.moving_image = True; self.move_start_pos = event.position(); self.move_offset = QPointF()
                    self.setCursor(Qt.ClosedHandCursor); self.update()
            return # Consume event if in move mode

        if event.button() == Qt.LeftButton:
            self.drawing = True; self.current_path = QPainterPath(); self.stroke_points = []
            logical_pos = self._map_view_point_to_logical_point(event.position())
            self.stroke_points.append(logical_pos)
            self._rebuild_current_path_from_stroke_points(finalize=False); self.update()

    def mouseMoveEvent(self, event: QMouseEvent):
        if not self.current_glyph_character: return
        if self.move_mode and self.moving_image:
            if event.buttons() & Qt.LeftButton:
                current_pos = event.position(); delta = current_pos - self.move_start_pos
                self.move_offset = delta # Store delta for paintEvent to use
                self.update() # Trigger repaint for live move preview
            return

        if (event.buttons() & Qt.LeftButton) and self.drawing:
            view_pos = event.position()
            logical_pos = self._map_view_point_to_logical_point(view_pos)
            # Add point if it's sufficiently far from the last one to avoid too many points
            if not self.stroke_points or (logical_pos - self.stroke_points[-1]).manhattanLength() >= 1.0:
                self.stroke_points.append(logical_pos)
                self._rebuild_current_path_from_stroke_points(finalize=False); self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.current_glyph_character: return
        if self.move_mode and self.moving_image:
            if event.button() == Qt.LeftButton:
                self._apply_image_move(self.move_offset) # Apply the accumulated move_offset
                self.moving_image = False; self.move_offset = QPointF() # Reset move state
                if self.move_mode: self.setCursor(Qt.OpenHandCursor) # Revert to open hand if still in move mode
                else: self.unsetCursor() # Clear cursor if move mode was toggled off during move
                self._save_state_to_undo_stack() # Save moved state
                if self.current_glyph_character: self.glyph_modified_signal.emit(self.current_glyph_character, self.image.copy(), self.editing_vrt2_glyph)
                self.update()
            return

        if event.button() == Qt.LeftButton and self.drawing:
            if self.stroke_points: # If there are any points in the current stroke
                view_pos = event.position()
                final_logical_pos = self._map_view_point_to_logical_point(view_pos)
                # Add final point if different enough, or if it's a single click (dot)
                if len(self.stroke_points) == 1 and self.stroke_points[0] == final_logical_pos:
                    pass # Single click, point already added on press
                elif not self.stroke_points or (final_logical_pos - self.stroke_points[-1]).manhattanLength() >= 0.1: # Allow closer final point
                    self.stroke_points.append(final_logical_pos)

                self._rebuild_current_path_from_stroke_points(finalize=True) # Finalize path geometry

                if not self.current_path.isEmpty():
                    image_painter = QPainter(self.image)
                    self._draw_path_on_painter(image_painter, self.current_path)
                    image_painter.end()
                    self._save_state_to_undo_stack() # Save drawn state
                    if self.current_glyph_character: self.glyph_modified_signal.emit(self.current_glyph_character, self.image.copy(), self.editing_vrt2_glyph)
            
            self.drawing = False; self.stroke_points = []; self.current_path = QPainterPath(); self.update()

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile()
                ext = Path(file_path).suffix.lower()
                if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    event.acceptProposedAction()
                    return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        if not self.current_glyph_character:
            QMessageBox.warning(self, "グリフ未選択", "グリフが選択されていません。画像をドロップできません。")
            event.ignore(); return

        mime_data = event.mimeData()
        if mime_data.hasUrls():
            file_path = mime_data.urls()[0].toLocalFile()
            try:
                dropped_qimage = QImage(file_path)
                if dropped_qimage.isNull():
                    QMessageBox.warning(self, "画像読み込みエラー", f"画像ファイル '{Path(file_path).name}' を読み込めませんでした。")
                    event.ignore(); return

                # Scale image to fit canvas, preserving aspect ratio, centered on white background
                target_size = self.image_size
                scaled_image = dropped_qimage.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                
                # Create a new QImage of target_size, fill white, then draw scaled_image onto it centered
                final_image = QImage(target_size, QImage.Format_ARGB32_Premultiplied)
                final_image.fill(QColor(Qt.white)) # White background
                
                painter = QPainter(final_image)
                x_offset = (target_size.width() - scaled_image.width()) // 2
                y_offset = (target_size.height() - scaled_image.height()) // 2
                painter.drawImage(x_offset, y_offset, scaled_image)
                painter.end()

                self.image = QPixmap.fromImage(final_image)
                self._save_state_to_undo_stack() # Save dropped image state
                if self.current_glyph_character: self.glyph_modified_signal.emit(self.current_glyph_character, self.image.copy(), self.editing_vrt2_glyph)
                self.update()
                event.acceptProposedAction()
            except Exception as e:
                QMessageBox.critical(self, "ドロップ処理エラー", f"画像の処理中にエラーが発生しました: {e}")
                event.ignore()
        else:
            event.ignore()

# --- Drawing Editor Widget (変更なし) ---
class DrawingEditorWidget(QWidget):
    gui_setting_changed_signal = Signal(str, str) # key, value
    vrt2_edit_mode_toggled = Signal(bool) # is_editing_vrt2_glyph
    transfer_to_vrt2_requested = Signal()
    advance_width_changed_signal = Signal(str, int) # char, new_adv_width
    reference_image_selected_signal = Signal(str, QPixmap, bool) # char, pixmap, is_vrt2
    reference_image_deleted_signal = Signal(str, bool) # char, is_vrt2
    glyph_to_reference_and_reset_requested = Signal(bool) # is_vrt2_target

    def __init__(self, parent=None):
        super().__init__(parent)
        self.canvas = Canvas()
        self.rotated_vrt2_chars: Set[str] = set() # Used for unicode_label display
        main_layout = QVBoxLayout(self)

        self.unicode_label = QLineEdit("Unicode: N/A") # Using QLineEdit for easy copy-paste
        self.unicode_label.setReadOnly(True)
        self.unicode_label.setAlignment(Qt.AlignCenter)
        font = self.unicode_label.font(); font.setPointSize(10); self.unicode_label.setFont(font)
        self.unicode_label.setStyleSheet("QLineEdit { border: none; background: transparent; }") # Make it look like QLabel

        main_layout.addWidget(self.unicode_label)
        main_layout.addWidget(self.canvas, 0, Qt.AlignCenter) # Canvas takes available space, centered

        controls_outer_layout = QVBoxLayout()
        controls_outer_layout.setSpacing(5) # Reduced spacing

        # Top controls: Tools, Undo/Redo, Pen Shape
        top_controls_layout = QHBoxLayout()
        self.pen_button = QPushButton("ペン (B)"); self.pen_button.setCheckable(True)
        self.eraser_button = QPushButton("消しゴム (E)"); self.eraser_button.setCheckable(True)
        self.move_button = QPushButton("移動 (V)"); self.move_button.setCheckable(True)

        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.addButton(self.pen_button); self.tool_button_group.addButton(self.eraser_button)
        self.tool_button_group.addButton(self.move_button)
        self.tool_button_group.setExclusive(True)

        self.pen_button.clicked.connect(self._handle_pen_button_clicked)
        self.eraser_button.clicked.connect(self._handle_eraser_button_clicked)
        self.move_button.clicked.connect(self._handle_move_button_clicked)

        top_controls_layout.addWidget(self.pen_button)
        top_controls_layout.addWidget(self.eraser_button)
        top_controls_layout.addWidget(self.move_button)
        top_controls_layout.addSpacing(10) # Spacer

        self.undo_button = QPushButton("Undo (Ctrl+Z)"); self.undo_button.clicked.connect(self.canvas.undo)
        top_controls_layout.addWidget(self.undo_button)
        self.redo_button = QPushButton("Redo (Ctrl+Y)"); self.redo_button.clicked.connect(self.canvas.redo)
        top_controls_layout.addWidget(self.redo_button)
        top_controls_layout.addSpacing(10)

        top_controls_layout.addWidget(QLabel("先端:"))
        self.shape_box = QComboBox(); self.shape_box.addItems(["丸", "四角"])
        self.shape_box.currentTextChanged.connect(self._handle_pen_shape_changed)
        top_controls_layout.addWidget(self.shape_box)
        top_controls_layout.addStretch(1)
        controls_outer_layout.addLayout(top_controls_layout)

        # Pen Width Slider
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(QLabel("太さ:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 100); self.slider.setValue(self.canvas.pen_width)
        self.slider.valueChanged.connect(self._handle_pen_width_changed)
        slider_layout.addWidget(self.slider, 1) # Slider takes more space
        controls_outer_layout.addLayout(slider_layout)

        # Pen Size Preset Buttons
        self.pen_size_buttons_group = QWidget() # Container for grid
        pen_size_grid_layout = QGridLayout(self.pen_size_buttons_group)
        pen_size_grid_layout.setSpacing(5)
        pen_sizes = [2, 5, 8, 10, 15, 20, 25, 30, 40, 50] # Common pen sizes
        cols = 5
        for i, size_val in enumerate(pen_sizes):
            button = QPushButton(str(size_val)); button.setToolTip(f"{size_val}px")
            button.clicked.connect(lambda checked=False, s=size_val: self._handle_pen_width_changed(s))
            row, col = divmod(i, cols)
            pen_size_grid_layout.addWidget(button, row, col)
        controls_outer_layout.addWidget(self.pen_size_buttons_group)

        # Display Options (Mirror Mode)
        self.mirror_checkbox = QCheckBox("左右反転表示")
        self.mirror_checkbox.toggled.connect(self._handle_mirror_mode_changed)
        display_options_layout = QHBoxLayout()
        display_options_layout.addStretch(1) # Push checkbox to the right
        display_options_layout.addWidget(self.mirror_checkbox)
        controls_outer_layout.addLayout(display_options_layout)
        
        # Glyph Margin Slider
        margin_layout = QHBoxLayout()
        margin_layout.addWidget(QLabel("グリフマージン:"))
        self.margin_slider = QSlider(Qt.Horizontal)
        max_margin_val = min(CANVAS_IMAGE_WIDTH, CANVAS_IMAGE_HEIGHT) // 4 # Max margin is 1/4 of smallest dimension
        self.margin_slider.setRange(0, max_margin_val if max_margin_val > 0 else 1) # Ensure range is valid
        self.margin_slider.setValue(self.canvas.glyph_margin_width)
        self.margin_slider.valueChanged.connect(self._handle_glyph_margin_slider_change)
        self.margin_value_label = QLabel(str(self.canvas.glyph_margin_width)) # Display current margin value
        margin_layout.addWidget(self.margin_slider, 1)
        margin_layout.addWidget(self.margin_value_label)
        controls_outer_layout.addLayout(margin_layout)

        # Reference Image Opacity Slider
        ref_opacity_layout = QHBoxLayout()
        ref_opacity_layout.addWidget(QLabel("下書き透明度:"))
        self.ref_opacity_slider = QSlider(Qt.Horizontal)
        self.ref_opacity_slider.setRange(0, 100) # 0% to 100%
        self.ref_opacity_slider.setValue(int(DEFAULT_REFERENCE_IMAGE_OPACITY * 100))
        self.ref_opacity_slider.valueChanged.connect(self._handle_ref_opacity_changed)
        self.ref_opacity_label = QLabel(str(int(DEFAULT_REFERENCE_IMAGE_OPACITY * 100)))
        ref_opacity_layout.addWidget(self.ref_opacity_slider, 1)
        ref_opacity_layout.addWidget(self.ref_opacity_label)
        controls_outer_layout.addLayout(ref_opacity_layout)

        # Advance Width Controls
        adv_width_layout = QHBoxLayout()
        self.adv_width_label = QLabel("文字送り幅:") # Text changes for VRT2
        adv_width_layout.addWidget(self.adv_width_label)
        self.adv_width_slider = QSlider(Qt.Horizontal)
        self.adv_width_slider.setRange(0, 1000) # Standard font units (0-1000 em)
        self.adv_width_slider.setValue(DEFAULT_ADVANCE_WIDTH)
        self.adv_width_slider.valueChanged.connect(self._on_adv_width_slider_changed)
        self.adv_width_spinbox = QSpinBox()
        self.adv_width_spinbox.setRange(0, 1000)
        self.adv_width_spinbox.setValue(DEFAULT_ADVANCE_WIDTH)
        self.adv_width_spinbox.valueChanged.connect(self._on_adv_width_spinbox_changed)
        adv_width_layout.addWidget(self.adv_width_slider, 1)
        adv_width_layout.addWidget(self.adv_width_spinbox)
        controls_outer_layout.addLayout(adv_width_layout)

        # VRT2 and Reference Image Controls (in a combined row)
        self.vrt2_and_ref_controls_layout = QHBoxLayout()
        self.vrt2_and_ref_controls_layout.setContentsMargins(0, 5, 0, 0) # Top margin

        # VRT2 Controls (Toggle, Transfer) - Grouped in a widget
        self.vrt2_controls_widget = QWidget()
        vrt2_layout = QHBoxLayout(self.vrt2_controls_widget)
        vrt2_layout.setContentsMargins(0,0,0,0) # No internal margins for this group
        self.vrt2_toggle_button = QPushButton("標準グリフ編集中") # Text changes based on mode
        self.vrt2_toggle_button.setCheckable(True)
        self.vrt2_toggle_button.toggled.connect(self._on_vrt2_toggle)
        vrt2_layout.addWidget(self.vrt2_toggle_button)
        self.transfer_to_vrt2_button = QPushButton("標準を縦書きへ転送")
        self.transfer_to_vrt2_button.clicked.connect(self.transfer_to_vrt2_requested)
        vrt2_layout.addWidget(self.transfer_to_vrt2_button)
        
        self.vrt2_and_ref_controls_layout.addWidget(self.vrt2_controls_widget)
        self.vrt2_and_ref_controls_layout.addStretch(1) # Push ref image buttons to the right

        # Reference Image Buttons (Load, Delete, Glyph to Ref) - Grouped
        self.ref_image_buttons_widget = QWidget()
        ref_image_buttons_layout = QHBoxLayout(self.ref_image_buttons_widget)
        ref_image_buttons_layout.setContentsMargins(0,0,0,0)
        ref_image_buttons_layout.setSpacing(5)

        self.load_ref_button = QPushButton("下書き読込")
        self.load_ref_button.clicked.connect(self._handle_load_reference_image_button_clicked)
        ref_image_buttons_layout.addWidget(self.load_ref_button)

        self.delete_ref_button = QPushButton("下書き削除")
        self.delete_ref_button.clicked.connect(self._handle_delete_reference_image_button_clicked)
        ref_image_buttons_layout.addWidget(self.delete_ref_button)

        self.glyph_to_ref_reset_button = QPushButton("グリフを下書きへ転送＆リセット")
        self.glyph_to_ref_reset_button.setToolTip("現在のグリフ画像を下書きにコピーし、グリフ編集エリアを白紙に戻します。")
        self.glyph_to_ref_reset_button.clicked.connect(self._handle_glyph_to_ref_reset_button_clicked)
        ref_image_buttons_layout.addWidget(self.glyph_to_ref_reset_button)
        
        self.vrt2_and_ref_controls_layout.addWidget(self.ref_image_buttons_widget)
        controls_outer_layout.addLayout(self.vrt2_and_ref_controls_layout)
        
        self.vrt2_controls_widget.setVisible(False) # Initially hidden, shown if char is eligible for VRT2

        main_layout.addLayout(controls_outer_layout)

        # Connect canvas signals to update UI elements without re-emitting control signals
        self.canvas.pen_width_changed.connect(self._update_slider_value_no_signal)
        self.canvas.tool_changed.connect(self._update_tool_buttons_state_no_signal)
        self.canvas.undo_redo_state_changed.connect(self._update_undo_redo_buttons_state)
        self.canvas.glyph_margin_width_changed.connect(self._update_glyph_margin_slider_and_label_no_signal)
        self.canvas.glyph_advance_width_changed.connect(self._update_adv_width_ui_no_signal)

        # Initialize UI states from canvas defaults
        self._update_tool_buttons_state_no_signal(self.canvas.current_tool)
        self._update_slider_value_no_signal(self.canvas.pen_width)
        self.shape_box.setCurrentText(DEFAULT_PEN_SHAPE if self.canvas.pen_shape == Qt.RoundCap else "四角")
        self.mirror_checkbox.setChecked(self.canvas.mirror_mode)
        self._update_glyph_margin_slider_and_label_no_signal(self.canvas.glyph_margin_width)
        self._update_adv_width_ui_no_signal(self.canvas.current_glyph_advance_width)
        self._update_undo_redo_buttons_state(False, False) # Initially no undo/redo
        self._update_ref_opacity_slider_no_signal(DEFAULT_REFERENCE_IMAGE_OPACITY)
        self.canvas.set_reference_image_opacity(DEFAULT_REFERENCE_IMAGE_OPACITY)

        self.set_enabled_controls(False) # All controls disabled until a glyph is loaded


    def set_rotated_vrt2_chars(self, chars: Set[str]):
        self.rotated_vrt2_chars = chars
        self.update_unicode_display(self.canvas.current_glyph_character) # Update if char is loaded

    def _handle_glyph_to_ref_reset_button_clicked(self):
        self.glyph_to_reference_and_reset_requested.emit(self.canvas.editing_vrt2_glyph)

    def _handle_load_reference_image_button_clicked(self):
        if not self.canvas.current_glyph_character:
            QMessageBox.warning(self, "グリフ未選択", "下書きを読み込むグリフが選択されていません。")
            return

        file_path, _ = QFileDialog.getOpenFileName(self, "下書き画像を選択", "", "画像ファイル (*.png *.jpg *.jpeg *.bmp)")
        if not file_path: return

        try:
            loaded_qimage = QImage(file_path)
            if loaded_qimage.isNull():
                QMessageBox.warning(self, "画像読み込みエラー", f"画像 '{Path(file_path).name}' を読み込めませんでした。")
                return

            # Prepare image same way as dropEvent
            target_size = self.canvas.image_size
            scaled_image = loaded_qimage.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            final_image = QImage(target_size, QImage.Format_ARGB32_Premultiplied)
            final_image.fill(QColor(Qt.white)) # White background
            
            painter = QPainter(final_image)
            x_offset = (target_size.width() - scaled_image.width()) // 2
            y_offset = (target_size.height() - scaled_image.height()) // 2
            painter.drawImage(x_offset, y_offset, scaled_image)
            painter.end()

            new_reference_pixmap = QPixmap.fromImage(final_image)
            self.canvas.reference_image = new_reference_pixmap # Set on canvas for immediate display
            self.canvas.update() # Repaint canvas
            self.delete_ref_button.setEnabled(self.pen_button.isEnabled()) # Enable delete if controls are on
            # Signal MainWindow to save this new reference image
            self.reference_image_selected_signal.emit(self.canvas.current_glyph_character, new_reference_pixmap.copy(), self.canvas.editing_vrt2_glyph)

        except Exception as e:
            QMessageBox.critical(self, "下書き処理エラー", f"下書き画像の処理中にエラーが発生しました: {e}")

    def _handle_delete_reference_image_button_clicked(self):
        if not self.canvas.current_glyph_character: return # Should not happen if button is enabled
        if self.canvas.reference_image is None: return # Should not happen

        reply = QMessageBox.question(self, "下書き削除の確認",
                                     f"文字 '{self.canvas.current_glyph_character}' の下書き画像を削除しますか？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.canvas.reference_image = None
            self.canvas.update()
            self.delete_ref_button.setEnabled(False) # Disable delete button
            self.reference_image_deleted_signal.emit(self.canvas.current_glyph_character, self.canvas.editing_vrt2_glyph)


    def _handle_ref_opacity_changed(self, value: int): # Slider value 0-100
        opacity_float = value / 100.0
        self.ref_opacity_label.setText(str(value))
        self.canvas.set_reference_image_opacity(opacity_float)
        self.gui_setting_changed_signal.emit(SETTING_REFERENCE_IMAGE_OPACITY, str(opacity_float))

    def _update_ref_opacity_slider_no_signal(self, opacity_float: float):
        slider_value = int(round(opacity_float * 100))
        self.ref_opacity_slider.blockSignals(True)
        self.ref_opacity_slider.setValue(slider_value)
        self.ref_opacity_slider.blockSignals(False)
        self.ref_opacity_label.setText(str(slider_value))

    def _on_adv_width_slider_changed(self, value: int):
        self.adv_width_spinbox.blockSignals(True)
        self.adv_width_spinbox.setValue(value)
        self.adv_width_spinbox.blockSignals(False)
        self.canvas.set_current_glyph_advance_width(value) # Update canvas internal state
        if self.canvas.current_glyph_character: # Only emit if a glyph is active
            self.advance_width_changed_signal.emit(self.canvas.current_glyph_character, value)


    def _on_adv_width_spinbox_changed(self, value: int):
        self.adv_width_slider.blockSignals(True)
        self.adv_width_slider.setValue(value)
        self.adv_width_slider.blockSignals(False)
        self.canvas.set_current_glyph_advance_width(value)
        if self.canvas.current_glyph_character: 
            self.advance_width_changed_signal.emit(self.canvas.current_glyph_character, value)

    def _update_adv_width_ui_no_signal(self, width: int):
        is_vrt2 = self.canvas.editing_vrt2_glyph if self.canvas else False
        self.adv_width_slider.blockSignals(True); self.adv_width_slider.setValue(width); self.adv_width_slider.blockSignals(False)
        self.adv_width_spinbox.blockSignals(True); self.adv_width_spinbox.setValue(width); self.adv_width_spinbox.blockSignals(False)
        self.adv_width_label.setText("文字送り高さ:" if is_vrt2 else "文字送り幅:")
        
        # Enable/disable based on general control state
        adv_controls_enabled = self.pen_button.isEnabled() # Use pen_button as proxy for general enabled state
        self.adv_width_slider.setEnabled(adv_controls_enabled)
        self.adv_width_spinbox.setEnabled(adv_controls_enabled)


    def _on_vrt2_toggle(self, checked: bool):
        if checked: self.vrt2_toggle_button.setText("縦書きグリフ編集中")
        else: self.vrt2_toggle_button.setText("標準グリフ編集中")
        self.vrt2_edit_mode_toggled.emit(checked)

    def update_vrt2_controls(self, show: bool, is_editing_vrt2: bool):
        self.vrt2_controls_widget.setVisible(show)
        if show: # Only update button text if showing
            self.vrt2_toggle_button.blockSignals(True)
            self.vrt2_toggle_button.setChecked(is_editing_vrt2)
            self.vrt2_toggle_button.setText("縦書きグリフ編集中" if is_editing_vrt2 else "標準グリフ編集中")
            self.vrt2_toggle_button.blockSignals(False)


    # --- Handlers for tool changes, emitting GUI settings ---
    def _handle_pen_button_clicked(self): self.canvas.set_brush_mode(); self.gui_setting_changed_signal.emit(SETTING_CURRENT_TOOL, "brush")
    def _handle_eraser_button_clicked(self): self.canvas.set_eraser_mode(); self.gui_setting_changed_signal.emit(SETTING_CURRENT_TOOL, "eraser")
    def _handle_move_button_clicked(self):
        is_move_tool_selected = self.move_button.isChecked() # This button's state *after* click
        self.canvas.set_move_mode(is_move_tool_selected)
        # canvas.current_tool will be updated by set_move_mode or set_brush/eraser_mode
        self.gui_setting_changed_signal.emit(SETTING_CURRENT_TOOL, self.canvas.current_tool)


    def _handle_pen_width_changed(self, width: int): self.canvas.set_pen_width(width); self.gui_setting_changed_signal.emit(SETTING_PEN_WIDTH, str(width))
    def _handle_pen_shape_changed(self, shape_name: str): self.canvas.set_pen_shape(shape_name); self.gui_setting_changed_signal.emit(SETTING_PEN_SHAPE, shape_name)
    def _handle_mirror_mode_changed(self, checked: bool): self.canvas.set_mirror_mode(checked); self.gui_setting_changed_signal.emit(SETTING_MIRROR_MODE, str(checked))
    def _handle_glyph_margin_slider_change(self, value: int): self.canvas.set_glyph_margin_width(value); self.gui_setting_changed_signal.emit(SETTING_GLYPH_MARGIN_WIDTH, str(value))


    # --- UI update methods that DO NOT re-emit signals from canvas ---
    def _update_slider_value_no_signal(self, width: int): self.slider.blockSignals(True); self.slider.setValue(width); self.slider.blockSignals(False)
    def _update_tool_buttons_state_no_signal(self, tool_name: str):
        self.pen_button.blockSignals(True); self.eraser_button.blockSignals(True); self.move_button.blockSignals(True)
        self.pen_button.setChecked(tool_name == "brush")
        self.eraser_button.setChecked(tool_name == "eraser")
        self.move_button.setChecked(tool_name == "move")
        self.pen_button.blockSignals(False); self.eraser_button.blockSignals(False); self.move_button.blockSignals(False)
    def _update_glyph_margin_slider_and_label_no_signal(self, value: int):
        self.margin_slider.blockSignals(True); self.margin_slider.setValue(value); self.margin_slider.blockSignals(False)
        self.margin_value_label.setText(str(value))


    def update_unicode_display(self, character: Optional[str]):
        base_text = ""
        if character:
            if character == '.notdef':
                base_text = "Glyph: .notdef"
            else:
                if isinstance(character, str) and len(character) == 1:
                    try:
                        base_text = f"Unicode: U+{ord(character):04X} ({character})"
                    except TypeError: # Should not happen for single char
                        base_text = f"Char: {character} (Error getting Unicode val)"
                else: # Should ideally be a single character or .notdef
                    base_text = f"Char: {character}"
        else:
            base_text = "Unicode: N/A"

        final_text = base_text
        # Add suffix if editing VRT2 or if standard char is in rotated VRT2 set
        if self.canvas and character: # Ensure canvas exists, character is not None
            if self.canvas.editing_vrt2_glyph: # Currently editing the special VRT2 glyph
                final_text += " vert" # Mark as special vertical
            elif character != '.notdef' and character in self.rotated_vrt2_chars: # Standard char that gets rotated
                final_text += " vert-r" # Mark as auto-rotated vertical
        
        self.unicode_label.setText(final_text)

    def set_enabled_controls(self, enabled: bool):
        self.pen_button.setEnabled(enabled); self.eraser_button.setEnabled(enabled)
        self.move_button.setEnabled(enabled); self.slider.setEnabled(enabled)
        
        if hasattr(self, 'pen_size_buttons_group'): # Check existence for robustness
            for button in self.pen_size_buttons_group.findChildren(QPushButton): button.setEnabled(enabled)
        
        self.shape_box.setEnabled(enabled); self.mirror_checkbox.setEnabled(enabled)
        self.margin_slider.setEnabled(enabled); self.margin_value_label.setEnabled(enabled)
        self.ref_opacity_slider.setEnabled(enabled); self.ref_opacity_label.setEnabled(enabled)
        self.unicode_label.setEnabled(enabled) # Or always enabled if just display

        # Advance width controls update based on current VRT2 state if enabled
        is_vrt2_currently = self.canvas.editing_vrt2_glyph if self.canvas else False
        adv_controls_enabled = enabled # Enable if general controls are enabled
        self.adv_width_slider.setEnabled(adv_controls_enabled)
        self.adv_width_spinbox.setEnabled(adv_controls_enabled)
        self.adv_width_label.setText("文字送り高さ:" if (enabled and is_vrt2_currently) else "文字送り幅:")

        # Reference image buttons
        self.load_ref_button.setEnabled(enabled) # Can always try to load if controls are on
        self.delete_ref_button.setEnabled(enabled and self.canvas.reference_image is not None)
        
        if hasattr(self, 'glyph_to_ref_reset_button'):
             # Enable if controls are on AND there's something on the canvas to transfer
             current_glyph_image = self.canvas.image
             # A more robust check for "empty" (all white) might be too slow here.
             # Just check if it's not null. A truly blank white image might still be transferable.
             current_glyph_has_content = (current_glyph_image and not current_glyph_image.isNull() and
                                          not (current_glyph_image.width() == 1 and current_glyph_image.height() == 1 and 
                                               current_glyph_image.pixelColor(0,0) == QColor(Qt.white).rgba())) # Avoid 1x1 white pixel check
             self.glyph_to_ref_reset_button.setEnabled(enabled and current_glyph_has_content)


        # VRT2 controls are more complex: depend on general enabled state AND char eligibility
        is_vrt2_widget_visible_and_char_eligible = self.vrt2_controls_widget.isVisible() # Already considers char
        self.vrt2_toggle_button.setEnabled(enabled and is_vrt2_widget_visible_and_char_eligible)
        self.transfer_to_vrt2_button.setEnabled(enabled and is_vrt2_widget_visible_and_char_eligible)

        # Undo/Redo state depends on stack AND general enabled state
        if enabled:
            self._update_undo_redo_buttons_state(len(self.canvas.undo_stack) > 1, bool(self.canvas.redo_stack))
        else: # If controls are disabled, undo/redo must be too
            self._update_undo_redo_buttons_state(False, False)
            # Reset some UI elements to default when disabled
            self._update_adv_width_ui_no_signal(DEFAULT_ADVANCE_WIDTH) # Reset display
            self.canvas.reference_image = None # Clear ref image from canvas if controls are off
            self._update_ref_opacity_slider_no_signal(DEFAULT_REFERENCE_IMAGE_OPACITY)
            self.canvas.set_reference_image_opacity(DEFAULT_REFERENCE_IMAGE_OPACITY)

        if not enabled: # If disabling all controls (e.g., no project loaded)
            self.canvas.current_glyph_character = None # Clear current glyph on canvas
            # Optionally, fill canvas with a placeholder color when inactive
            inactive_canvas_fill_color = QColor(220, 220, 220) # Light gray
            self.canvas.image.fill(inactive_canvas_fill_color)
            self.canvas.update()
            self.update_unicode_display(None)


    def _update_undo_redo_buttons_state(self, can_undo: bool, can_redo: bool):
        controls_are_generally_enabled = self.pen_button.isEnabled() # Use pen_button as a proxy
        self.undo_button.setEnabled(can_undo and controls_are_generally_enabled)
        self.redo_button.setEnabled(can_redo and controls_are_generally_enabled)

    def apply_gui_settings(self, settings: Dict[str, Any]):
        pen_width_str = settings.get(SETTING_PEN_WIDTH)
        if pen_width_str is not None:
            try: self.canvas.set_pen_width(int(pen_width_str))
            except ValueError: pass # Ignore if not a valid int
        
        pen_shape = settings.get(SETTING_PEN_SHAPE)
        if pen_shape is not None:
            self.shape_box.blockSignals(True); self.shape_box.setCurrentText(pen_shape); self.shape_box.blockSignals(False)
            self.canvas.set_pen_shape(pen_shape)

        current_tool = settings.get(SETTING_CURRENT_TOOL)
        if current_tool == "eraser": self.canvas.set_eraser_mode()
        elif current_tool == "move": self.canvas.set_move_mode(True) # Enable move mode
        else: self.canvas.set_brush_mode() # Default to brush

        mirror_mode_str = settings.get(SETTING_MIRROR_MODE)
        if mirror_mode_str is not None:
            checked = mirror_mode_str.lower() == 'true'
            self.mirror_checkbox.blockSignals(True); self.mirror_checkbox.setChecked(checked); self.mirror_checkbox.blockSignals(False)
            self.canvas.set_mirror_mode(checked)

        margin_width_str = settings.get(SETTING_GLYPH_MARGIN_WIDTH)
        if margin_width_str is not None:
            try: self.canvas.set_glyph_margin_width(int(margin_width_str))
            except ValueError: pass

        ref_opacity_str = settings.get(SETTING_REFERENCE_IMAGE_OPACITY)
        if ref_opacity_str is not None:
            try:
                ref_opacity_float = float(ref_opacity_str)
                self._update_ref_opacity_slider_no_signal(ref_opacity_float)
                self.canvas.set_reference_image_opacity(ref_opacity_float)
            except ValueError:
                # If invalid, revert to default
                self._update_ref_opacity_slider_no_signal(DEFAULT_REFERENCE_IMAGE_OPACITY)
                self.canvas.set_reference_image_opacity(DEFAULT_REFERENCE_IMAGE_OPACITY)
        else: # If setting not present, use default
            self._update_ref_opacity_slider_no_signal(DEFAULT_REFERENCE_IMAGE_OPACITY)
            self.canvas.set_reference_image_opacity(DEFAULT_REFERENCE_IMAGE_OPACITY)


# --- Glyph Item Widget (変更なし) ---
class GlyphItemWidget(QFrame):
    clicked = Signal(str) # emits character string on click
    def __init__(self, character: str, initial_pixmap: Optional[QPixmap] = None, parent=None):
        super().__init__(parent)
        self.character = character
        self.is_active = False
        self.is_vrt2_highlighted = False # For non-rotated VRT2 chars, tint background

        self.setFixedSize(GLYPH_ITEM_WIDTH, GLYPH_ITEM_HEIGHT)
        self.setFrameStyle(QFrame.Panel | QFrame.Raised)
        self.setLineWidth(1)
        self.setAutoFillBackground(True) # Important for palette changes to take effect
        # Set a base background color slightly darker than default window for better visibility
        base_palette = self.palette()
        widget_bg_color = base_palette.color(QPalette.Window).darker(108) # Adjust darkness as needed
        base_palette.setColor(QPalette.Window, widget_bg_color)
        self.setPalette(base_palette)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2) # Small internal margins
        layout.setSpacing(2)

        self.char_label = QLabel(character)
        self.char_label.setAlignment(Qt.AlignCenter)
        font = self.char_label.font(); font.setPointSize(10); self.char_label.setFont(font)
        self.char_label.setFixedHeight(20) # Fixed height for char label

        self.pixmap_label = QLabel() # Will hold the glyph preview
        self.pixmap_label.setAlignment(Qt.AlignCenter)
        self.pixmap_label.setFrameStyle(QFrame.Panel | QFrame.Sunken) # Sunken effect for preview
        self.pixmap_label.setMinimumSize(DEFAULT_GLYPH_PREVIEW_SIZE) # Ensure it's at least this big
        self.pixmap_label.setAutoFillBackground(True) # For background tinting
        self._original_pixmap_label_palette = self.pixmap_label.palette() # Store for reverting tint

        layout.addWidget(self.char_label)
        layout.addWidget(self.pixmap_label, 1) # Pixmap label takes remaining space

        self.update_preview(initial_pixmap)

    def set_active(self, active: bool):
        if self.is_active != active:
            self.is_active = active
            if active:
                self.setFrameShadow(QFrame.Sunken)
                self.setLineWidth(2) # Thicker border when active
                # Optionally change background or border color more dramatically
            else:
                self.setFrameShadow(QFrame.Raised)
                self.setLineWidth(1)
            self.style().unpolish(self); self.style().polish(self) # Force style refresh
            self.update()

    def set_vrt2_highlight(self, highlight: bool):
        if self.is_vrt2_highlighted != highlight:
            self.is_vrt2_highlighted = highlight
            pal = self.pixmap_label.palette()
            if highlight:
                pal.setColor(QPalette.Window, VRT2_PREVIEW_BACKGROUND_TINT)
            else: # Revert to original background
                original_bg = self._original_pixmap_label_palette.color(QPalette.Window)
                pal.setColor(QPalette.Window, original_bg)
            self.pixmap_label.setPalette(pal)
            # If preview is empty, re-render it to show new background or char on new bg
            current_pix = self.pixmap_label.pixmap()
            is_empty_preview = not current_pix or current_pix.isNull() or \
                               (current_pix.size() == DEFAULT_GLYPH_PREVIEW_SIZE and \
                                all(current_pix.toImage().pixelColor(x,y) == Qt.transparent for x in range(min(1, current_pix.width())) for y in range(min(1, current_pix.height())) )) # Basic check for transparent/empty
            if is_empty_preview: self.update_preview(None)


    def update_preview(self, pixmap: Optional[QPixmap]):
        if pixmap and not pixmap.isNull():
            # Scale the pixmap to fit the label while maintaining aspect ratio
            scaled_pixmap = pixmap.scaled(DEFAULT_GLYPH_PREVIEW_SIZE, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.pixmap_label.setPixmap(scaled_pixmap)
        else:
            # Create an empty preview placeholder (e.g., with the character text)
            empty_preview = QPixmap(DEFAULT_GLYPH_PREVIEW_SIZE)
            current_pixmap_bg_color = self.pixmap_label.palette().color(QPalette.Window)
            empty_preview.fill(current_pixmap_bg_color) # Fill with current background of label

            # Optionally, draw the character text on the empty preview
            painter = QPainter(empty_preview)
            pen = painter.pen()
            # Choose text color based on background lightness for visibility
            text_color = Qt.darkGray if current_pixmap_bg_color.lightnessF() > 0.5 else Qt.lightGray
            pen.setColor(text_color)
            painter.setPen(pen)
            
            char_display_text = self.character
            font = painter.font()
            # Adjust font size for display in small preview
            font.setPointSize(8 if self.character == '.notdef' else 10) # Smaller for .notdef
            painter.setFont(font)
            
            painter.drawText(empty_preview.rect(), Qt.AlignCenter, char_display_text)
            painter.end()
            self.pixmap_label.setPixmap(empty_preview)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.character)
        super().mousePressEvent(event)





















# --- Glyph Grid Widget (Modified for incremental redisplay) ---
class GlyphGridWidget(QWidget):
    glyph_selected_signal = Signal(str) 
    vrt2_glyph_selected_signal = Signal(str) 

    def __init__(self, parent=None):
        super().__init__(parent)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0) 
        main_layout.setSpacing(5) # QVBoxLayout自体のスペーシング

        top_controls_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("文字を検索...")
        self.search_input.returnPressed.connect(self._perform_search) 
        top_controls_layout.addWidget(self.search_input, 1) 
        self.search_button = QPushButton("検索")
        self.search_button.clicked.connect(self._perform_search)
        top_controls_layout.addWidget(self.search_button)
        main_layout.addLayout(top_controls_layout)

        self.show_written_only_checkbox = QCheckBox("書き込み済みグリフのみ表示")
        self.show_written_only_checkbox.toggled.connect(self._on_filter_changed)
        main_layout.addWidget(self.show_written_only_checkbox)


        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True) # これが重要
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff) 

        self.grid_container = QWidget() 
        self.grid_layout = QGridLayout(self.grid_container)
        
        fixed_spacing = 5 
        self.grid_layout.setSpacing(fixed_spacing) 
        self.grid_layout.setContentsMargins(fixed_spacing, fixed_spacing, fixed_spacing, fixed_spacing) 
        
        self.scroll_area.setWidget(self.grid_container)
        main_layout.addWidget(self.scroll_area, 1) 

        self.glyph_widgets: Dict[str, GlyphItemWidget] = {} 
        self.active_char_widget: Optional[GlyphItemWidget] = None

        self.special_vrt2_glyph_widgets: Dict[str, GlyphItemWidget] = {}
        self.active_special_vrt2_widget: Optional[GlyphItemWidget] = None
        
        self.non_rotated_vrt2_chars: set[str] = set() 

        self._all_glyph_data_for_filter: List[Tuple[str, Optional[QPixmap]]] = []
        self._special_vrt2_data_for_filter: List[Tuple[str, Optional[QPixmap]]] = []
        
        self._current_standard_glyph_count = 0
        self._current_vrt2_glyph_count = 0
        self._has_vrt2_section_separator = False
        self._vrt2_separator_widget: Optional[QLabel] = None

        # --- Batch processing and redisplay state ---
        self._master_batch_queue: List[Tuple[List[Tuple[str, Optional[QPixmap]]], bool, bool]] = []
        self._is_processing_master_batch: bool = False
        
        self._current_add_batch_data: List[Tuple[str, Optional[QPixmap]]] = []
        self._current_add_batch_index: int = 0
        self._current_add_batch_type_is_vrt2: bool = False
        self._current_add_batch_is_final_for_type: bool = False
        
        self._pending_standard_redisplay: List[Tuple[str, Optional[QPixmap]]] = []
        self._pending_vrt2_redisplay: List[Tuple[str, Optional[QPixmap]]] = []
        self._redisplay_active_char_cache: Optional[str] = None
        self._redisplay_active_vrt2_source_cache: bool = False
        
        self._incremental_redisplay_timer: Optional[QTimer] = None
        self._single_widget_add_timer: Optional[QTimer] = None





















    def set_search_enabled(self, enabled: bool):
        self.search_input.setEnabled(enabled)
        self.search_button.setEnabled(enabled)
        self.show_written_only_checkbox.setEnabled(enabled)

    def _perform_search(self):
        search_text = self.search_input.text()
        if not search_text: return 

        char_to_find = search_text[0] 
        if char_to_find == '.' and len(search_text) > 1: 
            if search_text.lower() == ".notdef": char_to_find = ".notdef"
        
        if char_to_find in self.glyph_widgets:
            self.glyph_selected_signal.emit(char_to_find)
            self.search_input.clear() 
            return
        elif char_to_find in self.special_vrt2_glyph_widgets:
            self.vrt2_glyph_selected_signal.emit(char_to_find)
            self.search_input.clear()
            return
        
        if any(char_to_find == data[0] for data in self._all_glyph_data_for_filter): 
            self.glyph_selected_signal.emit(char_to_find) 
            self.search_input.clear()
        elif any(char_to_find == data[0] for data in self._special_vrt2_data_for_filter): 
            self.vrt2_glyph_selected_signal.emit(char_to_find)
            self.search_input.clear()
        else:
            QMessageBox.information(self, "検索結果", f"文字 '{search_text}' はプロジェクトに見つかりません。")


    def _on_filter_changed(self):
        self.redisplay_from_cached_data()


    def set_non_rotated_vrt2_chars(self, chars: set[str]):
        if self.non_rotated_vrt2_chars != chars:
            self.non_rotated_vrt2_chars = chars
            for char, widget in self.glyph_widgets.items(): 
                widget.set_vrt2_highlight(char in self.non_rotated_vrt2_chars)
            
            if (self.glyph_widgets or self.special_vrt2_glyph_widgets): 
                self.redisplay_from_cached_data()


    def clear_grid(self):
        """Clears all glyph items from the grid and resets counts."""
        if self._incremental_redisplay_timer and self._incremental_redisplay_timer.isActive():
            self._incremental_redisplay_timer.stop()
        if self._single_widget_add_timer and self._single_widget_add_timer.isActive():
            self._single_widget_add_timer.stop()

        # Clear all queues and processing flags
        self._master_batch_queue.clear()
        self._is_processing_master_batch = False
        self._current_add_batch_data.clear()
        self._current_add_batch_index = 0
        self._pending_standard_redisplay.clear()
        self._pending_vrt2_redisplay.clear()

        while self.grid_layout.count():
            item = self.grid_layout.takeAt(0)
            widget = item.widget()
            if widget: widget.deleteLater()
        
        self.glyph_widgets.clear()
        self.special_vrt2_glyph_widgets.clear()
        self.active_char_widget = None
        self.active_special_vrt2_widget = None
        
        self._current_standard_glyph_count = 0
        self._current_vrt2_glyph_count = 0
        self._has_vrt2_section_separator = False
        if self._vrt2_separator_widget: 
            self._vrt2_separator_widget.deleteLater()
            self._vrt2_separator_widget = None 

        self.grid_container.setMinimumHeight(0) 
        self.grid_container.adjustSize() 
        self.scroll_area.updateGeometry()


    def set_filter_caches(self, standard_glyphs: List[Tuple[str, Optional[QPixmap]]], 
                           special_vrt2_glyphs: List[Tuple[str, Optional[QPixmap]]]):
        self._all_glyph_data_for_filter = standard_glyphs
        self._special_vrt2_data_for_filter = special_vrt2_glyphs

    def redisplay_from_cached_data(self):
        """Clears and re-populates the grid incrementally using the cached filter data."""
        if self._is_processing_master_batch or self._master_batch_queue:
             # If already processing, wait. Or, decide to cancel and restart.
             # For simplicity, let's just queue this request by clearing and re-adding.
             # Or better: stop current processing.
            if self._single_widget_add_timer and self._single_widget_add_timer.isActive():
                self._single_widget_add_timer.stop()
            if self._incremental_redisplay_timer and self._incremental_redisplay_timer.isActive():
                self._incremental_redisplay_timer.stop()
            self._master_batch_queue.clear()
            self._is_processing_master_batch = False


        self._redisplay_active_char_cache = None
        self._redisplay_active_vrt2_source_cache = False
        if self.active_char_widget:
            self._redisplay_active_char_cache = self.active_char_widget.character
        elif self.active_special_vrt2_widget:
            self._redisplay_active_char_cache = self.active_special_vrt2_widget.character
            self._redisplay_active_vrt2_source_cache = True

        self.clear_grid() # This also clears master_batch_queue

        self._pending_standard_redisplay = [gd for gd in self._all_glyph_data_for_filter 
                               if not self.show_written_only_checkbox.isChecked() or gd[1] is not None]
        self._pending_vrt2_redisplay = [gd for gd in self._special_vrt2_data_for_filter
                                   if not self.show_written_only_checkbox.isChecked() or gd[1] is not None]
        
        self._current_standard_glyph_count = 0
        self._current_vrt2_glyph_count = 0
        self._has_vrt2_section_separator = False 

        self._schedule_incremental_add()


    def _schedule_incremental_add(self):
        # This is for redisplay_from_cached_data path
        if not self._incremental_redisplay_timer:
            self._incremental_redisplay_timer = QTimer(self)
            self._incremental_redisplay_timer.setSingleShot(True)
            self._incremental_redisplay_timer.timeout.connect(self._process_incremental_add_chunk) # Renamed
        
        if not self._incremental_redisplay_timer.isActive() and not self._is_processing_master_batch:
             self._incremental_redisplay_timer.start(0) 


    def _process_incremental_add_chunk(self): # Renamed from _process_incremental_add
        # This method is for the redisplay_from_cached_data path
        # It takes one chunk from _pending_ lists and adds it to _master_batch_queue
        if self._pending_standard_redisplay:
            chunk_to_add = self._pending_standard_redisplay[:GLYPH_UI_UPDATE_BATCH_SIZE]
            self._pending_standard_redisplay = self._pending_standard_redisplay[GLYPH_UI_UPDATE_BATCH_SIZE:]
            is_final_overall = not self._pending_standard_redisplay and not self._pending_vrt2_redisplay
            self.add_glyph_batch(chunk_to_add, is_vrt2_batch=False, is_final_for_type=is_final_overall)
            return 

        if self._pending_vrt2_redisplay:
            chunk_to_add = self._pending_vrt2_redisplay[:GLYPH_UI_UPDATE_BATCH_SIZE]
            self._pending_vrt2_redisplay = self._pending_vrt2_redisplay[GLYPH_UI_UPDATE_BATCH_SIZE:]
            is_final_overall = not self._pending_vrt2_redisplay
            self.add_glyph_batch(chunk_to_add, is_vrt2_batch=True, is_final_for_type=is_final_overall)
            return
        
        # If both pending lists are empty, and no batches are being processed,
        # this implies the redisplay operation should finalize.
        # This is now handled by _start_processing_next_master_batch when _master_batch_queue is empty.

    def _finalize_incremental_add(self):
        # This is called when incremental redisplay (from filter change) is complete.
        self.grid_container.adjustSize() 
        current_content_height = self.grid_container.sizeHint().height()
        viewport_height = self.scroll_area.viewport().height()
        if current_content_height < viewport_height:
            self.grid_container.setMinimumHeight(viewport_height)
        else:
            self.grid_container.setMinimumHeight(0) 
        self.scroll_area.updateGeometry() 
        
        if self._redisplay_active_char_cache:
            self.set_active_glyph(self._redisplay_active_char_cache, self._redisplay_active_vrt2_source_cache)
        
        self._pending_standard_redisplay.clear()
        self._pending_vrt2_redisplay.clear()
        self._redisplay_active_char_cache = None
        self._redisplay_active_vrt2_source_cache = False


    def add_glyph_batch(self, glyph_data_batch: List[Tuple[str, Optional[QPixmap]]], 
                        is_vrt2_batch: bool, is_final_for_type: bool):
        if not glyph_data_batch:
            # If it's an empty batch but also the final one for this type (e.g., all VRT2 glyphs were filtered out)
            if is_final_for_type and is_vrt2_batch:
                if self._has_vrt2_section_separator and self._current_vrt2_glyph_count == 0:
                    if self._vrt2_separator_widget:
                        self.grid_layout.removeWidget(self._vrt2_separator_widget)
                        self._vrt2_separator_widget.deleteLater(); self._vrt2_separator_widget = None
                    self._has_vrt2_section_separator = False
                # If this empty final batch completes all known processing (no pending redisplay chunks either)
                if not self._is_processing_master_batch and not self._master_batch_queue and \
                   not self._pending_standard_redisplay and not self._pending_vrt2_redisplay:
                     self.grid_container.adjustSize(); self.scroll_area.updateGeometry()
            return

        self._master_batch_queue.append(
            (list(glyph_data_batch), is_vrt2_batch, is_final_for_type)
        )
        
        if not self._is_processing_master_batch:
            self._start_processing_next_master_batch()

    def _start_processing_next_master_batch(self):
        if not self._master_batch_queue:
            self._is_processing_master_batch = False
            # Check if we were in a redisplay flow (initiated by _on_filter_changed)
            if self._pending_standard_redisplay or self._pending_vrt2_redisplay:
                # There are still chunks in the _pending_ lists for redisplay, schedule the next one
                self._schedule_incremental_add()
            elif self._redisplay_active_char_cache:
                # All _pending_ lists are empty, and master queue is empty. Redisplay is complete.
                self._finalize_incremental_add()
            # Else: all batches from an initial load (not filter-redisplay) are done.
            # MainWindow._check_and_finalize_loading_state handles the overall finalization.
            return

        self._is_processing_master_batch = True
        
        current_batch_info = self._master_batch_queue.pop(0)
        self._current_add_batch_data, self._current_add_batch_type_is_vrt2, self._current_add_batch_is_final_for_type = current_batch_info
        self._current_add_batch_index = 0
        
        is_vrt2_current_batch = self._current_add_batch_type_is_vrt2
        
        # Handle VRT2 separator logic (only if this batch actually contains VRT2 items)
        if is_vrt2_current_batch and self._current_add_batch_data and not self._has_vrt2_section_separator:
            current_row_for_separator = (self._current_standard_glyph_count + GRID_COLUMNS -1) // GRID_COLUMNS
            if self._current_standard_glyph_count == 0 and self._vrt2_separator_widget is None:
                current_row_for_separator = 0

            if self._vrt2_separator_widget: 
                self.grid_layout.removeWidget(self._vrt2_separator_widget)
                self._vrt2_separator_widget.deleteLater(); self._vrt2_separator_widget = None
            
            separator_label = QLabel("--- 非回転縦書きグリフ ---") 
            separator_label.setAlignment(Qt.AlignCenter)
            self._vrt2_separator_widget = separator_label 
            self.grid_layout.addWidget(self._vrt2_separator_widget, current_row_for_separator, 0, 1, GRID_COLUMNS)
            self._has_vrt2_section_separator = True
            QApplication.processEvents() 

        self._schedule_single_widget_add()

    def _schedule_single_widget_add(self):
        if not self._single_widget_add_timer:
            self._single_widget_add_timer = QTimer(self)
            self._single_widget_add_timer.setSingleShot(True)
            self._single_widget_add_timer.timeout.connect(self._process_single_widget_add)
        
        if not self._single_widget_add_timer.isActive():
            self._single_widget_add_timer.start(0)

    def _process_single_widget_add(self):
        if not self._current_add_batch_data or self._current_add_batch_index >= len(self._current_add_batch_data):
            # Current internal batch (_current_add_batch_data) is finished
            is_vrt2_batch_that_finished = self._current_add_batch_type_is_vrt2
            is_final_for_type_of_finished_batch = self._current_add_batch_is_final_for_type
            
            self._current_add_batch_data.clear() # Clear the processed chunk
            self._current_add_batch_index = 0

            if is_final_for_type_of_finished_batch:
                # This was the last widget of the last chunk for this type (standard or VRT2)
                if is_vrt2_batch_that_finished and self._has_vrt2_section_separator and self._current_vrt2_glyph_count == 0:
                    if self._vrt2_separator_widget:
                        self.grid_layout.removeWidget(self._vrt2_separator_widget)
                        self._vrt2_separator_widget.deleteLater(); self._vrt2_separator_widget = None
                    self._has_vrt2_section_separator = False
                
                is_part_of_pending_redisplay = bool(self._pending_standard_redisplay or self._pending_vrt2_redisplay or self._master_batch_queue)
                if not is_part_of_pending_redisplay : # From initial load and this type is done
                    self.grid_container.adjustSize() 
                    self.scroll_area.updateGeometry()

            # Try to process the next master batch from the queue
            self._start_processing_next_master_batch()
            return

        # --- Process one widget from _current_add_batch_data ---
        char, pixmap = self._current_add_batch_data[self._current_add_batch_index]
        is_vrt2_batch_for_this_widget = self._current_add_batch_type_is_vrt2
        
        if is_vrt2_batch_for_this_widget:
            base_row_offset = (self._current_standard_glyph_count + GRID_COLUMNS -1) // GRID_COLUMNS
            if self._current_standard_glyph_count == 0: base_row_offset = 0
            if self._has_vrt2_section_separator: base_row_offset +=1
            current_item_index_in_type = self._current_vrt2_glyph_count 
        else:
            base_row_offset = 0
            current_item_index_in_type = self._current_standard_glyph_count

        item_widget = GlyphItemWidget(char, pixmap)
        if is_vrt2_batch_for_this_widget:
            item_widget.clicked.connect(self._handle_special_vrt2_glyph_click)
            item_widget.set_vrt2_highlight(False) 
            self.special_vrt2_glyph_widgets[char] = item_widget
        else:
            item_widget.clicked.connect(self.glyph_selected_signal)
            item_widget.set_vrt2_highlight(char in self.non_rotated_vrt2_chars)
            self.glyph_widgets[char] = item_widget
        
        row, col = divmod(current_item_index_in_type, GRID_COLUMNS)
        self.grid_layout.addWidget(item_widget, base_row_offset + row, col, Qt.AlignTop)

        if is_vrt2_batch_for_this_widget:
            self._current_vrt2_glyph_count += 1
        else:
            self._current_standard_glyph_count += 1
        
        self._current_add_batch_index += 1
        
        self._schedule_single_widget_add() # Schedule next widget in current internal batch


    def _handle_special_vrt2_glyph_click(self, character: str):
        self.vrt2_glyph_selected_signal.emit(character)


    def set_active_glyph(self, character: Optional[str], is_vrt2_source: bool = False):
        if self.active_char_widget:
            self.active_char_widget.set_active(False)
            self.active_char_widget = None
        if self.active_special_vrt2_widget:
            self.active_special_vrt2_widget.set_active(False)
            self.active_special_vrt2_widget = None
        
        widget_to_activate: Optional[GlyphItemWidget] = None
        if character:
            if is_vrt2_source: 
                if character in self.special_vrt2_glyph_widgets:
                    widget_to_activate = self.special_vrt2_glyph_widgets[character]
                    self.active_special_vrt2_widget = widget_to_activate
                elif character in self.glyph_widgets: 
                    widget_to_activate = self.glyph_widgets.get(character) 
                    if widget_to_activate: self.active_char_widget = widget_to_activate 
            else: 
                if character in self.glyph_widgets:
                    widget_to_activate = self.glyph_widgets[character]
                    self.active_char_widget = widget_to_activate
                elif character in self.special_vrt2_glyph_widgets: 
                    widget_to_activate = self.special_vrt2_glyph_widgets.get(character) 
                    if widget_to_activate: self.active_special_vrt2_widget = widget_to_activate


            if widget_to_activate:
                widget_to_activate.set_active(True)
                # Updated check for whether items are still being added
                is_adding_incrementally = (self._is_processing_master_batch or \
                                          bool(self._master_batch_queue) or \
                                          bool(self._pending_standard_redisplay) or \
                                          bool(self._pending_vrt2_redisplay))

                if not is_adding_incrementally:
                    scroll_call = functools.partial(self.scroll_area.ensureWidgetVisible, widget_to_activate, 50, 50) 
                    QTimer.singleShot(100, scroll_call) 


















    
    def update_glyph_preview(self, character: str, pixmap: QPixmap):
        updated_in_filter_cache = False
        old_pixmap_in_cache = None
        for i, (char_cache, current_pix_in_cache) in enumerate(self._all_glyph_data_for_filter):
            if char_cache == character:
                old_pixmap_in_cache = current_pix_in_cache
                self._all_glyph_data_for_filter[i] = (character, pixmap)
                updated_in_filter_cache = True
                break
        
        if self.show_written_only_checkbox.isChecked():
            was_written = old_pixmap_in_cache is not None
            is_written = pixmap is not None
            if was_written != is_written: 
                self.redisplay_from_cached_data()
                return 
        
        if character in self.glyph_widgets:
            self.glyph_widgets[character].update_preview(pixmap)
        elif updated_in_filter_cache and pixmap is not None and self.show_written_only_checkbox.isChecked():
            self.redisplay_from_cached_data()


    def update_single_special_vrt2_preview(self, character: str, pixmap: QPixmap):
        updated_in_filter_cache = False
        old_pixmap_in_cache = None
        for i, (char_cache, current_pix_in_cache) in enumerate(self._special_vrt2_data_for_filter):
            if char_cache == character:
                old_pixmap_in_cache = current_pix_in_cache
                self._special_vrt2_data_for_filter[i] = (character, pixmap)
                updated_in_filter_cache = True
                break
        
        if self.show_written_only_checkbox.isChecked():
            was_written = old_pixmap_in_cache is not None
            is_written = pixmap is not None
            if was_written != is_written:
                self.redisplay_from_cached_data()
                return

        if character in self.special_vrt2_glyph_widgets:
            self.special_vrt2_glyph_widgets[character].update_preview(pixmap)
        elif updated_in_filter_cache and pixmap is not None and self.show_written_only_checkbox.isChecked():
             self.redisplay_from_cached_data()

    def get_first_glyph_char(self) -> Optional[str]:
        source_list = self._all_glyph_data_for_filter
        if self.show_written_only_checkbox.isChecked():
            filtered_list = [char for char, pix in source_list if pix is not None]
            return filtered_list[0] if filtered_list else None
        else: 
            return source_list[0][0] if source_list else None
    
    def get_first_overall_glyph_char(self) -> Optional[str]:
        if self._all_glyph_data_for_filter:
            return self._all_glyph_data_for_filter[0][0]
        return None


    def get_navigable_glyphs_info(self) -> List[Tuple[str, bool]]:
        nav_list: List[Tuple[str, bool]] = []
        
        standard_data_to_use = [gd for gd in self._all_glyph_data_for_filter
                                if not self.show_written_only_checkbox.isChecked() or gd[1] is not None]
        for char, _ in standard_data_to_use:
            nav_list.append((char, False)) 

        special_vrt2_data_to_use = [gd for gd in self._special_vrt2_data_for_filter
                                    if not self.show_written_only_checkbox.isChecked() or gd[1] is not None]
        for char, _ in special_vrt2_data_to_use:
            nav_list.append((char, True)) 
            
        return nav_list




# --- Properties Widget (変更なし) ---
class PropertiesWidget(QWidget):
    character_set_changed_signal = Signal(str)
    rotated_vrt2_set_changed_signal = Signal(str)
    non_rotated_vrt2_set_changed_signal = Signal(str)
    font_name_changed_signal = Signal(str)
    font_weight_changed_signal = Signal(str)
    export_font_signal = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        font_name_layout = QHBoxLayout()
        font_name_layout.addWidget(QLabel("フォント名:"))
        self.font_name_input = QLineEdit()
        self.font_name_input.setPlaceholderText("例: MyCustomFont")
        self.font_name_input.editingFinished.connect(self._emit_font_name_change) # Emit on focus out or enter
        font_name_layout.addWidget(self.font_name_input)
        layout.addLayout(font_name_layout)

        font_weight_layout = QHBoxLayout()
        font_weight_layout.addWidget(QLabel("ウェイト:"))
        self.font_weight_combobox = QComboBox()
        self.font_weight_combobox.addItems(FONT_WEIGHT_OPTIONS)
        self.font_weight_combobox.setCurrentText(DEFAULT_FONT_WEIGHT) # Set default
        self.font_weight_combobox.currentTextChanged.connect(self._emit_font_weight_change)
        font_weight_layout.addWidget(self.font_weight_combobox)
        font_weight_layout.addStretch(1) # Allow combobox to not take full width if short
        layout.addLayout(font_weight_layout)
        
        layout.addSpacing(10)

        layout.addWidget(QLabel("プロジェクトの文字セット:"))
        self.char_set_text_edit = QTextEdit()
        self.char_set_text_edit.setPlaceholderText("例: あいうえお漢字...")
        self.char_set_text_edit.setFixedHeight(200) # Reasonable default height
        layout.addWidget(self.char_set_text_edit)
        apply_char_set_button = QPushButton("文字セットを適用")
        apply_char_set_button.clicked.connect(self._apply_char_set_changes)
        layout.addWidget(apply_char_set_button)

        layout.addSpacing(10)

        layout.addWidget(QLabel("回転縦書きグリフ (単純回転):"))
        self.r_vrt2_text_edit = QTextEdit()
        self.r_vrt2_text_edit.setPlaceholderText("例: （）「」…")
        self.r_vrt2_text_edit.setFixedHeight(60)
        layout.addWidget(self.r_vrt2_text_edit)
        apply_r_vrt2_button = QPushButton("回転縦書き文字セットを適用")
        apply_r_vrt2_button.clicked.connect(self._apply_r_vrt2_changes)
        layout.addWidget(apply_r_vrt2_button)

        layout.addSpacing(10)

        layout.addWidget(QLabel("非回転縦書き文字 (専用グリフ):"))
        self.nr_vrt2_text_edit = QTextEdit()
        self.nr_vrt2_text_edit.setPlaceholderText("例: 、。‘’“”…")
        self.nr_vrt2_text_edit.setFixedHeight(60)
        layout.addWidget(self.nr_vrt2_text_edit)
        apply_nr_vrt2_button = QPushButton("非回転縦書き文字セットを適用")
        apply_nr_vrt2_button.clicked.connect(self._apply_nr_vrt2_changes)
        layout.addWidget(apply_nr_vrt2_button)

        layout.addSpacing(20) # More space before export button

        self.export_font_button = QPushButton("フォントを書き出す")
        self.export_font_button.clicked.connect(self.export_font_signal)
        layout.addWidget(self.export_font_button)

        layout.addStretch(1) # Push all content to the top

    def _emit_font_name_change(self): self.font_name_changed_signal.emit(self.font_name_input.text())
    def _emit_font_weight_change(self, weight_text: str): self.font_weight_changed_signal.emit(weight_text)
    def _apply_char_set_changes(self): self.character_set_changed_signal.emit(self.char_set_text_edit.toPlainText())
    def _apply_r_vrt2_changes(self): self.rotated_vrt2_set_changed_signal.emit(self.r_vrt2_text_edit.toPlainText())
    def _apply_nr_vrt2_changes(self): self.non_rotated_vrt2_set_changed_signal.emit(self.nr_vrt2_text_edit.toPlainText())

    def load_font_name(self, name: str):
        self.font_name_input.blockSignals(True) # Avoid emitting signal during load
        self.font_name_input.setText(name)
        self.font_name_input.blockSignals(False)

    def load_font_weight(self, weight: str):
        self.font_weight_combobox.blockSignals(True)
        if weight in FONT_WEIGHT_OPTIONS: self.font_weight_combobox.setCurrentText(weight)
        else: self.font_weight_combobox.setCurrentText(DEFAULT_FONT_WEIGHT) # Fallback
        self.font_weight_combobox.blockSignals(False)

    def load_character_set(self, char_string: str): self.char_set_text_edit.setText(char_string)
    def load_r_vrt2_set(self, char_string: str): self.r_vrt2_text_edit.setText(char_string)
    def load_nr_vrt2_set(self, char_string: str): self.nr_vrt2_text_edit.setText(char_string)

    def set_enabled_controls(self, enabled: bool):
        self.font_name_input.setEnabled(enabled)
        self.font_weight_combobox.setEnabled(enabled)
        self.char_set_text_edit.setEnabled(enabled)
        self.r_vrt2_text_edit.setEnabled(enabled)
        self.nr_vrt2_text_edit.setEnabled(enabled)
        # Enable/disable "Apply" buttons
        apply_buttons = [btn for btn in self.findChildren(QPushButton) if btn != self.export_font_button]
        for btn in apply_buttons: btn.setEnabled(enabled)
        self.export_font_button.setEnabled(enabled)


# --- Batch Advance Width Dialog (変更なし) ---
class BatchAdvanceWidthDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("文字送り幅一括編集")
        self.setMinimumWidth(350) # Ensure dialog is not too small
        layout = QVBoxLayout(self)

        form_layout = QGridLayout()
        form_layout.addWidget(QLabel("適用する文字 (例: あいう / U+3042-U+304A / .notdef):"), 0, 0)
        self.char_spec_input = QLineEdit()
        self.char_spec_input.setPlaceholderText("あいうえお or U+XXXX-U+YYYY or .notdef")
        form_layout.addWidget(self.char_spec_input, 0, 1)

        form_layout.addWidget(QLabel("新しい文字送り幅 (0-1000):"), 1, 0)
        self.advance_width_spinbox = QSpinBox()
        self.advance_width_spinbox.setRange(0, 1000) # Range for advance width
        self.advance_width_spinbox.setValue(DEFAULT_ADVANCE_WIDTH) # Default value
        form_layout.addWidget(self.advance_width_spinbox, 1, 1)
        
        layout.addLayout(form_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Ok).setText("適用")
        self.button_box.button(QDialogButtonBox.Cancel).setText("キャンセル")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def get_values(self) -> Tuple[Optional[str], Optional[int]]:
        char_spec = self.char_spec_input.text().strip()
        adv_width = self.advance_width_spinbox.value()
        return char_spec if char_spec else None, adv_width


# --- MainWindow (Modified for batch loading) ---
class MainWindow(QMainWindow):
    KV_MODE_FONT_DISPLAY = 0
    KV_MODE_WRITTEN_GLYPHS = 1
    KV_MODE_HIDDEN = 2

    def __init__(self):
        super().__init__()
        self.setWindowTitle("P-Glyph")
        self.setGeometry(50, 50, 1550, 800)
        self.setFocusPolicy(Qt.StrongFocus)

        self.db_manager = DatabaseManager() 
        self.current_project_path: Optional[str] = None
        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(QThread.idealThreadCount())

        self.non_rotated_vrt2_chars: set[str] = set()
        self.project_glyph_chars_cache: Set[str] = set() # Cache of characters in the project (standard glyphs)
        # Caches for glyph data (char, pixmap_or_None) for filtering and KV display
        self._all_standard_glyphs_pixmap_cache: List[Tuple[str, Optional[QPixmap]]] = []
        self._all_nrvg_pixmap_cache: List[Tuple[str, Optional[QPixmap]]] = []


        self.export_process: Optional[QProcess] = None
        self.original_export_button_state: bool = False # To restore after export
        
        self._project_loading_in_progress: bool = False
        self._last_active_glyph_char_from_load: Optional[str] = None # To restore after project load


        self.drawing_editor_widget = DrawingEditorWidget()
        self.glyph_grid_widget = GlyphGridWidget()
        self.glyph_grid_widget.setFixedWidth(GLYPH_GRID_WIDTH)
        self.properties_widget = PropertiesWidget()
        self.properties_widget.setFixedWidth(PROPERTIES_WIDTH)

        # Kanji Viewer UI Elements (initialized in _init_ui_for_kanji_viewer_panel)
        self.kanji_viewer_panel_widget: Optional[QWidget] = None
        self.kanji_viewer_font_combo: Optional[QComboBox] = None
        self.bookmark_font_button: Optional[QPushButton] = None
        self.kanji_viewer_display_label: Optional[QLabel] = None
        self.kanji_viewer_related_tabs: Optional[VerticalTabWidget] = None

        # Kanji Viewer Data
        self.kanji_radicals_data = None
        self.radical_to_kanji_data = None
        self.KANJI_TO_DATA_FILENAME = "kanji2element.json"
        self.DATA_TO_KANJI_FILENAME = "element2kanji.json"
        self.related_kanji_label_style = "QLabel { border: 1px solid #DDDDDD; background-color: white; color: black; }"

        # Kanji Viewer Worker Management
        self.related_kanji_worker: Optional[RelatedKanjiWorker] = None
        self.current_related_kanji_process_id = 0
        self._worker_management_mutex = QMutex() # Protects worker and process_id
        self._kanji_viewer_data_loaded_successfully = False

        # Kanji Viewer Timers and State
        self.kv_resize_timer: Optional[QTimer] = None
        self._kv_initial_char_to_display = "永" # Default char for KV if no glyph active
        self._kv_available_fonts: List[str] = [] # Actual font family names

        self._kv_char_to_update: Optional[str] = None # Char for deferred update
        self._kv_deferred_update_timer: Optional[QTimer] = None
        self._kv_update_delay_ms = REFERENCE_GLYPH_DISPLAY_DELAY

        self.kv_display_mode = MainWindow.KV_MODE_FONT_DISPLAY # Default mode
        self.kv_mode_font_button: Optional[QPushButton] = None
        self.kv_mode_written_button: Optional[QPushButton] = None
        self.kv_mode_hide_button: Optional[QPushButton] = None
        self.kv_mode_button_group: Optional[QButtonGroup] = None

        self.font_bookmarks: List[str] = []
        self._load_font_bookmarks()


        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        self._init_ui_for_kanji_viewer_panel()
        if self.kanji_viewer_panel_widget:
            main_layout.addWidget(self.kanji_viewer_panel_widget, 0)

        main_layout.addWidget(self.glyph_grid_widget, 0)
        main_layout.addWidget(self.drawing_editor_widget, 1)
        main_layout.addWidget(self.properties_widget, 0)

        self.file_menu: Optional[QMenu] = None
        self.new_project_action: Optional[QAction] = None
        self.open_project_action: Optional[QAction] = None
        self.exit_action: Optional[QAction] = None
        self.edit_menu: Optional[QMenu] = None
        self.batch_adv_width_action: Optional[QAction] = None
        self.batch_import_glyphs_action: Optional[QAction] = None
        self.batch_import_reference_images_action: Optional[QAction] = None

        self._create_menus()
        self.setStatusBar(QStatusBar(self))

        # Initialize KV deferred update timer
        self._kv_deferred_update_timer = QTimer(self)
        self._kv_deferred_update_timer.setSingleShot(True)
        self._kv_deferred_update_timer.timeout.connect(self._process_deferred_kv_update)

        # Connect signals (moved here for clarity after UI elements are defined)
        if self.kanji_viewer_font_combo:
            self.kanji_viewer_font_combo.currentTextChanged.connect(
                self._handle_kv_font_combo_changed
            )

        self.glyph_grid_widget.glyph_selected_signal.connect(lambda char: self.load_glyph_for_editing(char, is_vrt2_edit_mode=False))
        self.glyph_grid_widget.vrt2_glyph_selected_signal.connect(lambda char: self.load_glyph_for_editing(char, is_vrt2_edit_mode=True))

        self.drawing_editor_widget.canvas.glyph_modified_signal.connect(self.handle_glyph_modification_from_canvas)
        self.drawing_editor_widget.reference_image_selected_signal.connect(self.save_reference_image_async)
        self.drawing_editor_widget.reference_image_deleted_signal.connect(self.handle_delete_reference_image_async)
        self.drawing_editor_widget.glyph_to_reference_and_reset_requested.connect(self.handle_glyph_to_reference_and_reset)

        self.properties_widget.character_set_changed_signal.connect(self.update_project_character_set)
        self.properties_widget.rotated_vrt2_set_changed_signal.connect(self.update_rotated_vrt2_set)
        self.properties_widget.non_rotated_vrt2_set_changed_signal.connect(self.update_non_rotated_vrt2_set)
        self.properties_widget.font_name_changed_signal.connect(self.update_font_name)
        self.properties_widget.font_weight_changed_signal.connect(self.update_font_weight)
        self.properties_widget.export_font_signal.connect(self.handle_export_font)

        self.drawing_editor_widget.gui_setting_changed_signal.connect(self.save_gui_setting_async)
        self.drawing_editor_widget.vrt2_edit_mode_toggled.connect(self.handle_vrt2_edit_mode_toggle)
        self.drawing_editor_widget.transfer_to_vrt2_requested.connect(self.handle_transfer_to_vrt2)
        self.drawing_editor_widget.advance_width_changed_signal.connect(self.save_glyph_advance_width_async)

        self._update_ui_for_project_state() # Set initial UI state (mostly disabled)
        self._load_kanji_viewer_data_and_fonts() # Load KV data regardless of project

        QTimer.singleShot(0, self._kv_initial_display_setup) # Setup KV display after event loop starts
        QTimer.singleShot(100, self._update_bookmark_button_state) # Update bookmark button after fonts populate


    def _get_font_bookmarks_path(self) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(script_dir, FONT_BOOKMARKS_FILENAME)

    def _load_font_bookmarks(self):
        bookmarks_path = self._get_font_bookmarks_path()
        if os.path.exists(bookmarks_path):
            try:
                with open(bookmarks_path, 'r', encoding='utf-8') as f:
                    loaded_bookmarks = json.load(f)
                if isinstance(loaded_bookmarks, list) and \
                   all(isinstance(item, str) for item in loaded_bookmarks):
                    self.font_bookmarks = loaded_bookmarks
                else:
                    # Invalid format, reset
                    self.font_bookmarks = []
            except json.JSONDecodeError:
                # File corrupted, reset
                self.font_bookmarks = []
            except Exception as e:
                # Other errors, reset for safety
                self.font_bookmarks = []
        else:
            self.font_bookmarks = [] # No file, start with empty list

    def _save_font_bookmarks(self):
        bookmarks_path = self._get_font_bookmarks_path()
        try:
            with open(bookmarks_path, 'w', encoding='utf-8') as f:
                json.dump(self.font_bookmarks, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # Log or show non-critical error, as this is not core functionality
            QMessageBox.warning(self, "ブックマーク保存エラー", f"フォントブックマークの保存に失敗しました: {e}")

    def _get_actual_font_name(self, display_name: str) -> str:
        if display_name.startswith("★ "):
            return display_name[2:]
        return display_name

    def _handle_kv_font_combo_changed(self, selected_display_name: str):
        self._update_bookmark_button_state() # Update button based on new selection
        
        actual_font_name = self._get_actual_font_name(selected_display_name)
        if actual_font_name and self.current_project_path and not self._project_loading_in_progress: 
            self.save_gui_setting_async(SETTING_KV_CURRENT_FONT, actual_font_name)
        
        char_to_update_kv_with = self.drawing_editor_widget.canvas.current_glyph_character or self._kv_initial_char_to_display
        self._trigger_kanji_viewer_update_for_current_glyph(char_to_update_kv_with)

    def _init_ui_for_kanji_viewer_panel(self):
        self.kanji_viewer_panel_widget = QWidget()
        self.kanji_viewer_panel_widget.setFixedWidth(350)
        self.kanji_viewer_panel_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        
        panel_layout = QVBoxLayout(self.kanji_viewer_panel_widget)
        panel_layout.setContentsMargins(5, 5, 5, 5)

        font_selection_layout = QHBoxLayout()
        font_selection_layout.addWidget(QLabel("参照:"))
        self.kanji_viewer_font_combo = QComboBox()
        self.kanji_viewer_font_combo.setMinimumWidth(150)
        font_selection_layout.addWidget(self.kanji_viewer_font_combo, 1)

        self.bookmark_font_button = QPushButton("★") # Star symbol for bookmark
        self.bookmark_font_button.setToolTip("現在のフォントをブックマークに追加/削除")
        fm = self.fontMetrics() # For sizing
        button_width = fm.horizontalAdvance("★") + fm.horizontalAdvance("  ") # Padding
        self.bookmark_font_button.setFixedWidth(button_width) 
        self.bookmark_font_button.setCheckable(True) # To indicate bookmarked state (though style will show it)
        self.bookmark_font_button.clicked.connect(self._toggle_font_bookmark)
        font_selection_layout.addWidget(self.bookmark_font_button)
        
        panel_layout.addLayout(font_selection_layout)


        self.kanji_viewer_display_label = QLabel()
        self.kanji_viewer_display_label.setFixedSize(300, 300) # Fixed size for consistent display
        self.kanji_viewer_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.kanji_viewer_display_label.setStyleSheet("border: 2px solid #CCCCCC; background-color: white; color: black;")
        default_font = QFont()
        default_font.setPointSize(10) # Small default, will be overridden
        self.kanji_viewer_display_label.setFont(default_font)
        self.kanji_viewer_display_label.setText(self._kv_initial_char_to_display)
        panel_layout.addWidget(self.kanji_viewer_display_label, 0, alignment=Qt.AlignmentFlag.AlignCenter)

        self.kanji_viewer_related_tabs = VerticalTabWidget()
        self.kanji_viewer_related_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panel_layout.addWidget(self.kanji_viewer_related_tabs, 1)

        # Kanji Viewer Mode Buttons
        kv_mode_button_layout = QHBoxLayout()
        self.kv_mode_button_group = QButtonGroup(self)
        self.kv_mode_button_group.setExclusive(True)

        self.kv_mode_font_button = QPushButton("外部フォントを参照")
        self.kv_mode_font_button.setCheckable(True)
        self.kv_mode_font_button.setToolTip("選択フォントで関連漢字を表示")
        kv_mode_button_layout.addWidget(self.kv_mode_font_button)
        self.kv_mode_button_group.addButton(self.kv_mode_font_button, MainWindow.KV_MODE_FONT_DISPLAY)

        self.kv_mode_written_button = QPushButton("プロジェクトを参照")
        self.kv_mode_written_button.setCheckable(True)
        self.kv_mode_written_button.setToolTip("書き込み済みの関連グリフを一覧表示")
        kv_mode_button_layout.addWidget(self.kv_mode_written_button)
        self.kv_mode_button_group.addButton(self.kv_mode_written_button, MainWindow.KV_MODE_WRITTEN_GLYPHS)

        self.kv_mode_hide_button = QPushButton("非表示")
        self.kv_mode_hide_button.setCheckable(True)
        self.kv_mode_hide_button.setToolTip("関連漢字表示を隠す")
        kv_mode_button_layout.addWidget(self.kv_mode_hide_button)
        self.kv_mode_button_group.addButton(self.kv_mode_hide_button, MainWindow.KV_MODE_HIDDEN)
        
        panel_layout.addLayout(kv_mode_button_layout)
        
        self.kv_mode_button_group.buttonToggled.connect(self._on_kv_display_mode_button_toggled)

        # Set default KV mode (usually written glyphs)
        if self.kv_mode_written_button: # Ensure button exists
             self.kv_mode_written_button.setChecked(True)


    def _load_kanji_viewer_data_and_fonts(self):
        error_title = "関連漢字データ読み込みエラー"
        def show_error_and_log(msg_key: str, is_critical: bool = False):
            full_msg = f"{msg_key} が見つからないか、読み込めませんでした。"
            # Log to console or status bar if needed
            if self.kanji_viewer_display_label: self.kanji_viewer_display_label.setText("関連データ\n読込失敗")
            if is_critical: QMessageBox.critical(self, error_title, f"{full_msg}\nアプリケーションの関連漢字機能は利用できません。")
            else: QMessageBox.warning(self, error_title, f"{full_msg}\n関連漢字機能は利用できません。")
            return False

        kanji_to_data_filepath = get_data_file_path(self.KANJI_TO_DATA_FILENAME)
        if not kanji_to_data_filepath:
            self._kanji_viewer_data_loaded_successfully = False
            show_error_and_log(self.KANJI_TO_DATA_FILENAME)
            self._populate_kv_fonts() # Still try to populate fonts
            return
        self.kanji_radicals_data = load_json_data(kanji_to_data_filepath)
        if not self.kanji_radicals_data:
            self._kanji_viewer_data_loaded_successfully = False
            show_error_and_log(self.KANJI_TO_DATA_FILENAME)
            self._populate_kv_fonts(); return

        data_to_kanji_filepath = get_data_file_path(self.DATA_TO_KANJI_FILENAME)
        if not data_to_kanji_filepath:
            self._kanji_viewer_data_loaded_successfully = False
            show_error_and_log(self.DATA_TO_KANJI_FILENAME)
            self._populate_kv_fonts(); return
        self.radical_to_kanji_data = load_json_data(data_to_kanji_filepath)
        if not self.radical_to_kanji_data:
            self._kanji_viewer_data_loaded_successfully = False
            show_error_and_log(self.DATA_TO_KANJI_FILENAME)
            self._populate_kv_fonts(); return
            
        self._kanji_viewer_data_loaded_successfully = True
        # Data loaded, now populate fonts
        self._populate_kv_fonts()


    def _populate_kv_fonts(self):
        if not self.kanji_viewer_font_combo: return
        
        current_selected_display_name = self.kanji_viewer_font_combo.currentText()
        current_selected_actual_name = self._get_actual_font_name(current_selected_display_name)

        self.kanji_viewer_font_combo.blockSignals(True)
        self.kanji_viewer_font_combo.clear()
        self._kv_available_fonts.clear() # Clear list of actual names
        
        system_families = QFontDatabase.families()
        
        # Validate bookmarks against available system fonts
        valid_bookmarked_fonts = []
        updated_bookmarks_list = False # Flag to re-save if bookmarks changed
        
        for bookmarked_font in list(self.font_bookmarks): # Iterate copy for safe removal
            if bookmarked_font in system_families:
                valid_bookmarked_fonts.append(bookmarked_font)
            else:
                # Font no longer in system, remove from bookmarks
                self.font_bookmarks.remove(bookmarked_font)
                updated_bookmarks_list = True
        
        if updated_bookmarks_list:
            self._save_font_bookmarks()

        unbookmarked_system_fonts = [f for f in system_families if f not in valid_bookmarked_fonts]

        # Sort bookmarked and unbookmarked lists separately
        valid_bookmarked_fonts.sort()
        unbookmarked_system_fonts.sort()

        display_items = []
        # Add bookmarked fonts first, prefixed with a star
        for font_name in valid_bookmarked_fonts:
            display_items.append(f"★ {font_name}")
            self._kv_available_fonts.append(font_name) # Store actual name
        
        # Then add unbookmarked system fonts
        for font_name in unbookmarked_system_fonts:
            display_items.append(font_name)
            self._kv_available_fonts.append(font_name)

        if display_items:
            self.kanji_viewer_font_combo.addItems(display_items)
            
            # Try to reselect previously selected font
            new_index_to_select = -1
            prospective_display_name_bookmarked = f"★ {current_selected_actual_name}"
            if current_selected_actual_name in valid_bookmarked_fonts and prospective_display_name_bookmarked in display_items:
                new_index_to_select = display_items.index(prospective_display_name_bookmarked)
            elif current_selected_actual_name in display_items: # Check for plain name match
                 try:
                     new_index_to_select = display_items.index(current_selected_actual_name)
                 except ValueError: # Should not happen if in display_items
                     pass 
            
            if new_index_to_select != -1:
                self.kanji_viewer_font_combo.setCurrentIndex(new_index_to_select)
            elif self.kanji_viewer_font_combo.count() > 0: # Fallback to a preferred default or first item
                default_selection_candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "ヒラギノ角ゴ ProN W3"]
                selected_idx_default = -1
                for pref_font_actual in default_selection_candidates:
                    pref_font_display_bookmarked = f"★ {pref_font_actual}"
                    if pref_font_display_bookmarked in display_items:
                        selected_idx_default = display_items.index(pref_font_display_bookmarked)
                        break
                    elif pref_font_actual in display_items:
                        selected_idx_default = display_items.index(pref_font_actual)
                        break
                
                if selected_idx_default != -1: self.kanji_viewer_font_combo.setCurrentIndex(selected_idx_default)
                else: self.kanji_viewer_font_combo.setCurrentIndex(0) # Absolute fallback
        else:
            # No fonts available at all (highly unlikely on most systems)
            if self.kanji_viewer_display_label:
                self.kanji_viewer_display_label.setText("フォント\nなし")
                self.kanji_viewer_display_label.setFont(QFont()) # Reset font

        self.kanji_viewer_font_combo.blockSignals(False)
        self._update_bookmark_button_state() # Ensure button state is correct after repopulation

    def _update_bookmark_button_state(self):
        if not self.bookmark_font_button or not self.kanji_viewer_font_combo:
            return

        current_display_name = self.kanji_viewer_font_combo.currentText()
        if not current_display_name: # No item selected or combo is empty
            self.bookmark_font_button.setChecked(False) # Should reflect unbookmarked
            self.bookmark_font_button.setEnabled(False)
            return

        actual_font_name = self._get_actual_font_name(current_display_name)
        is_bookmarked = actual_font_name in self.font_bookmarks
        
        self.bookmark_font_button.blockSignals(True) # Avoid triggering clicked signal
        self.bookmark_font_button.setChecked(is_bookmarked)
        self.bookmark_font_button.blockSignals(False)
        
        tooltip_action = "削除" if is_bookmarked else "追加"
        self.bookmark_font_button.setToolTip(f"フォント「{actual_font_name}」をブックマークから{tooltip_action}")
        
        # Enable button if there's a font selected
        self.bookmark_font_button.setEnabled(bool(actual_font_name) and self.kanji_viewer_font_combo.count() > 0)

        # Visual cue for bookmarked state (optional, can be done with stylesheet)
        if is_bookmarked:
            self.bookmark_font_button.setStyleSheet("""
                QPushButton { 
                    background-color: palette(highlight); 
                    color: palette(highlighted-text); 
                    border: 1px solid palette(dark);
                }
                QPushButton:hover {
                    background-color: palette(highlight); 
                }
            """)
        else:
            self.bookmark_font_button.setStyleSheet("""
                QPushButton { 
                    /* Default appearance */
                }
                QPushButton:hover {
                    background-color: palette(button); /* Standard hover */
                }
            """)

    def _toggle_font_bookmark(self):
        if not self.kanji_viewer_font_combo: return
        
        current_display_name = self.kanji_viewer_font_combo.currentText()
        if not current_display_name: return

        actual_font_name = self._get_actual_font_name(current_display_name)
        if not actual_font_name: return

        if actual_font_name in self.font_bookmarks:
            self.font_bookmarks.remove(actual_font_name)
            self.statusBar().showMessage(f"フォント「{actual_font_name}」をブックマークから削除しました。", 3000)
        else:
            self.font_bookmarks.append(actual_font_name)
            self.font_bookmarks.sort() # Keep bookmarks sorted
            self.statusBar().showMessage(f"フォント「{actual_font_name}」をブックマークに追加しました。", 3000)
            
        self._save_font_bookmarks()
        self._populate_kv_fonts() # Repopulate to reflect change in order/prefix, reselects current

    def _kv_initial_display_setup(self):
        # Ensure a font is selected in the combo if possible
        if self.kanji_viewer_font_combo and self.kanji_viewer_font_combo.count() > 0:
            if self.kanji_viewer_font_combo.currentIndex() == -1: # No selection yet
                self.kanji_viewer_font_combo.setCurrentIndex(0) # Select first available
        elif self.kanji_viewer_display_label: # No fonts in combo
            self.kanji_viewer_display_label.setText("利用可能な\nフォントが\nありません")
            self.kanji_viewer_display_label.setFont(QFont())


        # Determine initial character for KV display
        current_glyph = self.drawing_editor_widget.canvas.current_glyph_character
        char_to_display_initially = current_glyph or self._kv_initial_char_to_display
        
        # Trigger update (will be deferred)
        with QMutexLocker(self._worker_management_mutex):
            self._kv_char_to_update = char_to_display_initially
        if self.current_project_path or not self._project_loading_in_progress: # Only trigger if project loaded or no load active
            self._kv_deferred_update_timer.start(self._kv_update_delay_ms)


    def _kv_calculate_optimal_font_size(self, char: str, rect: QRect, family: str, margin: float = 0.8) -> int:
        if not char or rect.isEmpty() or not family: return 1 # Minimal size
        
        font = QFont(family)
        low = 1
        high = min(max(1, rect.height()), 1200) # Reasonable upper bound for pixel size
        best_size = 1
        iterations = 0; max_iterations = 100 # Safety break for binary search

        # Target dimensions within the rect, applying margin
        target_width = rect.width() * margin
        target_height = rect.height() * margin

        while low <= high and iterations < max_iterations:
            iterations += 1
            mid = (low + high) // 2
            if mid == 0: low = 1; continue # Ensure pixel size is at least 1

            font.setPixelSize(mid)
            fm = QFontMetrics(font)
            current_bound_rect = fm.boundingRect(char)
            
            if current_bound_rect.isNull() or current_bound_rect.isEmpty(): # Font might not support char at this size
                high = mid -1 # Try smaller
                continue

            if current_bound_rect.width() <= target_width and current_bound_rect.height() <= target_height:
                best_size = mid
                low = mid + 1 # Try larger
            else:
                high = mid - 1 # Too big, try smaller
        
        if iterations >= max_iterations:
            # Log warning if binary search takes too long (e.g. font metrics issue)
            pass
            
        return max(1, best_size)


    def _kv_set_label_font_and_text(self, label: QLabel, char: str, family: str, rect: QRect, margin: float = 0.9):
        if not label: return
        if not char or not family: # No char or family, clear the label
            label.setText("")
            label.setFont(QFont()) # Reset to default font
            return
        try:
            px_size = self._kv_calculate_optimal_font_size(char, rect, family, margin)
            font = QFont(family)
            font.setPixelSize(px_size)
            label.setFont(font)
            label.setText(char)
            label.setAlignment(Qt.AlignmentFlag.AlignCenter) # Ensure centered
        except Exception as e: # Catch any font setting errors
            label.setText("ERR") # Error indicator
            label.setFont(QFont())


    @Slot()
    def _process_deferred_kv_update(self):
        char_to_process = None
        with QMutexLocker(self._worker_management_mutex):
            if self._kv_char_to_update:
                char_to_process = self._kv_char_to_update
                # self._kv_char_to_update = None # Clear after processing, or keep for resize?
                # Keeping it allows resize to re-trigger for same char.
        if char_to_process:
            self._trigger_kanji_viewer_update_for_current_glyph(char_to_process)

    @Slot(QAbstractButton, bool)
    def _on_kv_display_mode_button_toggled(self, button: QAbstractButton, checked: bool):
        if not checked: return # Only act on the button that was checked

        new_mode = self.kv_mode_button_group.id(button)
        if new_mode == self.kv_display_mode and self.current_project_path and not self._project_loading_in_progress : return # No change if mode is same (and project context is stable)
            
        self.kv_display_mode = new_mode
        if self.current_project_path and not self._project_loading_in_progress: # Save if project loaded
            self.save_gui_setting_async(SETTING_KV_DISPLAY_MODE, str(new_mode))

        current_char_for_kv = self.drawing_editor_widget.canvas.current_glyph_character or self._kv_initial_char_to_display
        self._trigger_kanji_viewer_update_for_current_glyph(current_char_for_kv)

    def _trigger_kanji_viewer_update_for_current_glyph(self, current_char: str):
        if self._project_loading_in_progress: return # Don't update during load
        if not self.kanji_viewer_display_label or not self.kanji_viewer_related_tabs or not self.kanji_viewer_font_combo:
            return # UI elements not ready

        font_family_display_name = self.kanji_viewer_font_combo.currentText()
        font_family_for_main_display = self._get_actual_font_name(font_family_display_name) 

        if not font_family_for_main_display and self._kv_available_fonts: # Fallback if current selection is somehow invalid
            font_family_for_main_display = self._kv_available_fonts[0] # Use first available
        elif not font_family_for_main_display and not self._kv_available_fonts: # No fonts at all
             if self.kanji_viewer_display_label: 
                 self.kanji_viewer_display_label.setText("フォント\n選択不可")
                 self.kanji_viewer_display_label.setFont(QFont())
             if self.kanji_viewer_related_tabs: self.kanji_viewer_related_tabs.clear()
             return 

        # Handle KV_MODE_HIDDEN
        if self.kv_display_mode == MainWindow.KV_MODE_HIDDEN:
            if self.kanji_viewer_related_tabs: self.kanji_viewer_related_tabs.clear()
            if self.kanji_viewer_display_label:
                if current_char and len(current_char) == 1: # Still show current char in main display
                    self._kv_set_label_font_and_text(self.kanji_viewer_display_label, current_char, font_family_for_main_display, self.kanji_viewer_display_label.rect(), 0.85)
                else: # No valid char, clear display
                    self.kanji_viewer_display_label.setText("") 
                    self.kanji_viewer_display_label.setFont(QFont())
            return # No further processing for hidden mode

        # Handle invalid current_char for other modes
        if not current_char or len(current_char) != 1: # Requires single char for radical search
            if self.kanji_viewer_display_label:
                self.kanji_viewer_display_label.setText("")
                self.kanji_viewer_display_label.setFont(QFont())
            if self.kanji_viewer_related_tabs: self.kanji_viewer_related_tabs.clear()
            with QMutexLocker(self._worker_management_mutex): # Cancel any ongoing worker
                if self.related_kanji_worker and self.related_kanji_worker.isRunning():
                    self.related_kanji_worker.cancel()
            return

        # Update main display label
        if self.kanji_viewer_display_label:
            self._kv_set_label_font_and_text(self.kanji_viewer_display_label, current_char, font_family_for_main_display, self.kanji_viewer_display_label.rect(), 0.85)
            self.kanji_viewer_display_label.setStyleSheet("border: 2px solid #CCCCCC; background-color: white; color: black;")

        # Handle no data loaded for related kanji
        if not self._kanji_viewer_data_loaded_successfully:
            if self.kanji_viewer_related_tabs:
                self.kanji_viewer_related_tabs.clear()
                if self.kanji_viewer_related_tabs.count() == 0: # Avoid adding if already showing error
                    no_data_label = QLabel("関連漢字データが\n読み込まれていません。")
                    no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.kanji_viewer_related_tabs.addTab(no_data_label, "情報")
            return

        # Start worker for related kanji
        with QMutexLocker(self._worker_management_mutex):
            self.current_related_kanji_process_id += 1
            process_id = self.current_related_kanji_process_id
            if self.related_kanji_worker and self.related_kanji_worker.isRunning():
                self.related_kanji_worker.cancel() # Cancel previous worker

            # Determine font size for related kanji items based on available width
            tab_bar_instance = self.kanji_viewer_related_tabs.tabBar()
            tab_bar_width = tab_bar_instance.tab_fixed_width if hasattr(tab_bar_instance, 'tab_fixed_width') else 60
            
            content_area_width = self.kanji_viewer_related_tabs.width() - tab_bar_width - 20 # Approx padding/margins
            if content_area_width <= 0: # Fallback if width calculation is off
                content_area_width = self.kanji_viewer_panel_widget.width() - tab_bar_width - 30 if self.kanji_viewer_panel_widget else 250 - tab_bar_width - 30
            if content_area_width <= 0: content_area_width = 250

            ideal_item_width_for_3_cols = content_area_width / 3 - 10 # Target 3 cols, with spacing
            font_px_size_for_related = max(1, int(ideal_item_width_for_3_cols * 0.7)) # Font size ~70% of item width
            font_px_size_for_related = min(font_px_size_for_related, 50) # Max font size cap
            font_px_size_for_related = max(font_px_size_for_related, 16) # Min font size
            
            effective_font_family_for_worker = font_family_for_main_display # Use same font for related items if KV_MODE_FONT_DISPLAY
            
            new_worker = RelatedKanjiWorker(
                process_id, current_char, self.kanji_radicals_data, self.radical_to_kanji_data,
                effective_font_family_for_worker, font_px_size_for_related, self
            )
            new_worker.result_ready.connect(self._handle_kv_related_kanji_result)
            new_worker.error_occurred.connect(self._handle_kv_worker_error)
            new_worker.finished.connect(self._on_kv_worker_finished) # Cleanup
            new_worker.finished.connect(new_worker.deleteLater)
            self.related_kanji_worker = new_worker
            self.related_kanji_worker.start()

    @Slot(int, dict, str, int)
    def _handle_kv_related_kanji_result(self, process_id: int, results_dict: dict, font_family_from_worker: str, font_px_size: int):
        with QMutexLocker(self._worker_management_mutex):
            if process_id != self.current_related_kanji_process_id:
                return # Result from an old, cancelled worker
        
        if not self.kanji_viewer_related_tabs:
            return

        # Preserve current tab selection if possible
        current_tab_text_before_clear = ""
        current_tab_idx = self.kanji_viewer_related_tabs.currentIndex()
        if self.kanji_viewer_related_tabs.count() > 0 and current_tab_idx != -1:
            current_tab_text_before_clear = self.kanji_viewer_related_tabs.tabText(current_tab_idx)
        
        self.kanji_viewer_related_tabs.clear()

        char_for_msg_display = self.drawing_editor_widget.canvas.current_glyph_character
        if not char_for_msg_display: # Fallback if no char in editor (e.g., initial load)
            if self.related_kanji_worker and self.related_kanji_worker.input_char:
                char_for_msg_display = self.related_kanji_worker.input_char
            else:
                char_for_msg_display = "選択文字"

        effective_results_dict = results_dict
        # Filter results if in KV_MODE_WRITTEN_GLYPHS
        if self.kv_display_mode == MainWindow.KV_MODE_WRITTEN_GLYPHS:
            if not self.db_manager.db_path: # Should not happen if project is loaded
                no_project_label = QLabel("プロジェクトがロードされていません。")
                no_project_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.kanji_viewer_related_tabs.addTab(no_project_label, "エラー")
                return

            # Use the main window's cache of written glyphs for KV_MODE_WRITTEN_GLYPHS
            self.written_glyphs_cache_for_kv: Dict[str, Optional[QPixmap]] = {
                char: pixmap 
                for char, pixmap in self._all_standard_glyphs_pixmap_cache # Use MainWindow's cache
                if pixmap is not None # Only include those with actual pixmaps (written)
            }
            
            filtered_dict = {}
            for radical, kanji_list in results_dict.items():
                written_kanji_in_list = [k_char for k_char in kanji_list if k_char in self.written_glyphs_cache_for_kv]
                if written_kanji_in_list:
                    filtered_dict[radical] = written_kanji_in_list
            effective_results_dict = filtered_dict
            if not effective_results_dict: # If filtering removed all results
                 msg_text = f"「{char_for_msg_display}」に関連するグリフは\n見つかりませんでした。"
                 no_results_label = QLabel(msg_text)
                 no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter); no_results_label.setWordWrap(True)
                 container_widget = QWidget(); layout = QVBoxLayout(container_widget)
                 layout.addWidget(no_results_label); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                 self.kanji_viewer_related_tabs.addTab(container_widget, "") # Empty tab text
                 return

        if not effective_results_dict:
            msg_text = f"「{char_for_msg_display}」の構成部首データがないか、\n関連する漢字が見つかりませんでした。"
            if self.kv_display_mode == MainWindow.KV_MODE_WRITTEN_GLYPHS:
                 msg_text = f"「{char_for_msg_display}」に関連するグリフは\n見つかりませんでした。"
            elif self.kanji_radicals_data and char_for_msg_display in self.kanji_radicals_data and \
                 not self.kanji_radicals_data.get(char_for_msg_display): # Char has entry, but list of radicals is empty
                msg_text = f"「{char_for_msg_display}」には構成部首が登録されていません。"
            elif self.kanji_radicals_data and char_for_msg_display not in self.kanji_radicals_data:
                 msg_text = f"「{char_for_msg_display}」のデータがありません。"

            no_results_label = QLabel(msg_text)
            no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter); no_results_label.setWordWrap(True)
            container_widget = QWidget(); layout = QVBoxLayout(container_widget)
            layout.addWidget(no_results_label); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.kanji_viewer_related_tabs.addTab(container_widget, "") # Empty tab text
            return

        # Determine item size and layout for related kanji display
        tab_bar_instance = self.kanji_viewer_related_tabs.tabBar()
        tab_bar_width = tab_bar_instance.tab_fixed_width if hasattr(tab_bar_instance, 'tab_fixed_width') else 60
        content_area_width = self.kanji_viewer_related_tabs.width() - tab_bar_width - \
                             self.kanji_viewer_related_tabs.contentsMargins().left() - \
                             self.kanji_viewer_related_tabs.contentsMargins().right() - 10 # Account for margins/spacing
        if content_area_width <= 0: content_area_width = self.kanji_viewer_panel_widget.width() - tab_bar_width - 30 if self.kanji_viewer_panel_widget else 250 - tab_bar_width -30
        if content_area_width <= 0: content_area_width = 250

        item_side_length = font_px_size + 12 # Add padding around font size
        item_side_length = max(item_side_length, 36); item_side_length = min(item_side_length, 80) # Clamp item size
        preview_display_size = QSize(item_side_length - 4, item_side_length - 4) # Size for pixmap preview

        MAX_COLS = max(1, content_area_width // (item_side_length + 5)) # Calculate max columns based on width

        any_tabs_added = False; new_selected_index_to_restore = -1
        
        related_kanji_font: QFont # Define font for KV_MODE_FONT_DISPLAY
        if self.kv_display_mode == MainWindow.KV_MODE_FONT_DISPLAY:
            related_kanji_font = QFont(font_family_from_worker) # Use font from worker
            related_kanji_font.setPixelSize(font_px_size)

        sorted_radicals = sorted(effective_results_dict.keys())
        for radical in sorted_radicals:
            kanji_list = effective_results_dict[radical]
            if not kanji_list: continue
            
            any_tabs_added = True; tab_content_widget = QWidget()
            tab_content_widget.setStyleSheet(self.related_kanji_label_style) # Apply consistent styling
            
            tab_layout = QGridLayout(tab_content_widget)
            tab_layout.setSpacing(5); tab_layout.setContentsMargins(5, 5, 5, 5)
            
            row, col = 0, 0
            for kanji_char_to_display in kanji_list:
                kanji_label = QLabel()
                kanji_label.setFixedSize(item_side_length, item_side_length)
                kanji_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                if self.kv_display_mode == MainWindow.KV_MODE_WRITTEN_GLYPHS:
                    glyph_pixmap = self.written_glyphs_cache_for_kv.get(kanji_char_to_display)
                    if glyph_pixmap:
                        scaled_pixmap = glyph_pixmap.scaled(preview_display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        kanji_label.setPixmap(scaled_pixmap)
                    else: # Should not happen if written_glyphs_cache_for_kv is up-to-date
                        kanji_label.setText(kanji_char_to_display) # Fallback to text
                        font_for_fallback = QFont(self.font().family()) # Use app's default font
                        font_for_fallback.setPixelSize(max(10, int(item_side_length * 0.6)))
                        kanji_label.setFont(font_for_fallback)
                else: # KV_MODE_FONT_DISPLAY
                    kanji_label.setFont(related_kanji_font)
                    kanji_label.setText(kanji_char_to_display)
                
                tab_layout.addWidget(kanji_label, row, col)
                col += 1
                if col >= MAX_COLS: col = 0; row += 1
            
            # Add spacers to fill the last row if not full
            if col > 0 and col < MAX_COLS :
                for c_fill in range(col, MAX_COLS):
                    spacer_item = QSpacerItem(item_side_length, item_side_length, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                    tab_layout.addItem(spacer_item, row, c_fill)

            tab_layout.setRowStretch(row + 1, 1); tab_layout.setColumnStretch(MAX_COLS, 1) # Allow content to expand
            scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True)
            scroll_area.setWidget(tab_content_widget)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            tab_idx = self.kanji_viewer_related_tabs.addTab(scroll_area, radical)
            if radical == current_tab_text_before_clear: new_selected_index_to_restore = tab_idx

        if not any_tabs_added and effective_results_dict: # Results existed but all were empty lists
            no_results_label = QLabel(f"「{char_for_msg_display}」の各構成部首を共有する\n他の漢字は見つかりませんでした。")
            if self.kv_display_mode == MainWindow.KV_MODE_WRITTEN_GLYPHS:
                 no_results_label.setText(f"「{char_for_msg_display}」の各構成部首を共有する\nグリフは見つかりませんでした。")
            no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter); no_results_label.setWordWrap(True)
            container_widget = QWidget(); layout = QVBoxLayout(container_widget)
            layout.addWidget(no_results_label); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.kanji_viewer_related_tabs.addTab(container_widget, "")
            
        if new_selected_index_to_restore != -1: self.kanji_viewer_related_tabs.setCurrentIndex(new_selected_index_to_restore)
        elif self.kanji_viewer_related_tabs.count() > 0: self.kanji_viewer_related_tabs.setCurrentIndex(0) # Select first tab if new set

    @Slot(int, str)
    def _handle_kv_worker_error(self, process_id: int, error_message: str):
        with QMutexLocker(self._worker_management_mutex):
            if process_id != self.current_related_kanji_process_id: return # Error from old worker
        if not self.kanji_viewer_related_tabs: return

        self.kanji_viewer_related_tabs.clear()
        lbl = QLabel(f"関連漢字の取得エラー:\n{error_message}")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setWordWrap(True)
        cw = QWidget(); lo = QVBoxLayout(cw); lo.addWidget(lbl)
        self.kanji_viewer_related_tabs.addTab(cw, "エラー")

    @Slot()
    def _on_kv_worker_finished(self):
        sender_worker = self.sender() # Get the worker that emitted the signal
        if not isinstance(sender_worker, RelatedKanjiWorker): return # Type check
        
        with QMutexLocker(self._worker_management_mutex):
            if self.related_kanji_worker is sender_worker: # Only clear if it's the current worker
                self.related_kanji_worker = None # Allow it to be garbage collected


    def _create_menus(self):
        menu_bar = self.menuBar()
        self.file_menu = menu_bar.addMenu("&ファイル")
        self.new_project_action = QAction("&新規プロジェクト...", self)
        self.new_project_action.triggered.connect(self.new_project)
        self.file_menu.addAction(self.new_project_action)

        self.open_project_action = QAction("&プロジェクトを開く...", self)
        self.open_project_action.triggered.connect(self.open_project)
        self.file_menu.addAction(self.open_project_action)
        
        self.file_menu.addSeparator()
        self.exit_action = QAction("&終了", self)
        self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action)

        self.edit_menu = menu_bar.addMenu("&編集")
        self.batch_adv_width_action = QAction("文字送り幅一括編集...", self)
        self.batch_adv_width_action.triggered.connect(self.open_batch_advance_width_dialog)
        self.edit_menu.addAction(self.batch_adv_width_action)
        
        self.edit_menu.addSeparator()
        self.batch_import_glyphs_action = QAction("グリフの一括読み込み...", self)
        self.batch_import_glyphs_action.triggered.connect(self.batch_import_glyphs)
        self.edit_menu.addAction(self.batch_import_glyphs_action)
        
        self.batch_import_reference_images_action = QAction("下書きの一括読み込み...", self)
        self.batch_import_reference_images_action.triggered.connect(self.batch_import_reference_images)
        self.edit_menu.addAction(self.batch_import_reference_images_action)

    def _set_project_loading_state(self, loading: bool):
        self._project_loading_in_progress = loading
        
        # Enable/disable main UI components
        self.drawing_editor_widget.setEnabled(not loading and self.current_project_path is not None)
        self.properties_widget.setEnabled(not loading and self.current_project_path is not None)
        self.glyph_grid_widget.setEnabled(not loading and self.current_project_path is not None)
        
        # Menu items
        if self.batch_adv_width_action: self.batch_adv_width_action.setEnabled(not loading and self.current_project_path is not None)
        if self.batch_import_glyphs_action: self.batch_import_glyphs_action.setEnabled(not loading and self.current_project_path is not None)
        if self.batch_import_reference_images_action: self.batch_import_reference_images_action.setEnabled(not loading and self.current_project_path is not None)

        if self.new_project_action: self.new_project_action.setEnabled(not loading)
        if self.open_project_action: self.open_project_action.setEnabled(not loading)

        # Status bar message
        if loading:
            self.statusBar().showMessage("プロジェクト処理中...", 0) # Persistent message
        else:
            if self.current_project_path:
                self.statusBar().showMessage(f"プロジェクト '{os.path.basename(self.current_project_path)}' がロードされました。", 5000)
            else:
                 self.statusBar().clearMessage()
        
        self._update_ui_for_project_state() # Ensures UI state is consistent after loading state change


    def _update_ui_for_project_state(self):
        project_loaded_and_not_processing = self.current_project_path is not None and not self._project_loading_in_progress
        
        self.drawing_editor_widget.set_enabled_controls(project_loaded_and_not_processing)
        self.properties_widget.set_enabled_controls(project_loaded_and_not_processing)
        self.glyph_grid_widget.set_search_enabled(project_loaded_and_not_processing)

        # Edit menu actions
        if self.batch_adv_width_action: self.batch_adv_width_action.setEnabled(project_loaded_and_not_processing)
        if self.batch_import_glyphs_action: self.batch_import_glyphs_action.setEnabled(project_loaded_and_not_processing)
        if self.batch_import_reference_images_action: self.batch_import_reference_images_action.setEnabled(project_loaded_and_not_processing)

        # Kanji Viewer panel and its controls
        kv_data_ok = self._kanji_viewer_data_loaded_successfully and bool(self.kanji_viewer_font_combo and self.kanji_viewer_font_combo.count() > 0)
        kv_panel_enabled = kv_data_ok and not self._project_loading_in_progress # KV panel only active if data and fonts are OK, and not loading

        if self.kanji_viewer_panel_widget:
            self.kanji_viewer_panel_widget.setEnabled(kv_panel_enabled)
        
        if self.bookmark_font_button: # Bookmark button also depends on KV panel general state
             self.bookmark_font_button.setEnabled(kv_panel_enabled and self.kanji_viewer_font_combo.count() > 0)

        kv_buttons_enabled_base = kv_panel_enabled # Base enabled state for KV mode buttons

        if self.kanji_viewer_font_combo:
            self.kanji_viewer_font_combo.setEnabled(kv_panel_enabled) # Font combo tied to KV panel state

        if self.kv_mode_button_group:
            buttons = self.kv_mode_button_group.buttons() # Enable/disable all mode buttons
            for btn in buttons: 
                 btn.setEnabled(kv_buttons_enabled_base) 
        
        if project_loaded_and_not_processing:
            self.setWindowTitle(f"P-Glyph - {os.path.basename(self.current_project_path)}")
            # Ensure KV display mode button is correctly checked
            if kv_buttons_enabled_base and self.kv_mode_button_group: # Only if KV panel is active
                current_checked_button_id = self.kv_mode_button_group.checkedId()
                current_mode_to_set = self.kv_display_mode # Mode loaded from settings or default
                if current_checked_button_id == -1 or current_checked_button_id != current_mode_to_set:
                    button_to_set = self.kv_mode_button_group.button(current_mode_to_set)
                    if button_to_set:
                        self.kv_mode_button_group.blockSignals(True)
                        button_to_set.setChecked(True)
                        self.kv_mode_button_group.blockSignals(False)
                    elif self.kv_mode_written_button: # Fallback if desired mode button not found
                        self.kv_mode_button_group.blockSignals(True)
                        self.kv_mode_written_button.setChecked(True)
                        self.kv_mode_button_group.blockSignals(False)
        else: # No project loaded or currently processing
            self.setWindowTitle("P-Glyph")
            if not self._project_loading_in_progress: # If truly no project, reset UI elements
                # Reset data-dependent UI parts
                self.drawing_editor_widget.set_rotated_vrt2_chars(set()) # Clear rotated chars in editor
                self.glyph_grid_widget.set_non_rotated_vrt2_chars(set())
                self.glyph_grid_widget.clear_grid() # Use clear_grid for safety
                self._all_standard_glyphs_pixmap_cache.clear() # Clear MainWindow's internal caches
                self._all_nrvg_pixmap_cache.clear()
                self.glyph_grid_widget.set_filter_caches([],[]) # And GlyphGridWidget's filter caches

                # Reset drawing editor to a blank state
                self.drawing_editor_widget.canvas.reference_image = None
                self.drawing_editor_widget.canvas.set_reference_image_opacity(DEFAULT_REFERENCE_IMAGE_OPACITY)
                self.drawing_editor_widget.canvas.load_glyph("", None, None, DEFAULT_ADVANCE_WIDTH, is_vrt2=False) # Load blank
                # Reset properties widget
                self.properties_widget.load_character_set("")
                self.properties_widget.load_r_vrt2_set("")
                self.properties_widget.load_nr_vrt2_set("")
                self.properties_widget.load_font_name(DEFAULT_FONT_NAME)
                self.properties_widget.load_font_weight(DEFAULT_FONT_WEIGHT)
                # Reset glyph grid selection
                self.glyph_grid_widget.set_active_glyph(None)
                self.drawing_editor_widget.update_unicode_display(None)
                # Reset project-specific caches
                self.project_glyph_chars_cache.clear()
                self.non_rotated_vrt2_chars.clear()

                # Reset KV display mode to default if no project
                if self.kv_mode_button_group and self.kv_mode_written_button: # Ensure button group exists
                    self.kv_mode_button_group.blockSignals(True)
                    self.kv_mode_written_button.setChecked(True) # Default to written glyphs
                    self.kv_mode_button_group.blockSignals(False)
                self.kv_display_mode = MainWindow.KV_MODE_WRITTEN_GLYPHS

                # Reset KV main display label
                if self.kanji_viewer_display_label:
                    default_kv_font = ""
                    if self.kanji_viewer_font_combo and self.kanji_viewer_font_combo.count() > 0:
                        default_kv_font_display_name = self.kanji_viewer_font_combo.itemText(0)
                        if default_kv_font_display_name: # Check if itemText returns valid string
                             default_kv_font = self._get_actual_font_name(default_kv_font_display_name)
                    
                    self._kv_set_label_font_and_text(self.kanji_viewer_display_label, 
                                                    self._kv_initial_char_to_display, 
                                                    default_kv_font,
                                                    self.kanji_viewer_display_label.rect(), 0.85)

                if self.kanji_viewer_related_tabs:
                    self.kanji_viewer_related_tabs.clear()
        
        self._update_bookmark_button_state() # Always update bookmark button state

    def _load_font_settings_txt(self): # For new project defaults
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        settings_file_path = os.path.join(script_dir, FONT_SETTINGS_FILENAME)
        char_string = DEFAULT_CHAR_SET # Default if file ops fail
        try:
            if os.path.exists(settings_file_path):
                with open(settings_file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip() # Remove leading/trailing whitespace
                if content: char_string = content
            else: # File doesn't exist, try to create it with default content
                try:
                    with open(settings_file_path, 'w', encoding='utf-8') as f:
                        f.write(DEFAULT_CHAR_SET)
                    # char_string remains DEFAULT_CHAR_SET as it's just been written
                except IOError as e:
                    # Failed to write, log or ignore, use hardcoded default
                    pass

        except IOError as e:
            # Failed to read, log or ignore, use hardcoded default
            pass

        # Process the character string (remove duplicates, ensure single chars, sort)
        seen = set(); unique_ordered_chars = []
        for c in char_string:
            if len(c) == 1 and c not in seen : unique_ordered_chars.append(c); seen.add(c)
        return sorted(unique_ordered_chars, key=ord)


    def _load_vrt2_settings_txt(self, filename: str, default_set: str):
        script_dir = os.path.dirname(os.path.abspath(__file__)) 
        settings_file_path = os.path.join(script_dir, filename)
        char_string = default_set
        try:
            if os.path.exists(settings_file_path):
                with open(settings_file_path, 'r', encoding='utf-8') as f: content = f.read().strip()
                if content: char_string = content
            else:
                try:
                    with open(settings_file_path, 'w', encoding='utf-8') as f: f.write(default_set)
                    # char_string remains default_set
                except IOError as e: 
                    pass

        except IOError as e: 
            pass

        # Process
        seen = set(); unique_ordered_chars = []
        for c in char_string:
            if len(c) == 1 and c not in seen: unique_ordered_chars.append(c); seen.add(c)
        return sorted(unique_ordered_chars, key=ord)


    def new_project(self):
        if self._project_loading_in_progress:
            QMessageBox.information(self, "処理中", "プロジェクトの読み込みまたは作成処理が進行中です。")
            return

        filepath, _ = QFileDialog.getSaveFileName(self, "新規プロジェクトを作成", "", "Font Project Files (*.fontproj)")
        if filepath:
            if not filepath.endswith(".fontproj"): filepath += ".fontproj"
            
            self._set_project_loading_state(True) # Start loading state

            initial_chars = self._load_font_settings_txt()
            r_vrt2_chars = self._load_vrt2_settings_txt(R_VERT_FILENAME, DEFAULT_R_VERT_CHARS)
            nr_vrt2_chars = self._load_vrt2_settings_txt(VERT_FILENAME, DEFAULT_VERT_CHARS)

            if not initial_chars and DEFAULT_CHAR_SET: # Fallback for initial_chars if file was empty/unreadable
                initial_chars = sorted(list(set(c for c in DEFAULT_CHAR_SET if len(c) == 1)), key=ord)

            create_worker = CreateProjectWorker(filepath, initial_chars, r_vrt2_chars, nr_vrt2_chars)
            create_worker.signals.project_created.connect(self._on_project_created_and_start_load)
            create_worker.signals.error.connect(self._on_project_create_error)
            create_worker.signals.finished.connect(self._check_and_finalize_loading_state_after_create) 
            self.thread_pool.start(create_worker)

    @Slot()
    def _check_and_finalize_loading_state_after_create(self):
        # This slot is connected to CreateProjectWorker.finished.
        # If creation failed (_on_project_create_error was called), _project_loading_in_progress is already False.
        # If creation succeeded (_on_project_created_and_start_load was called), LoadProjectWorker is running,
        # and _project_loading_in_progress is True. LoadProjectWorker.finished will call _check_and_finalize_loading_state.
        # This primarily ensures that if no LoadProjectWorker was started (e.g. if project_created wasn't emitted for some reason
        # but finished was), we reset the loading state.
        # The actual turn-off of loading state is handled by LoadProjectWorker's finished signal or _on_project_create_error.
        pass 


    @Slot(str)
    def _on_project_created_and_start_load(self, filepath: str):
        # Project DB created successfully, now load it.
        # _project_loading_in_progress is already True.
        self.current_project_path = filepath
        self.db_manager.connect_db(filepath) # Connect main DB manager instance
        
        # Clear caches and grid before loading new project data
        self._all_standard_glyphs_pixmap_cache.clear()
        self._all_nrvg_pixmap_cache.clear()
        self.glyph_grid_widget.clear_grid()
        
        load_worker = LoadProjectWorker(self.current_project_path)
        load_worker.signals.basic_info_loaded.connect(self._on_project_basic_info_loaded)
        load_worker.signals.glyph_batch_loaded.connect(self._on_glyph_batch_loaded)
        load_worker.signals.load_progress.connect(self._on_load_progress)
        load_worker.signals.error.connect(self._on_project_load_error)
        load_worker.signals.finished.connect(self._check_and_finalize_loading_state) # This will set loading to False
        self.thread_pool.start(load_worker)

    @Slot(str)
    def _on_project_create_error(self, error_message: str):
        QMessageBox.critical(self, "プロジェクト作成エラー", error_message)
        self.current_project_path = None # Reset path
        self.project_glyph_chars_cache.clear()
        self.non_rotated_vrt2_chars.clear()
        self._set_project_loading_state(False) # End loading state on error

    def open_project(self):
        if self._project_loading_in_progress:
            QMessageBox.information(self, "処理中", "プロジェクトの読み込みまたは作成処理が進行中です。")
            return

        filepath, _ = QFileDialog.getOpenFileName(self, "プロジェクトを開く", "", "Font Project Files (*.fontproj)")
        if filepath:
            self._set_project_loading_state(True) # Start loading state
            self.current_project_path = filepath
            self.db_manager.connect_db(filepath) # Connect main DB manager
            
            # Clear existing data before loading
            self._all_standard_glyphs_pixmap_cache.clear()
            self._all_nrvg_pixmap_cache.clear()
            self.glyph_grid_widget.clear_grid()

            worker = LoadProjectWorker(self.current_project_path)
            worker.signals.basic_info_loaded.connect(self._on_project_basic_info_loaded)
            worker.signals.glyph_batch_loaded.connect(self._on_glyph_batch_loaded)
            worker.signals.load_progress.connect(self._on_load_progress)
            worker.signals.error.connect(self._on_project_load_error)
            worker.signals.finished.connect(self._check_and_finalize_loading_state) # Sets loading to False
            self.thread_pool.start(worker)

    @Slot()
    def _check_and_finalize_loading_state(self):
        # This is called when LoadProjectWorker finishes (success or error handled by its own signals).
        if self._project_loading_in_progress: # Check if we were indeed in a loading state
            self._set_project_loading_state(False)
            # After all data is loaded (or load failed and error handled), select initial glyph.
            self._select_initial_glyph_after_full_load()
            # Update status bar only if loading was successful (error would have own message)
            if self.current_project_path: # Check if project path is still valid (not cleared by error)
                self.statusBar().showMessage(f"プロジェクト '{os.path.basename(self.current_project_path)}' の読み込み完了。", 5000)


    @Slot(dict)
    def _on_project_basic_info_loaded(self, basic_data: dict):
        if not self.current_project_path: return # Should not happen if loading started correctly

        self.properties_widget.load_character_set("".join(basic_data['char_set_list']))
        self.properties_widget.load_r_vrt2_set("".join(basic_data['r_vrt2_list']))
        self.drawing_editor_widget.set_rotated_vrt2_chars(set(basic_data['r_vrt2_list']))
        
        self.properties_widget.load_nr_vrt2_set("".join(basic_data['nr_vrt2_list']))
        self.non_rotated_vrt2_chars = set(basic_data['nr_vrt2_list'])
        self.glyph_grid_widget.set_non_rotated_vrt2_chars(self.non_rotated_vrt2_chars)

        self.properties_widget.load_font_name(basic_data['font_name'])
        self.properties_widget.load_font_weight(basic_data['font_weight'])
        self.drawing_editor_widget.apply_gui_settings(basic_data['gui_settings'])

        # Kanji Viewer Font and Mode settings
        if self.kanji_viewer_font_combo:
            self._populate_kv_fonts() # Repopulate based on system, then try to select saved one
            kv_font_actual_name = basic_data['kv_font_actual_name']
            if kv_font_actual_name:
                target_display_name_bookmarked = f"★ {kv_font_actual_name}"
                target_display_name_plain = kv_font_actual_name
                found_idx = self.kanji_viewer_font_combo.findText(target_display_name_bookmarked)
                if found_idx == -1: found_idx = self.kanji_viewer_font_combo.findText(target_display_name_plain)
                
                if found_idx != -1:
                    self.kanji_viewer_font_combo.blockSignals(True)
                    self.kanji_viewer_font_combo.setCurrentIndex(found_idx)
                    self.kanji_viewer_font_combo.blockSignals(False)
                elif self.kanji_viewer_font_combo.count() > 0: # Font not found, select first and save if different
                    if self.kanji_viewer_font_combo.currentIndex() == -1:
                        self.kanji_viewer_font_combo.setCurrentIndex(0)
                    new_default_font_display = self.kanji_viewer_font_combo.currentText()
                    if new_default_font_display:
                        new_default_font_actual = self._get_actual_font_name(new_default_font_display)
                        if kv_font_actual_name != new_default_font_actual: # Avoid saving if it's already the new default
                            self.save_gui_setting_async(SETTING_KV_CURRENT_FONT, new_default_font_actual)
            elif self.kanji_viewer_font_combo.count() > 0 : # No saved font, but combo has items
                if self.kanji_viewer_font_combo.currentIndex() == -1:
                     self.kanji_viewer_font_combo.setCurrentIndex(0)
                current_font_display = self.kanji_viewer_font_combo.currentText()
                if current_font_display:
                    current_font_actual = self._get_actual_font_name(current_font_display)
                    self.save_gui_setting_async(SETTING_KV_CURRENT_FONT, current_font_actual)

        self.kv_display_mode = basic_data['kv_display_mode_val'] 
        if self.kv_mode_button_group:
            button_to_check = self.kv_mode_button_group.button(self.kv_display_mode)
            if button_to_check:
                self.kv_mode_button_group.blockSignals(True) # Avoid triggering mode change logic
                button_to_check.setChecked(True)
                self.kv_mode_button_group.blockSignals(False)
            elif self.kv_mode_written_button: # Fallback to default if specific button ID not found
                 self.kv_mode_button_group.blockSignals(True)
                 self.kv_mode_written_button.setChecked(True)
                 self.kv_mode_button_group.blockSignals(False)
        
        self._last_active_glyph_char_from_load = basic_data.get('last_active_glyph_char')


    @Slot(list, bool, bool)
    def _on_glyph_batch_loaded(self, glyph_batch: List[Tuple[str, Optional[QPixmap]]], 
                               is_vrt2_batch: bool, is_last_batch_for_type: bool):
        if not self.current_project_path: return # Project loading might have been cancelled

        # Accumulate into MainWindow's caches
        if not is_vrt2_batch:
            self._all_standard_glyphs_pixmap_cache.extend(glyph_batch)
            self.project_glyph_chars_cache.update(g[0] for g in glyph_batch)
            if is_last_batch_for_type: # All standard glyphs are now in MainWindow cache
                # Pass copies to GlyphGridWidget's filter cache
                self.glyph_grid_widget.set_filter_caches(
                    list(self._all_standard_glyphs_pixmap_cache), 
                    list(self._all_nrvg_pixmap_cache) # May be empty if NRVGs not loaded yet
                )
        else:
            self._all_nrvg_pixmap_cache.extend(glyph_batch)
            if is_last_batch_for_type: # All NRVGs are now in MainWindow cache
                 self.glyph_grid_widget.set_filter_caches( # Update with complete NRVG list
                    list(self._all_standard_glyphs_pixmap_cache), 
                    list(self._all_nrvg_pixmap_cache)
                )

        # Add this batch to the grid UI
        self.glyph_grid_widget.add_glyph_batch(glyph_batch, is_vrt2_batch, is_last_batch_for_type)
        
        # Initial glyph selection is now handled by _select_initial_glyph_after_full_load
        # called from _check_and_finalize_loading_state.


    def _select_initial_glyph_after_full_load(self):
        if not self.current_project_path or self._project_loading_in_progress: return

        char_to_load_initially: Optional[str] = None
        load_as_vrt2_initially = False

        if hasattr(self, '_last_active_glyph_char_from_load') and self._last_active_glyph_char_from_load:
            last_char = self._last_active_glyph_char_from_load
            # Check if it's a standard glyph
            if any(g[0] == last_char for g in self._all_standard_glyphs_pixmap_cache):
                 char_to_load_initially = last_char
                 load_as_vrt2_initially = False 
            # Check if it's a non-rotated VRT2 glyph (if not found as standard)
            elif any(g[0] == last_char for g in self._all_nrvg_pixmap_cache):
                 char_to_load_initially = last_char
                 load_as_vrt2_initially = True # This implies it's a special VRT2
            
        if not char_to_load_initially: # Fallback to first navigable glyph if last active not found/valid
            nav_info = self.glyph_grid_widget.get_navigable_glyphs_info()
            if nav_info:
                char_to_load_initially, load_as_vrt2_initially = nav_info[0]
        
        # Ensure VRT2 controls are in a sensible default state before loading a glyph
        self.drawing_editor_widget.update_vrt2_controls(False, False)

        if char_to_load_initially:
            self.load_glyph_for_editing(char_to_load_initially, is_vrt2_edit_mode=load_as_vrt2_initially)
        else: # No glyphs in project, load blank editor
            adv_width = DEFAULT_ADVANCE_WIDTH 
            self.drawing_editor_widget.canvas.load_glyph("", None, None, adv_width, is_vrt2=False)
            self.glyph_grid_widget.set_active_glyph(None)
            self.drawing_editor_widget.update_unicode_display(None)
            self.drawing_editor_widget._update_adv_width_ui_no_signal(adv_width)
        
        self._update_bookmark_button_state()


    @Slot(int, str)
    def _on_load_progress(self, progress: int, message: str):
        if self._project_loading_in_progress: # Only show progress if still in loading state
            self.statusBar().showMessage(f"{message} ({progress}%)", 0) # Persistent


    @Slot(str)
    def _on_project_load_error(self, error_message: str):
        QMessageBox.critical(self, "プロジェクト読み込みエラー", f"プロジェクトの読み込みに失敗しました:\n{error_message}")
        self.current_project_path = None
        self.db_manager.db_path = None 
        self.project_glyph_chars_cache.clear()
        self._all_standard_glyphs_pixmap_cache.clear()
        self._all_nrvg_pixmap_cache.clear()
        self.non_rotated_vrt2_chars.clear()
        # _set_project_loading_state(False) will be called by _check_and_finalize_loading_state
        # which is connected to LoadProjectWorker.finished
        self.statusBar().showMessage("プロジェクトの読み込みに失敗しました。", 5000)


    def _save_current_advance_width_sync(self, character: str, advance_width: int):
        """Synchronously saves advance width. Use sparingly (e.g., before switching glyphs)."""
        if not self.current_project_path or not character:
            return
        try:
            # Direct call to DB manager, not async worker
            self.db_manager.save_glyph_advance_width(character, advance_width)
        except Exception as e:
            # Log or handle error if needed, but this is sync so less critical for UI
            pass



    @Slot(str) 
    @Slot(str, bool) 
    def load_glyph_for_editing(self, character: str, is_vrt2_edit_mode: bool = False):
        if self._project_loading_in_progress: return

        # Save advance width of previously edited glyph (if any, and if it was standard glyph)
        current_canvas_char = self.drawing_editor_widget.canvas.current_glyph_character
        current_canvas_adv_width = self.drawing_editor_widget.adv_width_spinbox.value() # Get from UI

        if current_canvas_char: # If a glyph was loaded
            # Only save if switching to a different character OR different VRT2 mode for same char
            if current_canvas_char != character or \
               (current_canvas_char == character and is_vrt2_edit_mode != self.drawing_editor_widget.canvas.editing_vrt2_glyph):
                 # Only save advance width for standard glyphs, VRT2 doesn't have its own advance in `glyphs` table
                 if not self.drawing_editor_widget.canvas.editing_vrt2_glyph:
                     self._save_current_advance_width_sync(current_canvas_char, current_canvas_adv_width)
        
        # Trigger Kanji Viewer update (deferred)
        if self._kanji_viewer_data_loaded_successfully and character and len(character) == 1:
            with QMutexLocker(self._worker_management_mutex):
                self._kv_char_to_update = character 
            self._kv_deferred_update_timer.start(self._kv_update_delay_ms) 
        elif self.kanji_viewer_display_label : # No valid char for KV or KV data not loaded
            self.kanji_viewer_display_label.setText(character if character else "") # Show char or blank
            self.kanji_viewer_display_label.setFont(QFont()) # Reset font
            if self.kanji_viewer_related_tabs: self.kanji_viewer_related_tabs.clear()
            if self._kv_deferred_update_timer.isActive(): self._kv_deferred_update_timer.stop()
            with QMutexLocker(self._worker_management_mutex): self._kv_char_to_update = None


        if not self.current_project_path or not character: # No project or no char, load blank
            adv_width = DEFAULT_ADVANCE_WIDTH
            self.drawing_editor_widget.canvas.load_glyph("", None, None, adv_width, is_vrt2=False)
            self.drawing_editor_widget.set_enabled_controls(False)
            self.glyph_grid_widget.set_active_glyph(None)
            self.drawing_editor_widget.update_unicode_display(None)
            self.drawing_editor_widget._update_adv_width_ui_no_signal(adv_width)
            self.drawing_editor_widget.update_vrt2_controls(False, False) # Hide VRT2 controls
            return

        # Load glyph data from DB
        adv_width = self.db_manager.load_glyph_advance_width(character)
        pixmap = self.db_manager.load_glyph_image(character, is_vrt2=is_vrt2_edit_mode)
        
        reference_pixmap = None
        if is_vrt2_edit_mode:
            reference_pixmap = self.db_manager.load_vrt2_glyph_reference_image(character)
        else:
            reference_pixmap = self.db_manager.load_reference_image(character)
        
        self.drawing_editor_widget.canvas.load_glyph(character, pixmap, reference_pixmap, adv_width, is_vrt2=is_vrt2_edit_mode)
        self.drawing_editor_widget.set_enabled_controls(True)
        self.glyph_grid_widget.set_active_glyph(character, is_vrt2_source=is_vrt2_edit_mode)
        self.drawing_editor_widget.update_unicode_display(character)
        self.drawing_editor_widget._update_adv_width_ui_no_signal(adv_width)

        # Update VRT2-specific controls in DrawingEditorWidget
        is_char_in_nr_vrt2_set = character in self.non_rotated_vrt2_chars
        self.drawing_editor_widget.update_vrt2_controls(is_char_in_nr_vrt2_set, is_editing_vrt2=is_vrt2_edit_mode)
        
        # Explicitly re-enable/disable VRT2 buttons based on current char and editor state
        editor_is_generally_enabled = self.drawing_editor_widget.pen_button.isEnabled()
        self.drawing_editor_widget.vrt2_toggle_button.setEnabled(editor_is_generally_enabled and is_char_in_nr_vrt2_set)
        self.drawing_editor_widget.transfer_to_vrt2_button.setEnabled(editor_is_generally_enabled and is_char_in_nr_vrt2_set)
        
        # Update reference image button states
        self.drawing_editor_widget.delete_ref_button.setEnabled(editor_is_generally_enabled and reference_pixmap is not None)
        self.drawing_editor_widget.load_ref_button.setEnabled(editor_is_generally_enabled)
        
        if hasattr(self.drawing_editor_widget, 'glyph_to_ref_reset_button'):
            current_glyph_has_content_on_canvas = self.drawing_editor_widget.canvas.image and \
                                                  not self.drawing_editor_widget.canvas.image.isNull() # Basic check
            self.drawing_editor_widget.glyph_to_ref_reset_button.setEnabled(editor_is_generally_enabled and current_glyph_has_content_on_canvas)

        # Save last active standard glyph
        if character and not is_vrt2_edit_mode and not self._project_loading_in_progress: # Don't save during project load
            self.save_gui_setting_async(SETTING_LAST_ACTIVE_GLYPH, character)

        self.drawing_editor_widget.canvas.setFocus() # Focus canvas for keyboard shortcuts


    @Slot(str, QPixmap, bool)
    def handle_glyph_modification_from_canvas(self, character: str, pixmap: QPixmap, is_vrt2: bool):
        if not self.current_project_path or self._project_loading_in_progress: return
        worker = SaveGlyphWorker(self.current_project_path, character, pixmap, is_vrt2_glyph=is_vrt2)
        worker.signals.result.connect(self.on_glyph_save_success)
        worker.signals.error.connect(self.on_glyph_save_error)
        self.thread_pool.start(worker)
        
        # Update "glyph to ref & reset" button state
        if hasattr(self.drawing_editor_widget, 'glyph_to_ref_reset_button'):
            has_content = pixmap and not pixmap.isNull() # Check if pixmap has content
            can_enable_button = self.drawing_editor_widget.pen_button.isEnabled() and has_content
            self.drawing_editor_widget.glyph_to_ref_reset_button.setEnabled(can_enable_button)


    @Slot(str, QPixmap, bool)
    def on_glyph_save_success(self, character: str, saved_pixmap: QPixmap, is_vrt2_glyph: bool):
        if self._project_loading_in_progress: return # Don't update grid if project is still loading/changing
        if is_vrt2_glyph: # Saved a special VRT2 glyph
            if self.current_project_path: # Ensure project context is still valid
                self.glyph_grid_widget.update_single_special_vrt2_preview(character, saved_pixmap)
        else: # Saved a standard glyph
            self.glyph_grid_widget.update_glyph_preview(character, saved_pixmap)

    @Slot(str)
    def on_glyph_save_error(self, error_message: str):
        QMessageBox.warning(self, "保存エラー", f"グリフの保存中にエラーが発生しました:\n{error_message}")
        self.statusBar().showMessage(f"グリフ保存エラー: {error_message[:100]}...", 5000)

    @Slot(str, QPixmap, bool) # char, pixmap, is_vrt2
    def save_reference_image_async(self, character: str, pixmap: QPixmap, is_vrt2: bool):
        if self.current_project_path and character and not self._project_loading_in_progress:
            worker = SaveReferenceImageWorker(self.current_project_path, character, pixmap, is_vrt2_glyph=is_vrt2)
            worker.signals.result.connect(self.on_reference_image_save_success)
            worker.signals.error.connect(self.on_reference_image_save_error)
            self.thread_pool.start(worker)

    @Slot(str, bool) # char, is_vrt2
    def handle_delete_reference_image_async(self, character: str, is_vrt2: bool):
        if self.current_project_path and character and not self._project_loading_in_progress:
            worker = SaveReferenceImageWorker(self.current_project_path, character, None, is_vrt2_glyph=is_vrt2) # Pass None for pixmap to delete
            worker.signals.result.connect(self.on_reference_image_save_success)
            worker.signals.error.connect(self.on_reference_image_save_error)
            self.thread_pool.start(worker)

    @Slot(str, QPixmap, bool) # char, saved_pixmap (or None if deleted), is_vrt2_saved
    def on_reference_image_save_success(self, character: str, saved_pixmap: Optional[QPixmap], is_vrt2_saved: bool):
        if self._project_loading_in_progress: return
        # Check if the saved reference image corresponds to the currently edited glyph
        current_canvas_char = self.drawing_editor_widget.canvas.current_glyph_character
        current_canvas_is_vrt2_editing = self.drawing_editor_widget.canvas.editing_vrt2_glyph
        
        if character == current_canvas_char and is_vrt2_saved == current_canvas_is_vrt2_editing:
            self.drawing_editor_widget.canvas.reference_image = saved_pixmap.copy() if saved_pixmap else None
            self.drawing_editor_widget.canvas.update()
            # Update delete button state
            self.drawing_editor_widget.delete_ref_button.setEnabled(
                self.drawing_editor_widget.load_ref_button.isEnabled() and (saved_pixmap is not None)
            )
        status_message = f"下書き画像 '{character}' " + ("保存完了" if saved_pixmap else "削除完了")
        self.statusBar().showMessage(status_message, 3000)


    @Slot(str)
    def on_reference_image_save_error(self, error_message: str):
        QMessageBox.warning(self, "下書き画像保存エラー", f"下書き画像の保存中にエラーが発生しました:\n{error_message}")
        self.statusBar().showMessage(f"下書き画像保存エラー: {error_message[:100]}...", 5000)


    @Slot(str, str)
    def save_gui_setting_async(self, key: str, value: str):
        if self.current_project_path and not self._project_loading_in_progress:
            worker = SaveGuiStateWorker(self.current_project_path, key, value)
            worker.signals.error.connect(self.on_gui_save_error) # Can connect finished for logging if needed
            self.thread_pool.start(worker)
    
    @Slot(str)
    def on_gui_save_error(self, error_message: str): 
        # Optionally display this, or just log it. Could be noisy for frequent settings.
        self.statusBar().showMessage(f"GUI設定保存エラー: {error_message[:100]}...", 3000)

    @Slot(str, int)
    def save_glyph_advance_width_async(self, character: str, advance_width: int):
        if self.current_project_path and character and not self._project_loading_in_progress:
            worker = SaveAdvanceWidthWorker(self.current_project_path, character, advance_width)
            # Can connect error signal if specific error handling for adv width is needed
            worker.signals.error.connect(self.on_gui_save_error) # Reuse for general GUI/DB errors
            self.thread_pool.start(worker)

    def _process_char_set_string(self, char_string: str) -> List[str]:
        # Helper to process text edit content into a sorted list of unique single characters
        seen = set(); unique_ordered_chars = []
        for c in char_string:
            if len(c) == 1 and c not in seen: unique_ordered_chars.append(c); seen.add(c)
        return sorted(unique_ordered_chars, key=ord)


    @Slot(str)
    def update_project_character_set(self, new_char_string: str):
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。"); return
        processed_chars = self._process_char_set_string(new_char_string)
        try:
            # This DB call will add/remove rows in the 'glyphs' table
            self.db_manager.update_project_character_set(processed_chars) 
            
            # After DB update, trigger a full project reload to refresh UI and caches
            self._set_project_loading_state(True) 
            self._all_standard_glyphs_pixmap_cache.clear() # Clear caches before reloading
            self._all_nrvg_pixmap_cache.clear()
            self.glyph_grid_widget.clear_grid()

            worker = LoadProjectWorker(self.current_project_path)
            worker.signals.basic_info_loaded.connect(self._on_project_basic_info_loaded)
            worker.signals.glyph_batch_loaded.connect(self._on_glyph_batch_loaded)
            worker.signals.load_progress.connect(self._on_load_progress)
            worker.signals.error.connect(self._on_project_load_error)
            worker.signals.finished.connect(self._check_and_finalize_loading_state)
            self.thread_pool.start(worker)
            QMessageBox.information(self, "文字セット更新", "プロジェクトの文字セットが更新されました。データ再読み込み中です。")
        except Exception as e: # Catch errors from db_manager call directly
            QMessageBox.critical(self, "文字セット更新エラー", f"文字セットの更新に失敗しました: {e}")
            self._set_project_loading_state(False) # Ensure loading state is reset

    @Slot(str)
    def update_rotated_vrt2_set(self, new_char_string: str):
        if not self.current_project_path or self._project_loading_in_progress: 
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。"); return
        processed_chars = self._process_char_set_string(new_char_string)
        try:
            self.db_manager.update_rotated_vrt2_character_set(processed_chars)
            # This change only affects a setting and how editor displays unicode info.
            # A full reload is safest to ensure all parts of UI are consistent.
            self._set_project_loading_state(True)
            self._all_standard_glyphs_pixmap_cache.clear()
            self._all_nrvg_pixmap_cache.clear()
            self.glyph_grid_widget.clear_grid()

            worker = LoadProjectWorker(self.current_project_path)
            worker.signals.basic_info_loaded.connect(self._on_project_basic_info_loaded)
            worker.signals.glyph_batch_loaded.connect(self._on_glyph_batch_loaded)
            worker.signals.load_progress.connect(self._on_load_progress)
            worker.signals.error.connect(self._on_project_load_error)
            worker.signals.finished.connect(self._check_and_finalize_loading_state)
            self.thread_pool.start(worker)
            QMessageBox.information(self, "縦書き文字セット更新", "回転縦書き文字セットが更新されました。データ再読み込み中です。")
        except Exception as e: 
            QMessageBox.critical(self, "縦書き文字セット更新エラー", f"セットの更新に失敗: {e}")
            self._set_project_loading_state(False)

    @Slot(str)
    def update_non_rotated_vrt2_set(self, new_char_string: str):
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。"); return
        processed_chars = self._process_char_set_string(new_char_string)
        new_nr_vrt2_set = set(processed_chars)
        
        # Handle potential conflicts: If a char is in new NR-VRT2 set, remove from R-VRT2 set
        r_vrt2_set_from_db = set(self.db_manager.get_rotated_vrt2_character_set())
        conflicts_resolved_r_vrt2 = list(r_vrt2_set_from_db - new_nr_vrt2_set) # Chars remaining in R-VRT2

        try:
            self.db_manager.update_non_rotated_vrt2_character_set(list(new_nr_vrt2_set))
            if len(conflicts_resolved_r_vrt2) != len(r_vrt2_set_from_db): # If R-VRT2 set changed
                self.db_manager.update_rotated_vrt2_character_set(conflicts_resolved_r_vrt2)
            
            # This change adds/removes from vrt2_glyphs and potentially glyphs table. Reload needed.
            self._set_project_loading_state(True)
            self._all_standard_glyphs_pixmap_cache.clear()
            self._all_nrvg_pixmap_cache.clear()
            self.glyph_grid_widget.clear_grid()

            worker = LoadProjectWorker(self.current_project_path)
            worker.signals.basic_info_loaded.connect(self._on_project_basic_info_loaded)
            worker.signals.glyph_batch_loaded.connect(self._on_glyph_batch_loaded)
            worker.signals.load_progress.connect(self._on_load_progress)
            worker.signals.error.connect(self._on_project_load_error)
            worker.signals.finished.connect(self._check_and_finalize_loading_state)
            self.thread_pool.start(worker)
            QMessageBox.information(self, "非回転縦書き文字セット更新", "非回転縦書き文字セットが更新されました。データ再読み込み中です。")
        except Exception as e: 
            QMessageBox.critical(self, "非回転縦書き文字セット更新エラー", f"セットの更新に失敗: {e}")
            self._set_project_loading_state(False)

    @Slot(str)
    def update_font_name(self, name: str):
        if not self.current_project_path or self._project_loading_in_progress: return
        self.save_gui_setting_async(SETTING_FONT_NAME, name)
    @Slot(str)
    def update_font_weight(self, weight: str):
        if not self.current_project_path or self._project_loading_in_progress: return
        self.save_gui_setting_async(SETTING_FONT_WEIGHT, weight)

    @Slot(bool)
    def handle_vrt2_edit_mode_toggle(self, is_editing_vrt2: bool):
        if self._project_loading_in_progress: return
        current_char = self.drawing_editor_widget.canvas.current_glyph_character
        if current_char: # Reload the same character but in the new VRT2 mode
            self.load_glyph_for_editing(current_char, is_vrt2_edit_mode=is_editing_vrt2)

    @Slot()
    def handle_transfer_to_vrt2(self):
        current_char = self.drawing_editor_widget.canvas.current_glyph_character
        if not current_char or not self.current_project_path or self._project_loading_in_progress: return

        # Save current advance width if editing standard glyph before transfer
        current_adv_width_from_ui = self.drawing_editor_widget.adv_width_spinbox.value()
        if not self.drawing_editor_widget.canvas.editing_vrt2_glyph: # If currently editing standard
            self._save_current_advance_width_sync(current_char, current_adv_width_from_ui)

        # Disable buttons during transfer to prevent concurrent ops
        self.drawing_editor_widget.transfer_to_vrt2_button.setEnabled(False)
        self.drawing_editor_widget.vrt2_toggle_button.setEnabled(False)
        QApplication.processEvents() # Ensure UI updates

        # Load standard glyph image (source for transfer)
        standard_pixmap = self.db_manager.load_glyph_image(current_char, is_vrt2=False)
        if not standard_pixmap: # If no standard image, transfer a blank one
            standard_pixmap = QPixmap(self.drawing_editor_widget.canvas.image_size)
            standard_pixmap.fill(QColor(Qt.white))
        
        try:
            pixmap_for_worker = standard_pixmap.copy() # Ensure worker gets its own copy
            vrt2_save_worker = SaveGlyphWorker(self.current_project_path, current_char, pixmap_for_worker, is_vrt2_glyph=True)
            vrt2_save_worker.signals.result.connect(self._handle_transfer_to_vrt2_result)
            vrt2_save_worker.signals.error.connect(self._handle_transfer_to_vrt2_error)
            vrt2_save_worker.signals.finished.connect(self._reenable_vrt2_buttons_after_transfer) # Re-enable buttons
            self.thread_pool.start(vrt2_save_worker)
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "転送エラー", f"'{current_char}' の縦書きグリフへのデータ転送開始時にエラーが発生しました: {e}\n{traceback.format_exc()}")
            self._reenable_vrt2_buttons_after_transfer() # Ensure buttons re-enabled on immediate error


    def _handle_transfer_to_vrt2_result(self, character: str, saved_pixmap: QPixmap, is_vrt2_glyph: bool):
        if not is_vrt2_glyph or self._project_loading_in_progress: return # Only care about VRT2 result here
        if not self.current_project_path: return

        self.glyph_grid_widget.update_single_special_vrt2_preview(character, saved_pixmap)
        
        # If user is currently editing this char as VRT2, reload to show transferred image
        if self.drawing_editor_widget.canvas.editing_vrt2_glyph and \
           self.drawing_editor_widget.canvas.current_glyph_character == character:
            self.load_glyph_for_editing(character, is_vrt2_edit_mode=True)
        
        self.statusBar().showMessage(f"'{character}' の縦書きグリフへの転送成功。", 3000)

    def _handle_transfer_to_vrt2_error(self, error_message: str):
        current_char_display = self.drawing_editor_widget.canvas.current_glyph_character or "選択中の文字"
        QMessageBox.critical(self, "転送エラー", f"'{current_char_display}' の縦書きグリフへのデータ転送中にエラーが発生しました: {error_message}")

    def _reenable_vrt2_buttons_after_transfer(self):
        is_editor_generally_enabled = self.drawing_editor_widget.pen_button.isEnabled()
        char_for_vrt2_check = self.drawing_editor_widget.canvas.current_glyph_character
        is_char_in_nr_set_and_valid = (self.current_project_path is not None and \
                                       char_for_vrt2_check and \
                                       char_for_vrt2_check in self.non_rotated_vrt2_chars)
        
        if hasattr(self.drawing_editor_widget, 'transfer_to_vrt2_button'): 
            self.drawing_editor_widget.transfer_to_vrt2_button.setEnabled(is_editor_generally_enabled and is_char_in_nr_set_and_valid)
        if hasattr(self.drawing_editor_widget, 'vrt2_toggle_button'):
            self.drawing_editor_widget.vrt2_toggle_button.setEnabled(is_editor_generally_enabled and is_char_in_nr_set_and_valid)
        
        # Clear "transferring..." message if no specific success/error message was shown for this operation
        current_status_msg = self.statusBar().currentMessage()
        char_display_name = char_for_vrt2_check or "操作"
        if not (current_status_msg.endswith("転送成功。") or \
                "転送中にエラーが発生しました" in current_status_msg or \
                "データ転送開始時にエラーが発生しました" in current_status_msg):
            self.statusBar().showMessage(f"'{char_display_name}' の縦書きグリフ転送処理完了。", 3000)


    @Slot(bool) # is_vrt2_target (where the reference image will be saved)
    def handle_glyph_to_reference_and_reset(self, is_vrt2_target: bool):
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。")
            return

        canvas = self.drawing_editor_widget.canvas
        current_char = canvas.current_glyph_character

        if not current_char:
            QMessageBox.information(self, "操作不可", "グリフが選択されていません。")
            return
        
        # Ensure the target type (is_vrt2_target) matches the currently edited type on canvas
        if canvas.editing_vrt2_glyph != is_vrt2_target:
            QMessageBox.critical(self, "内部エラー", "グリフを下書きへ転送操作でモード不一致。")
            # This state should ideally not be reachable if UI enables button correctly.
            return

        current_glyph_pixmap = canvas.get_current_image()
        if current_glyph_pixmap.isNull(): # Or check if it's all white
            QMessageBox.information(self, "情報", "グリフに描画内容がありません。")
            return

        # Disable button during operation
        self.drawing_editor_widget.glyph_to_ref_reset_button.setEnabled(False)
        QApplication.processEvents()

        # 1. Save current glyph image as reference image for this char (standard or VRT2)
        ref_worker = SaveReferenceImageWorker(self.current_project_path, current_char, current_glyph_pixmap.copy(), is_vrt2_glyph=is_vrt2_target)
        ref_worker.signals.result.connect(self._on_glyph_to_ref_transfer_ref_save_success)
        ref_worker.signals.error.connect(lambda err: self._on_glyph_to_ref_transfer_error("下書き保存", err))
        ref_worker.signals.finished.connect(self._check_glyph_to_ref_reset_completion)
        self.thread_pool.start(ref_worker)

        # 2. Reset the current glyph image on canvas to blank white
        blank_pixmap = QPixmap(canvas.image_size)
        blank_pixmap.fill(QColor(Qt.white))
        
        canvas.image = blank_pixmap.copy() # Update canvas image directly
        canvas._save_state_to_undo_stack() # Add blank state to undo stack
        canvas.update() # Redraw canvas

        # 3. Save this new blank white image to the DB for the current glyph (standard or VRT2)
        glyph_worker = SaveGlyphWorker(self.current_project_path, current_char, blank_pixmap.copy(), is_vrt2_glyph=is_vrt2_target)
        glyph_worker.signals.result.connect(self._on_glyph_to_ref_transfer_glyph_reset_success)
        glyph_worker.signals.error.connect(lambda err: self._on_glyph_to_ref_transfer_error("グリフ白紙化", err))
        glyph_worker.signals.finished.connect(self._check_glyph_to_ref_reset_completion)
        self.thread_pool.start(glyph_worker)
        
        self._glyph_to_ref_reset_op_count = 2 # Track two async operations

    _glyph_to_ref_reset_op_count = 0 # Counter for async operations in glyph_to_ref_reset

    def _on_glyph_to_ref_transfer_ref_save_success(self, character: str, saved_ref_pixmap: Optional[QPixmap], is_vrt2_ref_saved: bool):
        if self._project_loading_in_progress: return
        # Update canvas's reference_image if it's the currently edited one
        canvas = self.drawing_editor_widget.canvas
        if canvas.current_glyph_character == character and canvas.editing_vrt2_glyph == is_vrt2_ref_saved:
            canvas.reference_image = saved_ref_pixmap.copy() if saved_ref_pixmap else None
            canvas.update()
            self.drawing_editor_widget.delete_ref_button.setEnabled(saved_ref_pixmap is not None)


    def _on_glyph_to_ref_transfer_glyph_reset_success(self, character: str, reset_glyph_pixmap: QPixmap, is_vrt2_glyph_reset: bool):
        if self._project_loading_in_progress: return
        # Update grid preview for the reset glyph
        if is_vrt2_glyph_reset:
            self.glyph_grid_widget.update_single_special_vrt2_preview(character, reset_glyph_pixmap)
        else:
            self.glyph_grid_widget.update_glyph_preview(character, reset_glyph_pixmap)

        # Button state updated in _check_glyph_to_ref_reset_completion
        # Since glyph is now blank, the "glyph to ref & reset" button should be disabled
        if hasattr(self.drawing_editor_widget, 'glyph_to_ref_reset_button'):
             self.drawing_editor_widget.glyph_to_ref_reset_button.setEnabled(False) # Glyph is now blank


    def _on_glyph_to_ref_transfer_error(self, operation_name: str, error_message: str):
        QMessageBox.warning(self, f"{operation_name}エラー", f"{operation_name}中にエラーが発生しました:\n{error_message}")
        # Button re-enablement handled by _check_glyph_to_ref_reset_completion if it's the last op

    def _check_glyph_to_ref_reset_completion(self):
        self._glyph_to_ref_reset_op_count -= 1
        if self._glyph_to_ref_reset_op_count == 0:
            self.statusBar().showMessage("グリフを下書きへ転送しリセットしました。", 3000)
            # Re-enable button if editor active, but content is now blank, so it should remain disabled.
            # The _on_glyph_to_ref_transfer_glyph_reset_success already disables it.
            if hasattr(self.drawing_editor_widget, 'glyph_to_ref_reset_button'):
                is_editor_enabled = self.drawing_editor_widget.pen_button.isEnabled()
                # Even if editor is enabled, content is blank, so keep button disabled.
                self.drawing_editor_widget.glyph_to_ref_reset_button.setEnabled(is_editor_enabled and False)


    @Slot()
    def handle_export_font(self):
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。")
            return

        if self.export_process and self.export_process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "情報", "フォント書き出し処理が既に実行中です。")
            return

        self.original_export_button_state = self.properties_widget.export_font_button.isEnabled()
        self.properties_widget.export_font_button.setEnabled(False)
        self.statusBar().showMessage("フォント書き出し中...", 0) # Persistent message
        QApplication.processEvents() # Ensure UI updates

        try:
            self.export_process = QProcess(self)
            self.export_process.setProcessChannelMode(QProcess.MergedChannels)

            # Try to find DB2OTF.py relative to main.py
            script_dir = Path(sys.argv[0]).resolve().parent
            db2otf_script_path = script_dir / "DB2OTF.py"

            if not db2otf_script_path.exists():
                QMessageBox.critical(self, "エラー", f"スクリプト {db2otf_script_path} が見つかりません。")
                self._cleanup_after_export()
                return

            python_executable = sys.executable # Use the same Python interpreter
            arguments = [str(db2otf_script_path), "--db_path", self.current_project_path]
            
            # For debugging, can print the command:
            command_str = f"\"{python_executable}\" \"{arguments[0]}\" \"--db_path\" \"{arguments[2]}\""
            # print(f"Executing: {command_str}")

            self.export_process.finished.connect(self._on_export_process_finished)
            self.export_process.errorOccurred.connect(self._on_export_process_error)
            
            self.export_process.start(python_executable, arguments)

        except Exception as e:
            import traceback
            QMessageBox.critical(self, "書き出しエラー", f"フォント書き出しの開始中に予期せぬエラーが発生しました: {e}")
            self._cleanup_after_export()

    @Slot(int, QProcess.ExitStatus)
    def _on_export_process_finished(self, exitCode: int, exitStatus: QProcess.ExitStatus):
        if not self.export_process: # Should not happen if signal is connected
            return

        output_bytes = self.export_process.readAllStandardOutput() # Or MergedChannels if set
        try:
            # Try decoding with preferred encoding, fallback for robustness
            output = output_bytes.data().decode(sys.stdout.encoding if sys.stdout.encoding else 'utf-8', errors='replace')
        except UnicodeDecodeError: # If primary decode fails
            output = output_bytes.data().decode('latin-1', errors='replace') # Common fallback


        if exitStatus == QProcess.NormalExit and exitCode == 0:
            QMessageBox.information(self, "書き出し完了", "フォントの書き出しが完了しました。")
            self.statusBar().showMessage("フォントの書き出しが完了しました。", 5000)
        else:
            error_reason = "クラッシュしました" if exitStatus == QProcess.CrashExit else f"エラー終了コード: {exitCode}"
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setWindowTitle("書き出しエラー")
            msg_box.setText(f"フォント書き出し中にエラーが発生しました ({error_reason})。")
            msg_box.setInformativeText(f"終了コード: {exitCode}, 終了ステータス: {exitStatus.name}") # Use enum name
            msg_box.setDetailedText(output if output.strip() else "出力はありませんでした。")
            msg_box.exec()
            self.statusBar().showMessage(f"フォント書き出しエラー ({error_reason})", 5000)
        
        self._cleanup_after_export()

    @Slot(QProcess.ProcessError)
    def _on_export_process_error(self, error: QProcess.ProcessError):
        if not self.export_process: # Should not happen
            return

        error_string = self.export_process.errorString()
        QMessageBox.critical(self, "書き出しプロセスエラー", 
                             f"フォント書き出しプロセスの実行に失敗しました。\nエラータイプ: {error.name}\n詳細: {error_string}")
        self.statusBar().showMessage("フォント書き出しプロセス失敗。", 5000)
        
        self._cleanup_after_export()

    def _cleanup_after_export(self):
        # Restore export button state if project is still loaded
        project_still_loaded = self.current_project_path is not None and not self._project_loading_in_progress
        can_be_enabled_after_export = project_still_loaded and self.original_export_button_state
        self.properties_widget.export_font_button.setEnabled(can_be_enabled_after_export)

        if self.statusBar().currentMessage() == "フォント書き出し中...":
            self.statusBar().clearMessage()
        
        if self.export_process:
            # Disconnect signals to prevent multiple calls if cleanup is somehow re-entrant
            try: self.export_process.finished.disconnect(self._on_export_process_finished)
            except RuntimeError: pass # Already disconnected
            try: self.export_process.errorOccurred.disconnect(self._on_export_process_error)
            except RuntimeError: pass

            self.export_process.deleteLater() # Schedule for deletion
            self.export_process = None


    def keyPressEvent(self, event: QKeyEvent):
        if self._project_loading_in_progress:
            event.ignore() # Ignore all key presses during critical load/save
            return

        # Standard Undo/Redo shortcuts
        if event.matches(QKeySequence.StandardKey.Undo):
            if self.drawing_editor_widget.undo_button.isEnabled(): # Check if action is valid
                self.drawing_editor_widget.canvas.undo()
            event.accept(); return
        if event.matches(QKeySequence.StandardKey.Redo):
            if self.drawing_editor_widget.redo_button.isEnabled():
                self.drawing_editor_widget.canvas.redo()
            event.accept(); return

        # Avoid tool shortcuts if focus is on an input widget
        focus_widget = QApplication.focusWidget()
        if isinstance(focus_widget, (QLineEdit, QTextEdit, QSpinBox)):
            super().keyPressEvent(event); return

        key = event.key()
        # Tool shortcuts (B, E, V) - only if a glyph is active
        if not event.isAutoRepeat() and self.drawing_editor_widget.canvas.current_glyph_character:
            if key == Qt.Key_E: self.drawing_editor_widget.eraser_button.click(); event.accept(); return
            if key == Qt.Key_B: self.drawing_editor_widget.pen_button.click(); event.accept(); return
            if key == Qt.Key_V: self.drawing_editor_widget.move_button.click(); event.accept(); return

        # Arrow key navigation for glyphs
        if (key == Qt.Key_Left or key == Qt.Key_Right) and self.current_project_path:
            navigable_glyphs_info = self.glyph_grid_widget.get_navigable_glyphs_info()
            if not navigable_glyphs_info: super().keyPressEvent(event); return

            current_char_in_editor = self.drawing_editor_widget.canvas.current_glyph_character
            current_is_vrt2_mode = self.drawing_editor_widget.canvas.editing_vrt2_glyph
            current_idx = -1

            # Find current glyph in the navigable list
            for i, (char, is_vrt2_nav_item) in enumerate(navigable_glyphs_info):
                if char == current_char_in_editor and is_vrt2_nav_item == current_is_vrt2_mode:
                    current_idx = i; break
            
            if current_idx == -1 and current_char_in_editor: # If exact VRT2 match failed, try char only (e.g. if VRT2 status changed)
                 for i, (char, _) in enumerate(navigable_glyphs_info):
                     if char == current_char_in_editor: # Take first char match
                         current_idx = i; break
            
            target_char_info: Tuple[str, bool]
            if current_idx == -1: # No current selection or not found, go to start/end
                target_char_info = navigable_glyphs_info[0] if key == Qt.Key_Right else navigable_glyphs_info[-1]
            else: # Navigate from current
                if key == Qt.Key_Left: next_idx = (current_idx - 1 + len(navigable_glyphs_info)) % len(navigable_glyphs_info)
                else: next_idx = (current_idx + 1) % len(navigable_glyphs_info)
                target_char_info = navigable_glyphs_info[next_idx]
            
            target_char, load_as_vrt2 = target_char_info
            self.load_glyph_for_editing(target_char, is_vrt2_edit_mode=load_as_vrt2)
            event.accept(); return
            
        super().keyPressEvent(event)


    def open_batch_advance_width_dialog(self):
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。")
            return
        dialog = BatchAdvanceWidthDialog(self)
        if dialog.exec() == QDialog.Accepted:
            char_spec, adv_width = dialog.get_values()
            if char_spec is not None and adv_width is not None:
                self._apply_batch_advance_width(char_spec, adv_width)

    def _parse_char_specification(self, char_spec: str) -> Tuple[Optional[List[str]], Optional[str]]:
        target_chars_set: Set[str] = set()
        original_spec = char_spec # For error messages if needed
        temp_spec = char_spec # Modifiable copy

        # Handle .notdef separately as it's not a standard Unicode char
        if ".notdef" in temp_spec.lower():
            target_chars_set.add(".notdef")
            temp_spec = re.sub(r"\.notdef", "", temp_spec, flags=re.IGNORECASE) # Remove it for further parsing

        # Parse U+XXXX-U+YYYY ranges (case-insensitive for U+)
        # Iterate backwards to handle string modification correctly
        range_matches = list(re.finditer(r"U\+([0-9A-Fa-f]{4,6})-U\+([0-9A-Fa-f]{4,6})", temp_spec, re.IGNORECASE))
        for match in reversed(range_matches): # Process from end to keep indices valid
            try:
                start_hex, end_hex = match.group(1), match.group(2)
                start_ord, end_ord = int(start_hex, 16), int(end_hex, 16)
                if start_ord > end_ord:
                    return None, f"Unicode範囲の開始値が終了値より大きいです: U+{start_hex}-U+{end_hex}"
                for i in range(start_ord, end_ord + 1):
                    target_chars_set.add(chr(i))
                temp_spec = temp_spec[:match.start()] + temp_spec[match.end():] # Remove processed part
            except (ValueError, OverflowError): # chr() can raise OverflowError for too large int
                return None, f"無効なUnicodeコードポイントが範囲内に含まれています: U+{match.group(0)}"


        # Parse individual U+XXXX codes (case-insensitive for U+)
        unicode_matches = list(re.finditer(r"U\+([0-9A-Fa-f]{4,6})", temp_spec, re.IGNORECASE))
        for match in reversed(unicode_matches): # Process from end
            try:
                hex_code = match.group(1)
                target_chars_set.add(chr(int(hex_code, 16)))
                temp_spec = temp_spec[:match.start()] + temp_spec[match.end():]
            except (ValueError, OverflowError):
                return None, f"無効なUnicodeコードポイントが含まれています: U+{match.group(0)}"
        
        # Remaining characters are literals, remove common separators
        cleaned_literals = re.sub(r"[\s,;\t]+", "", temp_spec) # Remove whitespace and common separators
        for char_val in cleaned_literals:
            if len(char_val) == 1 : target_chars_set.add(char_val) # Only add single characters

        if not target_chars_set and original_spec: # If spec was given but nothing parsed
             return None, "有効な文字またはUnicode指定が見つかりませんでした。"

        # Sort: .notdef first, then by Unicode order
        return sorted(list(target_chars_set), key=lambda x: (-1, x) if x == '.notdef' else (ord(x) if len(x)==1 else float('inf'),x )), None


    def _apply_batch_advance_width(self, char_spec: str, new_adv_width: int):
        if not self.current_project_path or self._project_loading_in_progress: return
        
        target_chars_list, error_msg = self._parse_char_specification(char_spec)
        if error_msg:
            QMessageBox.warning(self, "入力エラー", error_msg)
            return
        if not target_chars_list:
            QMessageBox.information(self, "情報", "適用対象の有効な文字が見つかりませんでした。")
            return

        # Ensure project_glyph_chars_cache is populated
        if not self.project_glyph_chars_cache: # Should be populated if project loaded
            # This is a fallback, ideally cache is always up-to-date
            all_glyphs_data_raw = self.db_manager.get_all_glyphs_with_preview_data() 
            self.project_glyph_chars_cache = {char_data[0] for char_data in all_glyphs_data_raw}
            if not self.project_glyph_chars_cache and target_chars_list: # Still no chars after trying to load
                QMessageBox.warning(self, "エラー", "プロジェクトにグリフが読み込まれていないか空です。")
                return

        updated_count = 0; skipped_count = 0
        current_char_in_editor = self.drawing_editor_widget.canvas.current_glyph_character
        
        # Batch update using a single DB connection
        conn = self.db_manager._get_connection() # Get a connection from the manager
        cursor = conn.cursor()
        try:
            for char_to_update in target_chars_list:
                if char_to_update in self.project_glyph_chars_cache: # Check if char is in the project
                    cursor.execute("UPDATE glyphs SET advance_width = ? WHERE character = ?", (new_adv_width, char_to_update))
                    if cursor.rowcount > 0: updated_count += 1
                    # Else: char might be in cache but somehow not in DB or no change needed - less likely
                else:
                    skipped_count += 1
            conn.commit()
        except Exception as e:
            conn.rollback()
            QMessageBox.critical(self, "一括更新エラー", f"データベース更新中にエラーが発生しました: {e}")
            return
        finally:
            conn.close()

        summary_message = f"{updated_count} 文字の送り幅を {new_adv_width} に更新しました。"
        if skipped_count > 0:
            summary_message += f"\n{skipped_count} 文字はプロジェクトに含まれていないためスキップされました。"
        QMessageBox.information(self, "一括更新完了", summary_message)

        # If current editor glyph was updated, refresh its display
        if current_char_in_editor and \
           not self.drawing_editor_widget.canvas.editing_vrt2_glyph and \
           current_char_in_editor in target_chars_list and \
           current_char_in_editor in self.project_glyph_chars_cache: # Check if it was one of the updated ones
            self.drawing_editor_widget.canvas.current_glyph_advance_width = new_adv_width
            self.drawing_editor_widget._update_adv_width_ui_no_signal(new_adv_width) # Update UI without re-emitting
            self.drawing_editor_widget.canvas.update() # Redraw canvas


    def batch_import_glyphs(self):
        self._process_batch_image_import(import_type="glyph")

    def batch_import_reference_images(self):
        self._process_batch_image_import(import_type="reference")

    def _process_batch_image_import(self, import_type: str): # "glyph" or "reference"
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。")
            return

        file_dialog_title = "グリフ画像の一括読み込み" if import_type == "glyph" else "下書き画像の一括読み込み"
        file_paths, _ = QFileDialog.getOpenFileNames(self, file_dialog_title, "", "画像ファイル (*.png *.jpg *.jpeg *.bmp)")
        if not file_paths: return

        target_image_size = QSize(CANVAS_IMAGE_WIDTH, CANVAS_IMAGE_HEIGHT)
        # Ensure caches are populated (fallbacks if not already)
        if not self.project_glyph_chars_cache: # For checking if char exists in standard glyphs
            all_glyphs_data_raw = self.db_manager.get_all_glyphs_with_preview_data()
            self.project_glyph_chars_cache = {char_data[0] for char_data in all_glyphs_data_raw}
        
        if import_type == "reference" and not self.non_rotated_vrt2_chars: # For checking VRT2 ref target
            self.non_rotated_vrt2_chars = set(self.db_manager.get_non_rotated_vrt2_character_set())


        processed_count = 0; skipped_filename_format = 0; skipped_unicode_conversion = 0
        skipped_char_not_in_project = 0; skipped_image_load_error = 0; db_update_errors = 0
        
        total_files = len(file_paths)
        self.statusBar().showMessage(f"{file_dialog_title} を開始します ({total_files} ファイル)...", 0)
        QApplication.processEvents()


        for idx, file_path_str in enumerate(file_paths):
            file_path = Path(file_path_str)
            self.statusBar().showMessage(f"処理中 ({idx+1}/{total_files}): {file_path.name}", 0)
            QApplication.processEvents()

            stem = file_path.stem # Filename without extension
            is_vrt2_ref_candidate = False
            hex_code = "" 
            
            # Try to parse filename: uniXXXXvert.png or U+XXXXvert.png for VRT2 reference images
            if import_type == "reference":
                match_vert = re.fullmatch(r"uni([0-9A-Fa-f]{4,6})vert", stem, re.IGNORECASE)
                if not match_vert:
                     match_vert_uplus = re.fullmatch(r"U\+([0-9A-Fa-f]{4,6})vert", stem, re.IGNORECASE)
                     if match_vert_uplus:
                         hex_code = match_vert_uplus.group(1)
                         is_vrt2_ref_candidate = True
                else:
                    hex_code = match_vert.group(1)
                    is_vrt2_ref_candidate = True

            # If not VRT2 ref, or for glyph import, parse uniXXXX.png or U+XXXX.png
            if not is_vrt2_ref_candidate: # Also covers import_type == "glyph"
                match = re.fullmatch(r"uni([0-9A-Fa-f]{4,6})", stem, re.IGNORECASE)
                if not match: # Try U+XXXX format
                    match_uplus = re.fullmatch(r"U\+([0-9A-Fa-f]{4,6})", stem, re.IGNORECASE)
                    if not match_uplus:
                        skipped_filename_format += 1; continue
                    hex_code = match_uplus.group(1)
                else:
                    hex_code = match.group(1)
            
            if not hex_code: # Should have been caught by previous checks
                skipped_filename_format += 1; continue
            try:
                unicode_val = int(hex_code, 16)
                character = chr(unicode_val)
            except (ValueError, OverflowError):
                skipped_unicode_conversion += 1; continue

            # Check if character is valid for the import type
            target_is_nrvg_for_ref = False # Flag if this ref image is for a non-rotated VRT2 glyph
            if import_type == "reference" and is_vrt2_ref_candidate:
                if character not in self.non_rotated_vrt2_chars: # Must be in NR-VRT2 set
                    skipped_char_not_in_project +=1 # Or "not eligible for VRT2 ref"
                    continue
                target_is_nrvg_for_ref = True
            elif character not in self.project_glyph_chars_cache: # Standard glyph or ref for standard
                skipped_char_not_in_project += 1; continue


            # Load, scale, and prepare image data
            qimage = QImage(file_path_str)
            if qimage.isNull():
                skipped_image_load_error += 1; continue

            scaled_image = qimage.scaled(target_image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            final_qimage = QImage(target_image_size, QImage.Format_ARGB32_Premultiplied)
            final_qimage.fill(QColor(Qt.white)) # Ensure white background
            painter = QPainter(final_qimage)
            x_offset = (target_image_size.width() - scaled_image.width()) // 2
            y_offset = (target_image_size.height() - scaled_image.height()) // 2
            painter.drawImage(x_offset, y_offset, scaled_image); painter.end()
            processed_pixmap = QPixmap.fromImage(final_qimage) # For UI update if needed

            byte_array = QByteArray(); buffer = QBuffer(byte_array)
            buffer.open(QIODevice.WriteOnly); final_qimage.save(buffer, "PNG")
            image_data_bytes = byte_array.data()

            # Save to DB (using direct DB methods for batch efficiency)
            success = False
            if import_type == "glyph":
                success = self.db_manager.update_glyph_image_data_bytes(character, image_data_bytes)
            elif import_type == "reference":
                if target_is_nrvg_for_ref:
                    success = self.db_manager.update_vrt2_glyph_reference_image_data_bytes(character, image_data_bytes)
                else:
                    success = self.db_manager.update_glyph_reference_image_data_bytes(character, image_data_bytes)

            if success:
                processed_count += 1
                # Update UI if the imported image matches current editor glyph
                if import_type == "glyph":
                    self.glyph_grid_widget.update_glyph_preview(character, processed_pixmap)
                
                current_editor_char = self.drawing_editor_widget.canvas.current_glyph_character
                is_editor_vrt2_editing = self.drawing_editor_widget.canvas.editing_vrt2_glyph
                
                if character == current_editor_char:
                    if import_type == "glyph" and not is_editor_vrt2_editing: # Standard glyph updated
                        self.load_glyph_for_editing(character, is_vrt2_edit_mode=False) # Reload to show
                    elif import_type == "reference":
                        if target_is_nrvg_for_ref and is_editor_vrt2_editing: # VRT2 ref updated
                            self.drawing_editor_widget.canvas.reference_image = processed_pixmap.copy()
                            self.drawing_editor_widget.canvas.update()
                            self.drawing_editor_widget.delete_ref_button.setEnabled(True)
                        elif not target_is_nrvg_for_ref and not is_editor_vrt2_editing: # Standard ref updated
                            self.drawing_editor_widget.canvas.reference_image = processed_pixmap.copy()
                            self.drawing_editor_widget.canvas.update()
                            self.drawing_editor_widget.delete_ref_button.setEnabled(True)
            else:
                # DB update failed (e.g., char somehow not in table after checks)
                table_info = "vrt2_glyphs.reference_image_data" if target_is_nrvg_for_ref else "glyphs.reference_image_data" if import_type == "reference" else "glyphs.image_data"
                # print(f"DB update failed for {character} in {table_info}") # Debug log
                db_update_errors +=1
        
        self.statusBar().showMessage(f"{file_dialog_title} 完了", 5000)
        summary_parts = [f"{processed_count} 件の画像を処理・保存しました。"]
        if skipped_filename_format > 0: summary_parts.append(f"{skipped_filename_format} 件: ファイル名形式エラー")
        if skipped_unicode_conversion > 0: summary_parts.append(f"{skipped_unicode_conversion} 件: Unicode変換エラー")
        if skipped_char_not_in_project > 0: summary_parts.append(f"{skipped_char_not_in_project} 件: 文字がプロジェクトに未登録/不適切")
        if skipped_image_load_error > 0: summary_parts.append(f"{skipped_image_load_error} 件: 画像読み込みエラー")
        if db_update_errors > 0: summary_parts.append(f"{db_update_errors} 件: DB更新エラー")
        QMessageBox.information(self, "一括読み込み結果", "\n".join(summary_parts))



    def closeEvent(self, event: QEvent):
        if self._project_loading_in_progress:
            reply = QMessageBox.question(self, '確認', 
                                         "プロジェクトの読み込み処理が進行中です。強制終了しますか？\n（データが破損する可能性があります）", 
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                event.ignore()
                return

        reply = QMessageBox.question(self, '確認', "アプリケーションを終了しますか？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            # Synchronously save current advance width if applicable
            if not self._project_loading_in_progress and \
               self.current_project_path and \
               self.drawing_editor_widget.canvas.current_glyph_character and \
               not self.drawing_editor_widget.canvas.editing_vrt2_glyph: # Only for standard glyphs
                current_char = self.drawing_editor_widget.canvas.current_glyph_character
                adv_width = self.drawing_editor_widget.adv_width_spinbox.value()
                self._save_current_advance_width_sync(current_char, adv_width)

            # Stop timers and workers
            if self._kv_deferred_update_timer and self._kv_deferred_update_timer.isActive():
                self._kv_deferred_update_timer.stop()
            with QMutexLocker(self._worker_management_mutex):
                if self.related_kanji_worker and self.related_kanji_worker.isRunning():
                    self.related_kanji_worker.cancel() # Request cancellation
            
            # Handle running export process
            if self.export_process and self.export_process.state() != QProcess.NotRunning:
                self.export_process.terminate() # Ask nicely first
                if not self.export_process.waitForFinished(3000): # Wait a bit
                    self.export_process.kill() # Force kill
                    self.export_process.waitForFinished(1000) # Wait for kill

            self.thread_pool.waitForDone(-1) # Wait for all QRunnables
            event.accept()
        else:
            event.ignore()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self._project_loading_in_progress: return

        # Debounce KV update on resize
        if self.kv_resize_timer and self.kv_resize_timer.isActive():
            self.kv_resize_timer.stop()
        
        if not self.kv_resize_timer: # Initialize if not exists
            self.kv_resize_timer = QTimer(self)
            self.kv_resize_timer.setSingleShot(True)
            self.kv_resize_timer.timeout.connect(self._on_kv_resize_finished)
        
        self.kv_resize_timer.start(250) # Delay before reacting to resize finished

    def _on_kv_resize_finished(self):
        if self._project_loading_in_progress: return
        # Trigger KV update if data is loaded and a char is active
        if self._kanji_viewer_data_loaded_successfully and \
           self.drawing_editor_widget.canvas.current_glyph_character:
            
            char_to_update = self.drawing_editor_widget.canvas.current_glyph_character
            if char_to_update and len(char_to_update) == 1: # Valid single char
                with QMutexLocker(self._worker_management_mutex):
                    self._kv_char_to_update = char_to_update
                self._kv_deferred_update_timer.start(self._kv_update_delay_ms)


if __name__ == "__main__":
    QApplication.setApplicationName("P-Glyph")
    # QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling) # Enable for HiDPI if needed

    app = QApplication(sys.argv)
    
    # Set application icon (try multiple paths for robustness)
    icon_path = "ico.ico" # Relative to execution dir
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path_data = os.path.join(script_dir, "data", "ico.ico")
        if os.path.exists(icon_path_data):
            app.setWindowIcon(QIcon(icon_path_data))
        else:
            # Try one level up (if script is in a subdir of project root)
            parent_dir = os.path.dirname(script_dir)
            icon_path_parent_data = os.path.join(parent_dir, "data", "ico.ico")
            if os.path.exists(icon_path_parent_data):
                 app.setWindowIcon(QIcon(icon_path_parent_data))
            # else: no icon found, system default will be used



    window = MainWindow()
    window.show()
    sys.exit(app.exec())
