

import sys
import os
import sqlite3
import functools
import re 
import json 
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Set, Union

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QSlider, QLabel, QComboBox,
    QCheckBox, QGridLayout, QButtonGroup, QFrame,
    QScrollArea, QTextEdit, QSizePolicy, QFileDialog,
    QDialog, QMessageBox, QLineEdit, QSpinBox, QDialogButtonBox,
    QStatusBar,
    QTabBar, QStyle, QStyleOptionTab, QSpacerItem, 
    QTabWidget, QAbstractButton, QMenu,
    QTableView, QAbstractItemView, QStyledItemDelegate, QStyleOptionViewItem,
    QStyleOptionFrame
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
    QRect, QMutex, QMutexLocker, QEvent, QThread,
    QAbstractTableModel, QModelIndex, QItemSelectionModel
)


# --- Constants ---
MAX_HISTORY_SIZE = 20
VIRTUAL_MARGIN = 30
CANVAS_IMAGE_WIDTH = 500
CANVAS_IMAGE_HEIGHT = 500
DEFAULT_GLYPH_PREVIEW_SIZE = QSize(64, 64) # Used by Delegate for preview calculation
# GLYPH_GRID_WIDTH = 400 # TabWidget will manage width
PROPERTIES_WIDTH = 250
# GRID_COLUMNS = 4 # Defined in GlyphTableModel now
# GLYPH_ITEM_WIDTH = 80 # Defined by Delegate's sizeHint and TableView column width
# GLYPH_ITEM_HEIGHT = 100 # Defined by Delegate's sizeHint and TableView row height
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
SETTING_LAST_ACTIVE_GLYPH_IS_VRT2 = "last_active_glyph_is_vrt2" # New setting for active tab/type
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
VRT2_PREVIEW_BACKGROUND_TINT = QColor("#957CF6") # Slightly adjusted for delegate background
REFERENCE_GLYPH_DISPLAY_DELAY = 450
FONT_BOOKMARKS_FILENAME = "font_bookmarks.json"

# Settings for Kanji Viewer state
SETTING_KV_CURRENT_FONT = "kv_current_font" 
SETTING_KV_DISPLAY_MODE = "kv_display_mode" 
DEFAULT_KV_MODE_FOR_SETTINGS = 1 

# GLYPH_LOAD_BATCH_SIZE = 100 # グリフ読み込みのバッチサイズ # Model handles its own loading
# GLYPH_UI_UPDATE_BATCH_SIZE = 100 # Batch size for UI updates in GlyphGridWidget # Not used with QTableView

# --- Constants for GlyphTableDelegate ---
DELEGATE_CHAR_LABEL_PADDING = 3
DELEGATE_CHAR_LABEL_BORDER_RADIUS = 3
DELEGATE_CELL_CONTENT_PADDING = 3
DELEGATE_PREVIEW_PADDING = 2
DELEGATE_FRAME_BORDER_WIDTH = 1
DELEGATE_SELECTION_OUTLINE_WIDTH = 1
DELEGATE_ITEM_MARGIN = 2
DELEGATE_PLACEHOLDER_COLOR = QColor(220, 220, 220)
DELEGATE_CELL_BASE_WIDTH = 80 # Approximate base width for item
DELEGATE_GRID_COLUMNS = 5 # Number of columns in the table view grid

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
                table_name = "vrt2_glyphs" if self.is_vrt2_glyph else "glyphs"
                # self.signals.error.emit(f"Error saving reference image for '{self.character}' in {table_name}: Character not found or no update occurred.")
                # Silently allow if char not found, could be due to set changes.
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
    basic_info_loaded = Signal(dict) 
    # Batch of glyph data: (character, has_image_initially, initial_image_bytes_if_any)
    # This signal structure is changed: MainWindow will now pass metadata lists to GlyphGridWidget's models.
    # Instead, we can emit progress for "metadata loaded" vs "initial images loaded".
    # Let's simplify: basic_info_loaded is enough. GlyphGridWidget models will load their own data.
    # However, for initial population, MainWindow still needs the lists of characters.
    # So, basic_info_loaded will now include:
    #   'char_set_list_with_img_info': List[Tuple[str, bool]] (char, has_image) for standard
    #   'nr_vrt2_list_with_img_info': List[Tuple[str, bool]] (char, has_image) for VRT2
    load_progress = Signal(int, str) 
    error = Signal(str)
    finished = Signal()

class LoadProjectWorker(QRunnable):
    def __init__(self, db_path: str):
        super().__init__()
        self.db_path = db_path
        self.signals = LoadProjectWorkerSignals()

    def _get_char_list_with_image_info(self, db_manager: "DatabaseManager", char_list: List[str], is_vrt2: bool) -> List[Tuple[str, bool]]:
        """For a list of characters, checks DB if image_data exists."""
        results = []
        if not char_list: return results
        conn = db_manager._get_connection() # Use local connection for thread safety
        cursor = conn.cursor()
        table = "vrt2_glyphs" if is_vrt2 else "glyphs"
        for char_val in char_list:
            cursor.execute(f"SELECT image_data IS NOT NULL FROM {table} WHERE character = ?", (char_val,))
            row = cursor.fetchone()
            has_image = row[0] if row and row[0] is not None else False
            results.append((char_val, has_image))
        conn.close()
        return results

    @Slot()
    def run(self):
        try:
            db_manager = DatabaseManager(self.db_path) 

            self.signals.load_progress.emit(0, "基本設定を読み込み中...")
            char_set_list_raw = db_manager.get_project_character_set()
            r_vrt2_list_raw = db_manager.get_rotated_vrt2_character_set()
            nr_vrt2_list_raw = db_manager.get_non_rotated_vrt2_character_set()
            
            # Get image existence info for initial model population
            self.signals.load_progress.emit(20, "グリフメタデータを確認中(標準)...")
            char_set_list_with_img_info = self._get_char_list_with_image_info(db_manager, char_set_list_raw, is_vrt2=False)
            
            self.signals.load_progress.emit(40, "グリフメタデータを確認中(縦書き)...")
            nr_vrt2_list_with_img_info = self._get_char_list_with_image_info(db_manager, nr_vrt2_list_raw, is_vrt2=True)

            self.signals.load_progress.emit(60, "GUI設定を読み込み中...")
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
            kv_display_mode_str = db_manager.load_gui_setting(SETTING_KV_DISPLAY_MODE, str(MainWindow.KV_MODE_WRITTEN_GLYPHS))
            try:
                kv_display_mode_val = int(kv_display_mode_str)
                if kv_display_mode_val not in [MainWindow.KV_MODE_FONT_DISPLAY, MainWindow.KV_MODE_WRITTEN_GLYPHS, MainWindow.KV_MODE_HIDDEN]:
                    kv_display_mode_val = MainWindow.KV_MODE_WRITTEN_GLYPHS
            except ValueError:
                kv_display_mode_val = MainWindow.KV_MODE_WRITTEN_GLYPHS
            
            last_active_glyph_char = db_manager.load_gui_setting(SETTING_LAST_ACTIVE_GLYPH)
            last_active_glyph_is_vrt2_str = db_manager.load_gui_setting(SETTING_LAST_ACTIVE_GLYPH_IS_VRT2, 'False')
            last_active_glyph_is_vrt2 = last_active_glyph_is_vrt2_str.lower() == 'true'


            basic_loaded_data = {
                'char_set_list': char_set_list_raw, # Raw list for properties widget
                'r_vrt2_list': r_vrt2_list_raw,
                'nr_vrt2_list': nr_vrt2_list_raw,
                'char_set_list_with_img_info': char_set_list_with_img_info, # For std model
                'nr_vrt2_list_with_img_info': nr_vrt2_list_with_img_info, # For vrt2 model
                'font_name': font_name,
                'font_weight': font_weight,
                'gui_settings': gui_settings,
                'kv_font_actual_name': kv_font_actual_name,
                'kv_display_mode_val': kv_display_mode_val,
                'last_active_glyph_char': last_active_glyph_char,
                'last_active_glyph_is_vrt2': last_active_glyph_is_vrt2,
            }
            self.signals.basic_info_loaded.emit(basic_loaded_data)
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
        """, ('.notdef', -1, empty_image_bytes, DEFAULT_ADVANCE_WIDTH)) 

        whitespace_chars_to_initialize = [' ', '　', '\t'] 
        for char_val in characters:
            if len(char_val) != 1: continue
            image_data_for_char = None 
            if char_val in whitespace_chars_to_initialize:
                 image_data_for_char = empty_image_bytes
            try:
                unicode_val = ord(char_val)
                cursor.execute("""
                    INSERT OR IGNORE INTO glyphs (character, unicode_val, image_data, advance_width)
                    VALUES (?, ?, ?, ?)
                """, (char_val, unicode_val, image_data_for_char, DEFAULT_ADVANCE_WIDTH))
            except TypeError: pass 

        for char_val in nr_vrt2_chars:
            if len(char_val) == 1: 
                 cursor.execute("INSERT OR IGNORE INTO vrt2_glyphs (character) VALUES (?)", (char_val,))
        
        self._save_default_gui_settings(cursor)
        conn.commit(); conn.close()

    def _save_default_gui_settings(self, cursor: sqlite3.Cursor):
        defaults = {
            SETTING_PEN_WIDTH: str(DEFAULT_PEN_WIDTH),
            SETTING_PEN_SHAPE: DEFAULT_PEN_SHAPE,
            SETTING_CURRENT_TOOL: DEFAULT_CURRENT_TOOL,
            SETTING_MIRROR_MODE: str(DEFAULT_MIRROR_MODE),
            SETTING_GLYPH_MARGIN_WIDTH: str(DEFAULT_GLYPH_MARGIN_WIDTH),
            SETTING_LAST_ACTIVE_GLYPH: "", 
            SETTING_LAST_ACTIVE_GLYPH_IS_VRT2: "False",
            SETTING_FONT_NAME: DEFAULT_FONT_NAME,
            SETTING_FONT_WEIGHT: DEFAULT_FONT_WEIGHT,
            SETTING_REFERENCE_IMAGE_OPACITY: str(DEFAULT_REFERENCE_IMAGE_OPACITY),
            SETTING_KV_CURRENT_FONT: "", 
            SETTING_KV_DISPLAY_MODE: str(DEFAULT_KV_MODE_FOR_SETTINGS), 
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
            seen = set(); unique_ordered_chars = []
            for c in row['value']:
                if len(c) == 1 and c not in seen: unique_ordered_chars.append(c); seen.add(c)
            return sorted(unique_ordered_chars, key=ord) 
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
            db_chars = {row['character'] for row in db_chars_query_result if not (is_main_set and row['character'] == '.notdef')}
            new_chars_set = set(valid_characters)
            empty_image_bytes_for_new_main_glyphs = self._create_empty_image_data() if is_main_set else None

            for char_to_add in new_chars_set - db_chars:
                if is_main_set:
                    cursor.execute("INSERT OR IGNORE INTO glyphs (character, unicode_val, image_data, advance_width) VALUES (?, ?, ?, ?)",
                                   (char_to_add, ord(char_to_add), empty_image_bytes_for_new_main_glyphs, DEFAULT_ADVANCE_WIDTH))
                elif updates_vrt2_table: 
                    cursor.execute("SELECT 1 FROM glyphs WHERE character = ?", (char_to_add,))
                    if not cursor.fetchone(): 
                         cursor.execute("INSERT OR IGNORE INTO glyphs (character, unicode_val, advance_width) VALUES (?, ?, ?)",
                                   (char_to_add, ord(char_to_add), DEFAULT_ADVANCE_WIDTH))
                    cursor.execute("INSERT OR IGNORE INTO vrt2_glyphs (character) VALUES (?)", (char_to_add,))

            for char_to_remove in db_chars - new_chars_set:
                if char_to_remove == '.notdef' and is_main_set: continue 
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
    
    def load_glyph_image_bytes(self, character: str, is_vrt2: bool = False) -> Optional[bytes]: # For ImageLoadWorker
        if not self.db_path: return None
        conn = self._get_connection(); cursor = conn.cursor()
        table = "vrt2_glyphs" if is_vrt2 else "glyphs"
        cursor.execute(f"SELECT image_data FROM {table} WHERE character = ?", (character,))
        row = cursor.fetchone(); conn.close()
        return row['image_data'] if row and row['image_data'] else None


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

    def save_vrt2_glyph_image(self, character: str, pixmap: QPixmap): 
        if not self.db_path: return
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

    def save_glyph_advance_width(self, character: str, advance_width: int): 
        if not self.db_path: return
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("UPDATE glyphs SET advance_width = ? WHERE character = ?", (advance_width, character))
        conn.commit(); conn.close()

    def get_all_glyphs_with_preview_data(self) -> List[Tuple[str, Optional[bytes]]]:
        if not self.db_path: return []
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("SELECT character, unicode_val, image_data FROM glyphs ORDER BY unicode_val ASC, character ASC")
        results = []
        all_db_rows = cursor.fetchall(); conn.close()
        for row in all_db_rows:
            char_val = row['character']
            image_bytes = row['image_data'] if row['image_data'] else None
            results.append((char_val, image_bytes))
        return results

    def get_all_defined_nrvg_with_preview_data(self) -> List[Tuple[str, Optional[bytes]]]:
        if not self.db_path: return []
        nrvg_chars_from_settings = self.get_non_rotated_vrt2_character_set() 
        if not nrvg_chars_from_settings: return []
        conn = self._get_connection(); cursor = conn.cursor(); results = []
        for char_val in nrvg_chars_from_settings: 
            cursor.execute("SELECT image_data FROM vrt2_glyphs WHERE character = ?", (char_val,))
            row = cursor.fetchone()
            image_bytes = None
            if row and row['image_data']: image_bytes = row['image_data']
            results.append((char_val, image_bytes)) 
        conn.close(); return results

    def load_gui_setting(self, key: str, default_value: Optional[str] = None) -> Optional[str]:
        if not self.db_path: return default_value
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("SELECT value FROM project_settings WHERE key = ?", (key,))
        row = cursor.fetchone(); conn.close()
        return row['value'] if row else default_value

    def update_glyph_image_data_bytes(self, character: str, image_data: bytes) -> bool:
        if not self.db_path: return False
        conn = self._get_connection(); cursor = conn.cursor()
        cursor.execute("SELECT 1 FROM glyphs WHERE character = ?", (character,)); exists = cursor.fetchone()
        if not exists: conn.close(); return False 
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
        self.ascender_height_for_baseline = DEFAULT_ASCENDER_HEIGHT 
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
        if width >= 0: 
            self.current_glyph_advance_width = width
            self.glyph_advance_width_changed.emit(width); self.update()

    def load_glyph(self, character: str, pixmap: Optional[QPixmap], reference_pixmap: Optional[QPixmap], advance_width: int, is_vrt2: bool = False):
        self.current_glyph_character = character; self.editing_vrt2_glyph = is_vrt2
        self.current_glyph_advance_width = advance_width; self.reference_image = reference_pixmap
        if pixmap:
            self.image = pixmap.copy()
            if self.image.size() != self.image_size: 
                temp_target_pixmap = QPixmap(self.image_size); temp_target_pixmap.fill(Qt.white)
                painter = QPainter(temp_target_pixmap); painter.drawPixmap(0, 0, self.image); painter.end()
                self.image = temp_target_pixmap
        else: self.image = QPixmap(self.image_size); self.image.fill(QColor(Qt.white))
        self.undo_stack = []; self.redo_stack = []
        self._save_state_to_undo_stack(is_initial_load=True); self.update()

    def set_reference_image_opacity(self, opacity: float): self.reference_image_opacity = max(0.0, min(1.0, opacity)); self.update()
    def get_current_image_and_type(self) -> Tuple[QPixmap, bool]: return self.image.copy(), self.editing_vrt2_glyph
    def get_current_image(self) -> QPixmap: return self.image.copy() 
    def _emit_undo_redo_state(self): self.undo_redo_state_changed.emit(len(self.undo_stack) > 1, bool(self.redo_stack))

    def _save_state_to_undo_stack(self, is_initial_load: bool = False):
        if len(self.undo_stack) >= MAX_HISTORY_SIZE: self.undo_stack.pop(0)
        self.undo_stack.append(self.image.copy())
        if not is_initial_load: self.redo_stack.clear() 
        self._emit_undo_redo_state()

    def undo(self):
        if len(self.undo_stack) > 1: 
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
        if self.move_mode: self.move_mode = False; self.unsetCursor() 
        if self.current_pen_color != Qt.white: self.last_brush_color = self.current_pen_color 
        self.set_current_pen_color(QColor(Qt.white)); self.drawing = False; self.current_tool = "eraser"
        self.tool_changed.emit("eraser"); self.update()

    def set_brush_mode(self):
        if self.move_mode: self.move_mode = False; self.unsetCursor() 
        self.set_current_pen_color(self.last_brush_color); self.drawing = False; self.current_tool = "brush"
        self.tool_changed.emit("brush"); self.update()

    def set_mirror_mode(self, enabled: bool):
        if self.mirror_mode == enabled: return
        self.mirror_mode = enabled
        if self.drawing: self.drawing = False; self.stroke_points = []; self.current_path = QPainterPath() 
        self.update()

    def set_move_mode(self, enabled: bool):
        if enabled and self.current_tool == "move": return 
        if not enabled and self.current_tool != "move": return 
        
        self.move_mode = enabled; self.moving_image = False; self.move_offset = QPointF()
        if enabled:
            self.drawing = False 
            self.setCursor(Qt.OpenHandCursor); self.current_tool = "move"; self.tool_changed.emit("move")
        else: 
            self.unsetCursor()
            if self.current_tool == "move": 
                if self.current_pen_color == QColor(Qt.white): self.set_eraser_mode()
                else: self.set_brush_mode()
        self.update()

    def _map_view_point_to_logical_point(self, view_point: QPointF) -> QPointF:
        image_local_view_point = view_point - QPointF(VIRTUAL_MARGIN, VIRTUAL_MARGIN)
        if self.mirror_mode: 
            return QPointF(self.image_size.width() - image_local_view_point.x(), image_local_view_point.y())
        return image_local_view_point

    def _generate_path_from_points(self, points: list[QPointF], finalize=False) -> QPainterPath:
        path = QPainterPath(); n = len(points)
        if n == 0: return path
        path.moveTo(points[0])
        if n == 1: path.lineTo(points[0]); return path 
        for i in range(n - 1):
            p1 = points[i]; p2 = points[i+1]
            p0 = points[i-1] if i > 0 else p1 
            p3 = points[i+2] if i + 2 < n else p2 
            alpha = 0.5 
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
               path_to_draw.elementAt(0).isMoveTo(): 
                first_point = QPointF(path_to_draw.elementAt(0).x, path_to_draw.elementAt(0).y)
                rect_size = float(self.pen_width)
                rect = QRectF(first_point.x() - rect_size / 2.0, first_point.y() - rect_size / 2.0, rect_size, rect_size)
                painter.drawRect(rect)
                return
            sampling_step_length = max(1.0, float(self.pen_width) / 10.0) 
            num_samples = 0
            if sampling_step_length > 0: num_samples = max(1, int(path_length / sampling_step_length))
            else: num_samples = 1 
            for i in range(num_samples + 1): 
                percent = 0.0
                if num_samples > 0 : percent = float(i) / num_samples
                else: percent = 0.0 
                point_on_path = path_to_draw.pointAtPercent(percent)
                if point_on_path.isNull(): continue 
                rect_size = float(self.pen_width)
                rect = QRectF(point_on_path.x() - rect_size / 2.0, point_on_path.y() - rect_size / 2.0, rect_size, rect_size)
                painter.drawRect(rect)

    def _apply_image_move(self, view_drag_delta: QPointF):
        logical_offset_x = view_drag_delta.x()
        if self.mirror_mode: logical_offset_x = -view_drag_delta.x() 
        logical_offset_y = view_drag_delta.y()
        int_logical_offset = QPointF(round(logical_offset_x), round(logical_offset_y)).toPoint()
        if int_logical_offset.x() == 0 and int_logical_offset.y() == 0: return 
        moved_image = QPixmap(self.image_size); moved_image.fill(QColor(Qt.white)) 
        painter = QPainter(moved_image)
        painter.drawPixmap(int_logical_offset, self.image) 
        painter.end()
        self.image = moved_image 

    def paintEvent(self, event: QPaintEvent):
        canvas_painter = QPainter(self)
        canvas_painter.fillRect(self.rect(), QColor(200, 200, 200)) 
        image_area_rect_in_widget = QRectF(VIRTUAL_MARGIN, VIRTUAL_MARGIN, self.image_size.width(), self.image_size.height())
        canvas_painter.fillRect(image_area_rect_in_widget, Qt.white)
        canvas_painter.save(); canvas_painter.translate(image_area_rect_in_widget.topLeft()) 
        temp_pixmap_for_drawing_content = QPixmap(self.image_size)
        temp_pixmap_for_drawing_content.fill(Qt.transparent) 
        image_content_painter = QPainter(temp_pixmap_for_drawing_content)
        image_content_painter.drawPixmap(0,0, self.image) 
        if not self.move_mode and self.drawing and not self.current_path.isEmpty(): 
            self._draw_path_on_painter(image_content_painter, self.current_path)
        image_content_painter.end()
        preview_draw_offset = QPointF(0,0)
        if self.move_mode and self.moving_image: preview_draw_offset = self.move_offset 
        if self.mirror_mode:
            canvas_painter.save()
            canvas_painter.translate(self.image_size.width(), 0); canvas_painter.scale(-1, 1) 
            mirrored_preview_offset = QPointF(-preview_draw_offset.x(), preview_draw_offset.y()) 
            canvas_painter.drawPixmap(mirrored_preview_offset.toPoint(), temp_pixmap_for_drawing_content)
            canvas_painter.restore()
        else:
            canvas_painter.drawPixmap(preview_draw_offset.toPoint(), temp_pixmap_for_drawing_content)
        if self.reference_image and self.reference_image_opacity > 0.0 and \
           self.reference_image.size() == self.image_size: 
            canvas_painter.save()
            canvas_painter.setOpacity(self.reference_image_opacity)
            canvas_painter.setCompositionMode(QPainter.CompositionMode_Multiply) 
            if self.mirror_mode: 
                canvas_painter.save()
                canvas_painter.translate(self.image_size.width(), 0); canvas_painter.scale(-1, 1)
                canvas_painter.drawPixmap(QPoint(0,0), self.reference_image)
                canvas_painter.restore()
            else:
                canvas_painter.drawPixmap(QPoint(0,0), self.reference_image)
            canvas_painter.restore()
        img_height = float(self.image_size.height()); img_width = float(self.image_size.width())
        adv_pen = QPen(QColor(255, 0, 0, 180), 1, Qt.DotLine); adv_pen.setCosmetic(True)
        canvas_painter.setPen(adv_pen)
        if self.editing_vrt2_glyph: 
            line_y_canvas_adv = (float(self.current_glyph_advance_width) / 1000.0) * img_height
            if line_y_canvas_adv >= 0 : canvas_painter.drawLine(QPointF(0, line_y_canvas_adv), QPointF(img_width, line_y_canvas_adv))
        else: 
            line_x_canvas_adv = (float(self.current_glyph_advance_width) / 1000.0) * img_width
            if line_x_canvas_adv >= 0: canvas_painter.drawLine(QPointF(line_x_canvas_adv, 0), QPointF(line_x_canvas_adv, img_height))
        guideline_pen = QPen(QColor(180, 180, 180), 1, Qt.DashLine); guideline_pen.setCosmetic(True)
        canvas_painter.setPen(guideline_pen)
        for i in range(1, 3): 
            y_pos = (img_height * i / 3.0); canvas_painter.drawLine(QPointF(0, y_pos), QPointF(img_width, y_pos))
            x_pos = (img_width * i / 3.0); canvas_painter.drawLine(QPointF(x_pos, 0), QPointF(x_pos, img_height))
        crosshair_pen = QPen(QColor(150, 150, 150), 1, Qt.SolidLine); crosshair_pen.setCosmetic(True)
        canvas_painter.setPen(crosshair_pen)
        center_x, center_y = img_width / 2.0, img_height / 2.0; CROSSHAIR_ARM_LENGTH = 25 
        canvas_painter.drawLine(QPointF(center_x - CROSSHAIR_ARM_LENGTH, center_y), QPointF(center_x + CROSSHAIR_ARM_LENGTH, center_y))
        canvas_painter.drawLine(QPointF(center_x, center_y - CROSSHAIR_ARM_LENGTH), QPointF(center_x, center_y + CROSSHAIR_ARM_LENGTH))
        base_margin_px = float(self.glyph_margin_width) 
        if base_margin_px > 0:
            margin_pen = QPen(QColor(100, 100, 200), 1, Qt.DotLine); margin_pen.setCosmetic(True)
            canvas_painter.setPen(margin_pen)
            left_x, top_y, right_x, bottom_y = 0.0,0.0,0.0,0.0
            if self.editing_vrt2_glyph:
                left_x = base_margin_px; right_x = img_width - base_margin_px
                top_y = base_margin_px
                advance_edge_y = (float(self.current_glyph_advance_width) / 1000.0) * img_height
                bottom_y = advance_edge_y - base_margin_px
            else: 
                left_x = base_margin_px
                advance_edge_x = (float(self.current_glyph_advance_width) / 1000.0) * img_width
                right_x = advance_edge_x - base_margin_px
                top_y = base_margin_px; bottom_y = img_height - base_margin_px
            margin_rect = QRectF(left_x, top_y, right_x - left_x, bottom_y - top_y)
            if margin_rect.isValid() and margin_rect.width() > 0 and margin_rect.height() > 0 :
                canvas_painter.drawRect(margin_rect)
        canvas_painter.restore() 
        outer_margin_path = QPainterPath()
        outer_margin_path.setFillRule(Qt.OddEvenFill) 
        outer_margin_path.addRect(QRectF(self.rect())) 
        outer_margin_path.addRect(image_area_rect_in_widget) 
        canvas_painter.save()
        canvas_painter.setBrush(QColor(200, 200, 200)) 
        canvas_painter.setPen(Qt.NoPen)
        canvas_painter.drawPath(outer_margin_path)
        canvas_painter.restore()
        canvas_painter.save()
        border_pen = QPen(Qt.darkGray); border_pen.setWidth(1) 
        canvas_painter.setPen(border_pen); canvas_painter.setBrush(Qt.NoBrush)
        canvas_painter.drawRect(image_area_rect_in_widget.adjusted(0,0,-1,-1)) 
        canvas_painter.restore()

    def mousePressEvent(self, event: QMouseEvent):
        if not self.current_glyph_character: return 
        if self.move_mode:
            if event.button() == Qt.LeftButton:
                image_rect_in_widget = QRectF(VIRTUAL_MARGIN, VIRTUAL_MARGIN, self.image_size.width(), self.image_size.height())
                if image_rect_in_widget.contains(event.position()):
                    self.moving_image = True; self.move_start_pos = event.position(); self.move_offset = QPointF()
                    self.setCursor(Qt.ClosedHandCursor); self.update()
            return 
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
                self.move_offset = delta 
                self.update() 
            return
        if (event.buttons() & Qt.LeftButton) and self.drawing:
            view_pos = event.position()
            logical_pos = self._map_view_point_to_logical_point(view_pos)
            if not self.stroke_points or (logical_pos - self.stroke_points[-1]).manhattanLength() >= 1.0:
                self.stroke_points.append(logical_pos)
                self._rebuild_current_path_from_stroke_points(finalize=False); self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if not self.current_glyph_character: return
        if self.move_mode and self.moving_image:
            if event.button() == Qt.LeftButton:
                self._apply_image_move(self.move_offset) 
                self.moving_image = False; self.move_offset = QPointF() 
                if self.move_mode: self.setCursor(Qt.OpenHandCursor) 
                else: self.unsetCursor() 
                self._save_state_to_undo_stack() 
                if self.current_glyph_character: self.glyph_modified_signal.emit(self.current_glyph_character, self.image.copy(), self.editing_vrt2_glyph)
                self.update()
            return
        if event.button() == Qt.LeftButton and self.drawing:
            if self.stroke_points: 
                view_pos = event.position()
                final_logical_pos = self._map_view_point_to_logical_point(view_pos)
                if len(self.stroke_points) == 1 and self.stroke_points[0] == final_logical_pos: pass 
                elif not self.stroke_points or (final_logical_pos - self.stroke_points[-1]).manhattanLength() >= 0.1: 
                    self.stroke_points.append(final_logical_pos)
                self._rebuild_current_path_from_stroke_points(finalize=True) 
                if not self.current_path.isEmpty():
                    image_painter = QPainter(self.image)
                    self._draw_path_on_painter(image_painter, self.current_path)
                    image_painter.end()
                    self._save_state_to_undo_stack() 
                    if self.current_glyph_character: self.glyph_modified_signal.emit(self.current_glyph_character, self.image.copy(), self.editing_vrt2_glyph)
            self.drawing = False; self.stroke_points = []; self.current_path = QPainterPath(); self.update()

    def dragEnterEvent(self, event: QDragEnterEvent):
        mime_data = event.mimeData()
        if mime_data.hasUrls():
            urls = mime_data.urls()
            if urls and urls[0].isLocalFile():
                file_path = urls[0].toLocalFile(); ext = Path(file_path).suffix.lower()
                if ext in ['.png', '.jpg', '.jpeg', '.bmp']: event.acceptProposedAction(); return
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
                target_size = self.image_size
                scaled_image = dropped_qimage.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                final_image = QImage(target_size, QImage.Format_ARGB32_Premultiplied)
                final_image.fill(QColor(Qt.white)) 
                painter = QPainter(final_image)
                x_offset = (target_size.width() - scaled_image.width()) // 2
                y_offset = (target_size.height() - scaled_image.height()) // 2
                painter.drawImage(x_offset, y_offset, scaled_image); painter.end()
                self.image = QPixmap.fromImage(final_image)
                self._save_state_to_undo_stack() 
                if self.current_glyph_character: self.glyph_modified_signal.emit(self.current_glyph_character, self.image.copy(), self.editing_vrt2_glyph)
                self.update(); event.acceptProposedAction()
            except Exception as e:
                QMessageBox.critical(self, "ドロップ処理エラー", f"画像の処理中にエラーが発生しました: {e}")
                event.ignore()
        else: event.ignore()

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
        self.rotated_vrt2_chars: Set[str] = set() 
        main_layout = QVBoxLayout(self)
        self.unicode_label = QLineEdit("Unicode: N/A") 
        self.unicode_label.setReadOnly(True); self.unicode_label.setAlignment(Qt.AlignCenter)
        font = self.unicode_label.font(); font.setPointSize(10); self.unicode_label.setFont(font)
        self.unicode_label.setStyleSheet("QLineEdit { border: none; background: transparent; }") 
        main_layout.addWidget(self.unicode_label)
        main_layout.addWidget(self.canvas, 0, Qt.AlignCenter) 
        controls_outer_layout = QVBoxLayout(); controls_outer_layout.setSpacing(5) 
        top_controls_layout = QHBoxLayout()
        self.pen_button = QPushButton("ペン (B)"); self.pen_button.setCheckable(True)
        self.eraser_button = QPushButton("消しゴム (E)"); self.eraser_button.setCheckable(True)
        self.move_button = QPushButton("移動 (V)"); self.move_button.setCheckable(True)
        self.tool_button_group = QButtonGroup(self)
        self.tool_button_group.addButton(self.pen_button); self.tool_button_group.addButton(self.eraser_button)
        self.tool_button_group.addButton(self.move_button); self.tool_button_group.setExclusive(True)
        self.pen_button.clicked.connect(self._handle_pen_button_clicked)
        self.eraser_button.clicked.connect(self._handle_eraser_button_clicked)
        self.move_button.clicked.connect(self._handle_move_button_clicked)
        top_controls_layout.addWidget(self.pen_button); top_controls_layout.addWidget(self.eraser_button)
        top_controls_layout.addWidget(self.move_button); top_controls_layout.addSpacing(10) 
        self.undo_button = QPushButton("Undo (Ctrl+Z)"); self.undo_button.clicked.connect(self.canvas.undo)
        top_controls_layout.addWidget(self.undo_button)
        self.redo_button = QPushButton("Redo (Ctrl+Y)"); self.redo_button.clicked.connect(self.canvas.redo)
        top_controls_layout.addWidget(self.redo_button); top_controls_layout.addSpacing(10)
        top_controls_layout.addWidget(QLabel("先端:"))
        self.shape_box = QComboBox(); self.shape_box.addItems(["丸", "四角"])
        self.shape_box.currentTextChanged.connect(self._handle_pen_shape_changed)
        top_controls_layout.addWidget(self.shape_box); top_controls_layout.addStretch(1)
        controls_outer_layout.addLayout(top_controls_layout)
        slider_layout = QHBoxLayout(); slider_layout.addWidget(QLabel("太さ:"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(1, 100); self.slider.setValue(self.canvas.pen_width)
        self.slider.valueChanged.connect(self._handle_pen_width_changed)
        slider_layout.addWidget(self.slider, 1) 
        controls_outer_layout.addLayout(slider_layout)
        self.pen_size_buttons_group = QWidget() 
        pen_size_grid_layout = QGridLayout(self.pen_size_buttons_group); pen_size_grid_layout.setSpacing(5)
        pen_sizes = [2, 5, 8, 10, 15, 20, 25, 30, 40, 50] 
        cols = 5
        for i, size_val in enumerate(pen_sizes):
            button = QPushButton(str(size_val)); button.setToolTip(f"{size_val}px")
            button.clicked.connect(lambda checked=False, s=size_val: self._handle_pen_width_changed(s))
            row, col = divmod(i, cols)
            pen_size_grid_layout.addWidget(button, row, col)
        controls_outer_layout.addWidget(self.pen_size_buttons_group)
        self.mirror_checkbox = QCheckBox("左右反転表示")
        self.mirror_checkbox.toggled.connect(self._handle_mirror_mode_changed)
        display_options_layout = QHBoxLayout(); display_options_layout.addStretch(1) 
        display_options_layout.addWidget(self.mirror_checkbox)
        controls_outer_layout.addLayout(display_options_layout)
        margin_layout = QHBoxLayout(); margin_layout.addWidget(QLabel("グリフマージン:"))
        self.margin_slider = QSlider(Qt.Horizontal)
        max_margin_val = min(CANVAS_IMAGE_WIDTH, CANVAS_IMAGE_HEIGHT) // 4 
        self.margin_slider.setRange(0, max_margin_val if max_margin_val > 0 else 1) 
        self.margin_slider.setValue(self.canvas.glyph_margin_width)
        self.margin_slider.valueChanged.connect(self._handle_glyph_margin_slider_change)
        self.margin_value_label = QLabel(str(self.canvas.glyph_margin_width)) 
        margin_layout.addWidget(self.margin_slider, 1); margin_layout.addWidget(self.margin_value_label)
        controls_outer_layout.addLayout(margin_layout)
        ref_opacity_layout = QHBoxLayout(); ref_opacity_layout.addWidget(QLabel("下書き透明度:"))
        self.ref_opacity_slider = QSlider(Qt.Horizontal); self.ref_opacity_slider.setRange(0, 100) 
        self.ref_opacity_slider.setValue(int(DEFAULT_REFERENCE_IMAGE_OPACITY * 100))
        self.ref_opacity_slider.valueChanged.connect(self._handle_ref_opacity_changed)
        self.ref_opacity_label = QLabel(str(int(DEFAULT_REFERENCE_IMAGE_OPACITY * 100)))
        ref_opacity_layout.addWidget(self.ref_opacity_slider, 1); ref_opacity_layout.addWidget(self.ref_opacity_label)
        controls_outer_layout.addLayout(ref_opacity_layout)
        adv_width_layout = QHBoxLayout()
        self.adv_width_label = QLabel("文字送り幅:") 
        adv_width_layout.addWidget(self.adv_width_label)
        self.adv_width_slider = QSlider(Qt.Horizontal); self.adv_width_slider.setRange(0, 1000) 
        self.adv_width_slider.setValue(DEFAULT_ADVANCE_WIDTH)
        self.adv_width_slider.valueChanged.connect(self._on_adv_width_slider_changed)
        self.adv_width_spinbox = QSpinBox(); self.adv_width_spinbox.setRange(0, 1000)
        self.adv_width_spinbox.setValue(DEFAULT_ADVANCE_WIDTH)
        self.adv_width_spinbox.valueChanged.connect(self._on_adv_width_spinbox_changed)
        adv_width_layout.addWidget(self.adv_width_slider, 1); adv_width_layout.addWidget(self.adv_width_spinbox)
        controls_outer_layout.addLayout(adv_width_layout)
        self.vrt2_and_ref_controls_layout = QHBoxLayout()
        self.vrt2_and_ref_controls_layout.setContentsMargins(0, 5, 0, 0) 
        self.vrt2_controls_widget = QWidget()
        vrt2_layout = QHBoxLayout(self.vrt2_controls_widget); vrt2_layout.setContentsMargins(0,0,0,0) 
        self.vrt2_toggle_button = QPushButton("標準グリフ編集中") 
        self.vrt2_toggle_button.setCheckable(True); self.vrt2_toggle_button.toggled.connect(self._on_vrt2_toggle)
        vrt2_layout.addWidget(self.vrt2_toggle_button)
        self.transfer_to_vrt2_button = QPushButton("標準を縦書きへ転送")
        self.transfer_to_vrt2_button.clicked.connect(self.transfer_to_vrt2_requested)
        vrt2_layout.addWidget(self.transfer_to_vrt2_button)
        self.vrt2_and_ref_controls_layout.addWidget(self.vrt2_controls_widget)
        self.vrt2_and_ref_controls_layout.addStretch(1) 
        self.ref_image_buttons_widget = QWidget()
        ref_image_buttons_layout = QHBoxLayout(self.ref_image_buttons_widget)
        ref_image_buttons_layout.setContentsMargins(0,0,0,0); ref_image_buttons_layout.setSpacing(5)
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
        self.vrt2_controls_widget.setVisible(False) 
        main_layout.addLayout(controls_outer_layout)
        self.canvas.pen_width_changed.connect(self._update_slider_value_no_signal)
        self.canvas.tool_changed.connect(self._update_tool_buttons_state_no_signal)
        self.canvas.undo_redo_state_changed.connect(self._update_undo_redo_buttons_state)
        self.canvas.glyph_margin_width_changed.connect(self._update_glyph_margin_slider_and_label_no_signal)
        self.canvas.glyph_advance_width_changed.connect(self._update_adv_width_ui_no_signal)
        self._update_tool_buttons_state_no_signal(self.canvas.current_tool)
        self._update_slider_value_no_signal(self.canvas.pen_width)
        self.shape_box.setCurrentText(DEFAULT_PEN_SHAPE if self.canvas.pen_shape == Qt.RoundCap else "四角")
        self.mirror_checkbox.setChecked(self.canvas.mirror_mode)
        self._update_glyph_margin_slider_and_label_no_signal(self.canvas.glyph_margin_width)
        self._update_adv_width_ui_no_signal(self.canvas.current_glyph_advance_width)
        self._update_undo_redo_buttons_state(False, False) 
        self._update_ref_opacity_slider_no_signal(DEFAULT_REFERENCE_IMAGE_OPACITY)
        self.canvas.set_reference_image_opacity(DEFAULT_REFERENCE_IMAGE_OPACITY)
        self.set_enabled_controls(False) 

    def set_rotated_vrt2_chars(self, chars: Set[str]):
        self.rotated_vrt2_chars = chars
        self.update_unicode_display(self.canvas.current_glyph_character) 

    def _handle_glyph_to_ref_reset_button_clicked(self):
        self.glyph_to_reference_and_reset_requested.emit(self.canvas.editing_vrt2_glyph)

    def _handle_load_reference_image_button_clicked(self):
        if not self.canvas.current_glyph_character:
            QMessageBox.warning(self, "グリフ未選択", "下書きを読み込むグリフが選択されていません。"); return
        file_path, _ = QFileDialog.getOpenFileName(self, "下書き画像を選択", "", "画像ファイル (*.png *.jpg *.jpeg *.bmp)")
        if not file_path: return
        try:
            loaded_qimage = QImage(file_path)
            if loaded_qimage.isNull():
                QMessageBox.warning(self, "画像読み込みエラー", f"画像 '{Path(file_path).name}' を読み込めませんでした。"); return
            target_size = self.canvas.image_size
            scaled_image = loaded_qimage.scaled(target_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            final_image = QImage(target_size, QImage.Format_ARGB32_Premultiplied); final_image.fill(QColor(Qt.white)) 
            painter = QPainter(final_image)
            x_offset = (target_size.width() - scaled_image.width()) // 2
            y_offset = (target_size.height() - scaled_image.height()) // 2
            painter.drawImage(x_offset, y_offset, scaled_image); painter.end()
            new_reference_pixmap = QPixmap.fromImage(final_image)
            self.canvas.reference_image = new_reference_pixmap; self.canvas.update() 
            self.delete_ref_button.setEnabled(self.pen_button.isEnabled()) 
            self.reference_image_selected_signal.emit(self.canvas.current_glyph_character, new_reference_pixmap.copy(), self.canvas.editing_vrt2_glyph)
        except Exception as e: QMessageBox.critical(self, "下書き処理エラー", f"下書き画像の処理中にエラーが発生しました: {e}")

    def _handle_delete_reference_image_button_clicked(self):
        if not self.canvas.current_glyph_character: return 
        if self.canvas.reference_image is None: return 
        reply = QMessageBox.question(self, "下書き削除の確認",
                                     f"文字 '{self.canvas.current_glyph_character}' の下書き画像を削除しますか？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.canvas.reference_image = None; self.canvas.update()
            self.delete_ref_button.setEnabled(False) 
            self.reference_image_deleted_signal.emit(self.canvas.current_glyph_character, self.canvas.editing_vrt2_glyph)

    def _handle_ref_opacity_changed(self, value: int): 
        opacity_float = value / 100.0
        self.ref_opacity_label.setText(str(value))
        self.canvas.set_reference_image_opacity(opacity_float)
        self.gui_setting_changed_signal.emit(SETTING_REFERENCE_IMAGE_OPACITY, str(opacity_float))

    def _update_ref_opacity_slider_no_signal(self, opacity_float: float):
        slider_value = int(round(opacity_float * 100))
        self.ref_opacity_slider.blockSignals(True); self.ref_opacity_slider.setValue(slider_value)
        self.ref_opacity_slider.blockSignals(False); self.ref_opacity_label.setText(str(slider_value))

    def _on_adv_width_slider_changed(self, value: int):
        self.adv_width_spinbox.blockSignals(True); self.adv_width_spinbox.setValue(value)
        self.adv_width_spinbox.blockSignals(False); self.canvas.set_current_glyph_advance_width(value) 
        if self.canvas.current_glyph_character: 
            self.advance_width_changed_signal.emit(self.canvas.current_glyph_character, value)

    def _on_adv_width_spinbox_changed(self, value: int):
        self.adv_width_slider.blockSignals(True); self.adv_width_slider.setValue(value)
        self.adv_width_slider.blockSignals(False); self.canvas.set_current_glyph_advance_width(value)
        if self.canvas.current_glyph_character: 
            self.advance_width_changed_signal.emit(self.canvas.current_glyph_character, value)

    def _update_adv_width_ui_no_signal(self, width: int):
        is_vrt2 = self.canvas.editing_vrt2_glyph if self.canvas else False
        self.adv_width_slider.blockSignals(True); self.adv_width_slider.setValue(width); self.adv_width_slider.blockSignals(False)
        self.adv_width_spinbox.blockSignals(True); self.adv_width_spinbox.setValue(width); self.adv_width_spinbox.blockSignals(False)
        self.adv_width_label.setText("文字送り高さ:" if is_vrt2 else "文字送り幅:")
        adv_controls_enabled = self.pen_button.isEnabled() 
        self.adv_width_slider.setEnabled(adv_controls_enabled)
        self.adv_width_spinbox.setEnabled(adv_controls_enabled)

    def _on_vrt2_toggle(self, checked: bool):
        if checked: self.vrt2_toggle_button.setText("縦書きグリフ編集中")
        else: self.vrt2_toggle_button.setText("標準グリフ編集中")
        self.vrt2_edit_mode_toggled.emit(checked)

    def update_vrt2_controls(self, show: bool, is_editing_vrt2: bool):
        self.vrt2_controls_widget.setVisible(show)
        if show: 
            self.vrt2_toggle_button.blockSignals(True)
            self.vrt2_toggle_button.setChecked(is_editing_vrt2)
            self.vrt2_toggle_button.setText("縦書きグリフ編集中" if is_editing_vrt2 else "標準グリフ編集中")
            self.vrt2_toggle_button.blockSignals(False)

    def _handle_pen_button_clicked(self): self.canvas.set_brush_mode(); self.gui_setting_changed_signal.emit(SETTING_CURRENT_TOOL, "brush")
    def _handle_eraser_button_clicked(self): self.canvas.set_eraser_mode(); self.gui_setting_changed_signal.emit(SETTING_CURRENT_TOOL, "eraser")
    def _handle_move_button_clicked(self):
        is_move_tool_selected = self.move_button.isChecked() 
        self.canvas.set_move_mode(is_move_tool_selected)
        self.gui_setting_changed_signal.emit(SETTING_CURRENT_TOOL, self.canvas.current_tool)

    def _handle_pen_width_changed(self, width: int): self.canvas.set_pen_width(width); self.gui_setting_changed_signal.emit(SETTING_PEN_WIDTH, str(width))
    def _handle_pen_shape_changed(self, shape_name: str): self.canvas.set_pen_shape(shape_name); self.gui_setting_changed_signal.emit(SETTING_PEN_SHAPE, shape_name)
    def _handle_mirror_mode_changed(self, checked: bool): self.canvas.set_mirror_mode(checked); self.gui_setting_changed_signal.emit(SETTING_MIRROR_MODE, str(checked))
    def _handle_glyph_margin_slider_change(self, value: int): self.canvas.set_glyph_margin_width(value); self.gui_setting_changed_signal.emit(SETTING_GLYPH_MARGIN_WIDTH, str(value))

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
            if character == '.notdef': base_text = "Glyph: .notdef"
            else:
                if isinstance(character, str) and len(character) == 1:
                    try: base_text = f"Unicode: U+{ord(character):04X} ({character})"
                    except TypeError: base_text = f"Char: {character} (Error getting Unicode val)"
                else: base_text = f"Char: {character}"
        else: base_text = "Unicode: N/A"
        final_text = base_text
        if self.canvas and character: 
            if self.canvas.editing_vrt2_glyph: final_text += " vert" 
            elif character != '.notdef' and character in self.rotated_vrt2_chars: final_text += " vert-r" 
        self.unicode_label.setText(final_text)

    def set_enabled_controls(self, enabled: bool):
        self.pen_button.setEnabled(enabled); self.eraser_button.setEnabled(enabled)
        self.move_button.setEnabled(enabled); self.slider.setEnabled(enabled)
        if hasattr(self, 'pen_size_buttons_group'): 
            for button in self.pen_size_buttons_group.findChildren(QPushButton): button.setEnabled(enabled)
        self.shape_box.setEnabled(enabled); self.mirror_checkbox.setEnabled(enabled)
        self.margin_slider.setEnabled(enabled); self.margin_value_label.setEnabled(enabled)
        self.ref_opacity_slider.setEnabled(enabled); self.ref_opacity_label.setEnabled(enabled)
        self.unicode_label.setEnabled(enabled) 
        is_vrt2_currently = self.canvas.editing_vrt2_glyph if self.canvas else False
        adv_controls_enabled = enabled 
        self.adv_width_slider.setEnabled(adv_controls_enabled); self.adv_width_spinbox.setEnabled(adv_controls_enabled)
        self.adv_width_label.setText("文字送り高さ:" if (enabled and is_vrt2_currently) else "文字送り幅:")
        self.load_ref_button.setEnabled(enabled) 
        self.delete_ref_button.setEnabled(enabled and self.canvas.reference_image is not None)
        if hasattr(self, 'glyph_to_ref_reset_button'):
             current_glyph_image = self.canvas.image
             current_glyph_has_content = (current_glyph_image and not current_glyph_image.isNull() and
                                          not (current_glyph_image.width() == 1 and current_glyph_image.height() == 1 and 
                                               current_glyph_image.pixelColor(0,0) == QColor(Qt.white).rgba())) 
             self.glyph_to_ref_reset_button.setEnabled(enabled and current_glyph_has_content)
        is_vrt2_widget_visible_and_char_eligible = self.vrt2_controls_widget.isVisible() 
        self.vrt2_toggle_button.setEnabled(enabled and is_vrt2_widget_visible_and_char_eligible)
        self.transfer_to_vrt2_button.setEnabled(enabled and is_vrt2_widget_visible_and_char_eligible)
        if enabled: self._update_undo_redo_buttons_state(len(self.canvas.undo_stack) > 1, bool(self.canvas.redo_stack))
        else: 
            self._update_undo_redo_buttons_state(False, False)
            self._update_adv_width_ui_no_signal(DEFAULT_ADVANCE_WIDTH) 
            self.canvas.reference_image = None 
            self._update_ref_opacity_slider_no_signal(DEFAULT_REFERENCE_IMAGE_OPACITY)
            self.canvas.set_reference_image_opacity(DEFAULT_REFERENCE_IMAGE_OPACITY)
        if not enabled: 
            self.canvas.current_glyph_character = None 
            inactive_canvas_fill_color = QColor(220, 220, 220) 
            self.canvas.image.fill(inactive_canvas_fill_color)
            self.canvas.update(); self.update_unicode_display(None)

    def _update_undo_redo_buttons_state(self, can_undo: bool, can_redo: bool):
        controls_are_generally_enabled = self.pen_button.isEnabled() 
        self.undo_button.setEnabled(can_undo and controls_are_generally_enabled)
        self.redo_button.setEnabled(can_redo and controls_are_generally_enabled)

    def apply_gui_settings(self, settings: Dict[str, Any]):
        pen_width_str = settings.get(SETTING_PEN_WIDTH)
        if pen_width_str is not None:
            try: self.canvas.set_pen_width(int(pen_width_str))
            except ValueError: pass 
        pen_shape = settings.get(SETTING_PEN_SHAPE)
        if pen_shape is not None:
            self.shape_box.blockSignals(True); self.shape_box.setCurrentText(pen_shape); self.shape_box.blockSignals(False)
            self.canvas.set_pen_shape(pen_shape)
        current_tool = settings.get(SETTING_CURRENT_TOOL)
        if current_tool == "eraser": self.canvas.set_eraser_mode()
        elif current_tool == "move": self.canvas.set_move_mode(True) 
        else: self.canvas.set_brush_mode() 
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
                self._update_ref_opacity_slider_no_signal(DEFAULT_REFERENCE_IMAGE_OPACITY)
                self.canvas.set_reference_image_opacity(DEFAULT_REFERENCE_IMAGE_OPACITY)
        else: 
            self._update_ref_opacity_slider_no_signal(DEFAULT_REFERENCE_IMAGE_OPACITY)
            self.canvas.set_reference_image_opacity(DEFAULT_REFERENCE_IMAGE_OPACITY)

class ImageLoaderSignals(QObject):
    image_loaded = Signal(str, QPixmap) # character_key, pixmap
    load_failed = Signal(str)          # character_key

class ImageLoadWorker(QRunnable):
    def __init__(self, db_path: str, character_key: str, is_vrt2: bool, db_manager: DatabaseManager):
        super().__init__()
        self.db_path = db_path # Though db_manager is passed, path might be useful for direct conn if needed
        self.character_key = character_key
        self.is_vrt2 = is_vrt2
        self.signals = ImageLoaderSignals()
        self.db_manager = db_manager # Use the main app's db_manager instance

    @Slot()
    def run(self):
        try:
            img_bytes = self.db_manager.load_glyph_image_bytes(self.character_key, self.is_vrt2)

            if img_bytes:
                pixmap = QPixmap()
                if pixmap.loadFromData(img_bytes):
                    self.signals.image_loaded.emit(self.character_key, pixmap)
                    return
            self.signals.load_failed.emit(self.character_key)
        except Exception as e:
            self.signals.load_failed.emit(self.character_key)

# --- GlyphTableModel (New, based on dbtest.py and main.py needs) ---
class GlyphTableModel(QAbstractTableModel):
    def __init__(self, db_manager: DatabaseManager, is_vrt2_tab: bool, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._db_manager = db_manager
        self._is_vrt2_tab = is_vrt2_tab
        
        # (char_key, display_char, has_initial_image_from_db)
        self._glyph_metadata_all: List[Tuple[str, str, bool]] = []
        # Indices into _glyph_metadata_all for currently visible items
        self._filtered_indices: List[int] = []
        
        self._pixmap_cache: Dict[str, QPixmap] = {} # char_key -> QPixmap
        self._requested_to_load: Set[str] = set()   # char_key currently being loaded
        
        self._column_count = DELEGATE_GRID_COLUMNS
        self.thread_pool = QThreadPool() # Each model instance can have its own pool
        self.thread_pool.setMaxThreadCount(max(1, QThread.idealThreadCount() // 2)) # Conservative thread count

        self._show_written_only = False
        self.non_rotated_vrt2_chars_for_highlight: Set[str] = set() # For standard tab highlight

    def set_character_data(self, char_data: List[Tuple[str, bool]]): # List of (char, has_initial_image)
        self.beginResetModel()
        self._glyph_metadata_all = []
        self._pixmap_cache.clear()
        self._requested_to_load.clear()

        for char_key, has_initial_image in char_data:
            # display_char is same as char_key for this application
            self._glyph_metadata_all.append((char_key, char_key, has_initial_image))
        
        self._rebuild_filtered_indices()
        self.endResetModel()

    def set_filter_written_only(self, written_only: bool):
        if self._show_written_only != written_only:
            self._show_written_only = written_only
            self.beginResetModel()
            self._rebuild_filtered_indices()
            self.endResetModel()
            
    def _rebuild_filtered_indices(self):
        self._filtered_indices = []
        for i, (char_key, _, has_initial_image) in enumerate(self._glyph_metadata_all):
            if not self._show_written_only:
                self._filtered_indices.append(i)
            else: # Show written only
                # Written if it initially had an image OR if it's in pixmap_cache (loaded/drawn later)
                if has_initial_image or char_key in self._pixmap_cache:
                    self._filtered_indices.append(i)
    
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid(): return 0
        return (len(self._filtered_indices) + self._column_count - 1) // self._column_count

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid(): return 0
        return self._column_count

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole) -> Any:
        if not index.isValid(): return None
        
        flat_idx_in_filtered = index.row() * self._column_count + index.column()
        if flat_idx_in_filtered >= len(self._filtered_indices): return None
        
        metadata_idx = self._filtered_indices[flat_idx_in_filtered]
        char_key, display_char, _ = self._glyph_metadata_all[metadata_idx]

        if role == Qt.DisplayRole: return display_char # For accessibility or tooltip if needed
        elif role == Qt.UserRole: return char_key     # The actual character string (key)
        elif role == Qt.UserRole + 1: # Pixmap data
            if char_key in self._pixmap_cache:
                return self._pixmap_cache[char_key]
            else:
                if char_key not in self._requested_to_load:
                    self._request_image_load(char_key)
                return None # Placeholder will be drawn by delegate
        elif role == Qt.UserRole + 2: # Is VRT2 highlight needed (for standard tab)?
            return (not self._is_vrt2_tab) and (char_key in self.non_rotated_vrt2_chars_for_highlight)
        return None

    def _request_image_load(self, char_key: str):
        if not self._db_manager.db_path: return # No DB path, cannot load
        self._requested_to_load.add(char_key)
        worker = ImageLoadWorker(self._db_manager.db_path, char_key, self._is_vrt2_tab, self._db_manager)
        worker.signals.image_loaded.connect(self._on_image_loaded)
        worker.signals.load_failed.connect(self._on_image_load_failed)
        self.thread_pool.start(worker)

    @Slot(str, QPixmap)
    def _on_image_loaded(self, char_key: str, pixmap: QPixmap):
        self._pixmap_cache[char_key] = pixmap
        if char_key in self._requested_to_load:
            self._requested_to_load.remove(char_key)

        # Find if this char_key is currently visible and get its QModelIndex
        try:
            # Find original index in _glyph_metadata_all
            original_idx_in_all = -1
            for i, (ck, _, _) in enumerate(self._glyph_metadata_all):
                if ck == char_key:
                    original_idx_in_all = i
                    break
            
            if original_idx_in_all == -1: return # Should not happen

            # If filter is on and item's visibility might have changed
            if self._show_written_only:
                # Check if item *was not* in _filtered_indices but *now should be*
                # This happens if has_initial_image was false, but now pixmap is loaded.
                _, _, has_initial_image = self._glyph_metadata_all[original_idx_in_all]
                was_visible_due_to_initial = has_initial_image 
                
                is_now_in_filtered_indices = False
                current_flat_idx_in_filtered = -1
                for i_filt, meta_idx in enumerate(self._filtered_indices):
                    if meta_idx == original_idx_in_all:
                        is_now_in_filtered_indices = True
                        current_flat_idx_in_filtered = i_filt
                        break

                if not was_visible_due_to_initial and not is_now_in_filtered_indices: # Was not visible, now should be
                    self.beginResetModel()
                    self._rebuild_filtered_indices() # Rebuild and emit reset
                    self.endResetModel()
                    return 
                elif is_now_in_filtered_indices : # Was already visible, just update its cell
                     row = current_flat_idx_in_filtered // self._column_count
                     col = current_flat_idx_in_filtered % self._column_count
                     model_idx = self.index(row, col)
                     if model_idx.isValid():
                         self.dataChanged.emit(model_idx, model_idx, [Qt.UserRole + 1])
                # else: item became visible but filter rebuild handled it or item still not visible
            
            else: # Filter is off, item is always visible if in _glyph_metadata_all
                if original_idx_in_all in self._filtered_indices: # Should always be true if filter off
                    flat_idx = self._filtered_indices.index(original_idx_in_all)
                    row = flat_idx // self._column_count
                    col = flat_idx % self._column_count
                    model_idx = self.index(row, col)
                    if model_idx.isValid():
                        self.dataChanged.emit(model_idx, model_idx, [Qt.UserRole + 1])
        except ValueError: # char_key not found in _filtered_indices.index, means it's not visible
            pass # No UI update needed if item not visible


    @Slot(str)
    def _on_image_load_failed(self, char_key: str):
        if char_key in self._requested_to_load:
            self._requested_to_load.remove(char_key)
        # Option: could store a "failed_to_load" marker in cache to prevent re-request spam
        # For now, just removes from requested. If cell becomes visible again, it will re-request.

    def update_glyph_pixmap(self, char_key: str, pixmap: Optional[QPixmap]):
        # Called when a glyph is saved (potentially changing from None to Pixmap or vice versa)
        _, _, initial_has_image = next((m for m in self._glyph_metadata_all if m[0] == char_key), (None,None,False))

        if pixmap is None: # Image was deleted (e.g., became blank)
            if char_key in self._pixmap_cache:
                del self._pixmap_cache[char_key]
        else: # Image was updated/created
            self._pixmap_cache[char_key] = pixmap

        if self._show_written_only:
            # If the filter is active, the "written" status might have changed,
            # potentially altering the number of items in the filtered list.
            # A full model reset is the safest way to handle this.
            self.beginResetModel()
            self._rebuild_filtered_indices() # This uses the now-updated _pixmap_cache
            self.endResetModel()
            return # Model reset takes care of UI updates.
        
        # If filter is not on, the item remains in the list; just its data changed.
        # Find the cell and emit dataChanged for the specific item.
        try:
            original_idx_in_all = -1
            for i, (ck, _, _) in enumerate(self._glyph_metadata_all):
                if ck == char_key:
                    original_idx_in_all = i
                    break
            
            if original_idx_in_all == -1:
                return # Character not found in metadata (should not happen if logic is correct)

            # Since filter is off, _filtered_indices contains all indices from _glyph_metadata_all
            # or at least, original_idx_in_all should map directly to its position if it exists.
            # A more direct way if filter is off: flat_idx = original_idx_in_all
            if original_idx_in_all < len(self._filtered_indices): # Check if it's within current possibly filtered (though here assumed not)
                flat_idx = self._filtered_indices.index(original_idx_in_all) # Find its position in the *filtered* list
                row, col = divmod(flat_idx, self._column_count)
                model_idx = self.index(row, col)
                if model_idx.isValid():
                    self.dataChanged.emit(model_idx, model_idx, [Qt.UserRole + 1])
        except ValueError:
            # This might happen if original_idx_in_all is somehow not in _filtered_indices
            # even when _show_written_only is false. This indicates a mismatch.
            # However, if _show_written_only is false, _filtered_indices should map 1:1 to _glyph_metadata_all.
            pass

    def get_char_key_at_flat_index(self, flat_idx: int) -> Optional[str]:
        if 0 <= flat_idx < len(self._filtered_indices):
            metadata_idx = self._filtered_indices[flat_idx]
            return self._glyph_metadata_all[metadata_idx][0]
        return None

    def get_flat_index_of_char_key(self, char_key: str) -> Optional[int]:
        for flat_idx, metadata_idx in enumerate(self._filtered_indices):
            if self._glyph_metadata_all[metadata_idx][0] == char_key:
                return flat_idx
        return None

    def get_metadata_count(self) -> int: # Number of items currently displayed by model
        return len(self._filtered_indices)

    def is_glyph_written(self, char_key: str) -> bool:
        """Checks if a glyph is considered 'written' (has an initial image or is cached)."""
        if char_key in self._pixmap_cache: # If in cache, it has a pixmap (so it's written)
            return True
        # Check initial state from metadata
        for ck, _, has_initial_image in self._glyph_metadata_all:
            if ck == char_key:
                return has_initial_image
        return False # Not found in metadata, or no image


# --- GlyphTableDelegate (New, based on dbtest.py and main.py needs) ---
class GlyphTableDelegate(QStyledItemDelegate):
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.char_label_font = QFont("Arial", 10) # Or use application default font
        fm = QFontMetrics(self.char_label_font)
        self.char_label_height = fm.height() + 2 * DELEGATE_CHAR_LABEL_PADDING
        
        # To be set by GlyphGridWidget if this delegate is for the standard glyphs tab
        self.non_rotated_vrt2_chars: Set[str] = set()

    def _get_adjusted_color(self, base_color: QColor, factor_light: int, factor_dark: int, palette: QPalette) -> QColor:
        current_is_dark = palette.color(QPalette.ColorRole.Window).lightnessF() < 0.5
        return base_color.darker(factor_dark) if current_is_dark else base_color.lighter(factor_light)

    def _paint_frame_effect(self, painter: QPainter, rect: QRect, palette: QPalette, sunken: bool):
        pen = QPen()
        pen.setWidth(DELEGATE_FRAME_BORDER_WIDTH)
        
        # Use colors from the palette for a more theme-consistent look
        light_color = palette.color(QPalette.ColorRole.Light)
        dark_color = palette.color(QPalette.ColorRole.Dark)
        mid_color = palette.color(QPalette.ColorRole.Mid)

        if sunken:
            top_left_color = dark_color
            bottom_right_color = light_color
        else: # Raised
            top_left_color = light_color
            bottom_right_color = dark_color

        painter.setPen(QPen(top_left_color, DELEGATE_FRAME_BORDER_WIDTH))
        painter.drawLine(rect.topLeft(), rect.topRight())
        painter.drawLine(rect.topLeft(), rect.bottomLeft())

        painter.setPen(QPen(bottom_right_color, DELEGATE_FRAME_BORDER_WIDTH))
        # Adjust for pixel-perfect lines when width > 1, not strictly needed for width 1
        painter.drawLine(rect.topRight(), rect.bottomRight())
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex) -> None:
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        char_key = index.data(Qt.UserRole)       # Actual character string (key)
        pixmap_data = index.data(Qt.UserRole + 1) # QPixmap or None
        is_nrvg_highlight = index.data(Qt.UserRole + 2) # Bool or None

        if not char_key:
            painter.restore()
            return

        rect = option.rect
        palette = option.palette
        
        # Base background for the entire cell (respects view's alternating row colors if set)
        # QStyledItemDelegate.paint(self, painter, option, index) # This would draw default item bg + text
        # For full custom, fill it ourselves:
        painter.fillRect(rect, option.palette.color(QPalette.ColorRole.Base))


        # Item visual container
        item_rect = rect.adjusted(DELEGATE_ITEM_MARGIN, DELEGATE_ITEM_MARGIN, 
                                  -DELEGATE_ITEM_MARGIN, -DELEGATE_ITEM_MARGIN)
        if not item_rect.isValid():
            painter.restore()
            return

        # Background of the item itself (like the old QFrame)
        item_bg_color = palette.color(QPalette.ColorRole.Button) # A common color for unselected items
        if option.state & QStyle.State_Selected:
            item_bg_color = palette.color(QPalette.ColorRole.Highlight).lighter(110) # Slightly lighter highlight
        
        painter.fillRect(item_rect, item_bg_color)


        # Frame effect (raised or sunken for selection)
        frame_rect = item_rect.adjusted(0,0,-1,-1) # For pixel perfect border
        self._paint_frame_effect(painter, frame_rect, palette, sunken=(option.state & QStyle.State_Selected))

        # If selected, draw an additional outline for more emphasis
        if option.state & QStyle.State_Selected:
            painter.setPen(QPen(palette.color(QPalette.ColorRole.Highlight), DELEGATE_SELECTION_OUTLINE_WIDTH))
            # Draw outline slightly inside the margin, or around item_rect
            painter.drawRect(item_rect.adjusted(0,0,-1,-1)) 
            # Content rect needs to be inset further if selection outline is thick
            content_offset = DELEGATE_FRAME_BORDER_WIDTH + DELEGATE_SELECTION_OUTLINE_WIDTH
        else:
            content_offset = DELEGATE_FRAME_BORDER_WIDTH
        
        content_draw_rect = item_rect.adjusted(content_offset + DELEGATE_CELL_CONTENT_PADDING,
                                               content_offset + DELEGATE_CELL_CONTENT_PADDING,
                                               -(content_offset + DELEGATE_CELL_CONTENT_PADDING),
                                               -(content_offset + DELEGATE_CELL_CONTENT_PADDING))
        if not content_draw_rect.isValid():
            painter.restore()
            return

        # Character Label
        char_label_actual_rect = QRect(content_draw_rect.left(), content_draw_rect.top(),
                                   content_draw_rect.width(), self.char_label_height)
        
        label_bg_color = self._get_adjusted_color(palette.color(QPalette.ColorRole.Window), 105, 110, palette)
        label_text_color = palette.color(QPalette.ColorRole.Text)

        if option.state & QStyle.State_Selected:
            label_bg_color = palette.color(QPalette.ColorRole.Highlight)
            label_text_color = palette.color(QPalette.ColorRole.HighlightedText)
        elif is_nrvg_highlight: # Standard glyph that is also in NR-VRT2 set (highlight)
            label_bg_color = VRT2_PREVIEW_BACKGROUND_TINT.lighter(110)
            # Text color for VRT2 highlight can be adjusted for contrast
            label_text_color = Qt.black if VRT2_PREVIEW_BACKGROUND_TINT.lightnessF() > 0.5 else Qt.white


        painter.setBrush(label_bg_color)
        label_border_color = self._get_adjusted_color(label_bg_color, 115, 125, palette)
        painter.setPen(QPen(label_border_color, 1))
        painter.drawRoundedRect(char_label_actual_rect, DELEGATE_CHAR_LABEL_BORDER_RADIUS, DELEGATE_CHAR_LABEL_BORDER_RADIUS)
        
        painter.setPen(label_text_color)
        painter.setFont(self.char_label_font)
        text_draw_rect = char_label_actual_rect.adjusted(DELEGATE_CHAR_LABEL_PADDING, 0, -DELEGATE_CHAR_LABEL_PADDING, 0)
        # Truncate display_char if too long for label (though char_key is usually single char)
        display_text = char_key if len(char_key) < 5 else char_key[:4] + "…"
        painter.drawText(text_draw_rect, Qt.AlignCenter | Qt.TextSingleLine, display_text)

        # Preview Area
        preview_top_y = char_label_actual_rect.bottom() + DELEGATE_CELL_CONTENT_PADDING
        available_preview_width = content_draw_rect.width()
        available_preview_height = content_draw_rect.bottom() - preview_top_y
        
        if available_preview_width > 0 and available_preview_height > 0:
            preview_square_dim = min(available_preview_width, available_preview_height)
            preview_area_rect = QRect(
                content_draw_rect.left() + (available_preview_width - preview_square_dim) // 2,
                preview_top_y + (available_preview_height - preview_square_dim) // 2,
                preview_square_dim,
                preview_square_dim
            )

            preview_bg_color = palette.color(QPalette.ColorRole.Window)
            if is_nrvg_highlight and not (option.state & QStyle.State_Selected): # Apply tint if not selected
                 preview_bg_color = VRT2_PREVIEW_BACKGROUND_TINT


            painter.setBrush(preview_bg_color)
            preview_border_color = self._get_adjusted_color(palette.color(QPalette.ColorRole.Mid), 100, 100, palette)
            painter.setPen(QPen(preview_border_color, 1))
            painter.drawRect(preview_area_rect)

            if pixmap_data and not pixmap_data.isNull():
                target_pixmap_rect = preview_area_rect.adjusted(DELEGATE_PREVIEW_PADDING, DELEGATE_PREVIEW_PADDING,
                                                             -DELEGATE_PREVIEW_PADDING, -DELEGATE_PREVIEW_PADDING)
                if target_pixmap_rect.isValid() and target_pixmap_rect.width() > 0:
                    scaled_pixmap = pixmap_data.scaled(target_pixmap_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                    px = target_pixmap_rect.left() + (target_pixmap_rect.width() - scaled_pixmap.width()) / 2
                    py = target_pixmap_rect.top() + (target_pixmap_rect.height() - scaled_pixmap.height()) / 2
                    painter.drawPixmap(QPoint(int(px), int(py)), scaled_pixmap)
            else: # Placeholder if no image loaded / load failed
                painter.fillRect(preview_area_rect, DELEGATE_PLACEHOLDER_COLOR)
                if preview_area_rect.width() > 10 and preview_area_rect.height() > 10:
                    loading_font = QFont(self.char_label_font.family(), 8) # Smaller font
                    painter.setFont(loading_font)
                    painter.setPen(palette.color(QPalette.ColorRole.Mid)) # Muted color
                    painter.drawText(preview_area_rect, Qt.AlignCenter, "...")
        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        # Calculate height based on char label and preview area (similar to old GlyphItemWidget)
        # Assume a fixed base width (DELEGATE_CELL_BASE_WIDTH) and calculate height
        # This should match QTableView.setColumnWidth and .verticalHeader().setDefaultSectionSize()
        
        preview_min_dim = DEFAULT_GLYPH_PREVIEW_SIZE.height() # Based on a common preview size
        
        total_vertical_padding = (2 * DELEGATE_ITEM_MARGIN) + \
                                 (2 * DELEGATE_FRAME_BORDER_WIDTH) + \
                                 (2 * DELEGATE_CELL_CONTENT_PADDING) + \
                                 DELEGATE_CELL_CONTENT_PADDING # (label-preview space) +
                                 # (2 * DELEGATE_PREVIEW_PADDING) # Inside preview area
        
        # If selected, selection outline might take space
        selection_bonus = 2 * DELEGATE_SELECTION_OUTLINE_WIDTH if (option.state & QStyle.State_Selected) else 0
        
        cell_height = self.char_label_height + preview_min_dim + total_vertical_padding + selection_bonus
        return QSize(DELEGATE_CELL_BASE_WIDTH, int(cell_height))







# --- GlyphGridWidget の修正箇所 ---
# --- GlyphGridWidget の修正箇所 ---
class GlyphGridWidget(QWidget):
    glyph_selected_signal = Signal(str)
    vrt2_glyph_selected_signal = Signal(str)

    def __init__(self, db_manager: DatabaseManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._db_manager = db_manager
        self.non_rotated_vrt2_chars: Set[str] = set()
        self._is_selecting_programmatically = False # 再帰防止フラグ

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(5)

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

        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget, 1)

        self.std_glyph_view = QTableView()
        self.std_glyph_model = GlyphTableModel(self._db_manager, is_vrt2_tab=False, parent=self)
        self.std_glyph_delegate = GlyphTableDelegate(self.std_glyph_view)
        self._setup_table_view(self.std_glyph_view, self.std_glyph_model, self.std_glyph_delegate)
        self.tab_widget.addTab(self.std_glyph_view, "標準グリフ")

        self.vrt2_glyph_view = QTableView()
        self.vrt2_glyph_model = GlyphTableModel(self._db_manager, is_vrt2_tab=True, parent=self)
        self.vrt2_glyph_delegate = GlyphTableDelegate(self.vrt2_glyph_view)
        self._setup_table_view(self.vrt2_glyph_view, self.vrt2_glyph_model, self.vrt2_glyph_delegate)
        self.tab_widget.addTab(self.vrt2_glyph_view, "縦書きグリフ")

        self.tab_widget.currentChanged.connect(self._on_tab_changed)
        self.current_active_view = self.std_glyph_view
        # self.current_active_view.setFocus() # 初期フォーカスは MainWindow で制御することがあるため、ここからは削除検討

        grid_width = DELEGATE_GRID_COLUMNS * DELEGATE_CELL_BASE_WIDTH + \
                     (DELEGATE_GRID_COLUMNS * DELEGATE_ITEM_MARGIN * 2) + \
                     2
        self.setFixedWidth(grid_width)
        self.setFocusPolicy(Qt.StrongFocus) # GlyphGridWidget 自体もフォーカスを受け取れるように

    def _setup_table_view(self, view: QTableView, model: GlyphTableModel, delegate: GlyphTableDelegate):
        view.setModel(model)
        view.setItemDelegate(delegate)
        view.horizontalHeader().hide()
        view.verticalHeader().hide()
        view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectItems)
        view.setShowGrid(False)
        view.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        view.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

        temp_option = QStyleOptionViewItem()
        temp_option_selected = QStyleOptionViewItem()
        temp_option_selected.state = QStyle.State_Selected

        initial_cell_size_normal = delegate.sizeHint(temp_option, QModelIndex())
        initial_cell_size_selected = delegate.sizeHint(temp_option_selected, QModelIndex())

        actual_cell_width = max(initial_cell_size_normal.width(), initial_cell_size_selected.width())
        actual_cell_height = max(initial_cell_size_normal.height(), initial_cell_size_selected.height())

        for i in range(model.columnCount()):
            view.setColumnWidth(i, actual_cell_width)
        view.verticalHeader().setDefaultSectionSize(actual_cell_height)

        view.clicked.connect(self._on_item_clicked)
        view.selectionModel().currentChanged.connect(self._on_current_item_changed_in_view)

        view.setStyleSheet("""
            QTableView { background-color: palette(window); border: 1px solid palette(mid); gridline-color: transparent; }
            QTableView::item { border: none; padding: 0px; margin: 0px; background-color: transparent; }
            QTableView::item:selected { background-color: transparent; }
        """)
        view.setFocusPolicy(Qt.StrongFocus) # 各TableViewもフォーカスを受け取れるように

        # イベントフィルターをインストール
        view.installEventFilter(self)


    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.KeyPress:
            if watched is self.std_glyph_view or watched is self.vrt2_glyph_view:
                key_event = QKeyEvent(event) 
                if key_event.key() in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]:
                    if key_event.modifiers() == Qt.NoModifier or key_event.modifiers() == Qt.KeypadModifier : 
                        self.keyPressEvent(key_event) 
                        if key_event.isAccepted():
                            return True 
                        else:
                            # If custom logic didn't accept, but we still want to consume to prevent default QTableView nav
                            return True # Or False if you want QTableView's default to run if yours doesn't accept
                    else:
                        return False # Allow QTableView default for modified arrow keys
                else:
                     return False # Allow QTableView default for non-arrow keys
        return super().eventFilter(watched, event)

    def set_search_and_filter_enabled(self, enabled: bool):
        self.search_input.setEnabled(enabled)
        self.search_button.setEnabled(enabled)
        self.show_written_only_checkbox.setEnabled(enabled)
        self.tab_widget.setEnabled(enabled)

    def _perform_search(self):
        search_text = self.search_input.text().strip()
        if not search_text: return

        char_to_find = search_text[0]
        if char_to_find == '.' and len(search_text) > 1 and search_text.lower() == ".notdef":
            char_to_find = ".notdef"
        
        target_model = self.std_glyph_model if self.current_active_view == self.std_glyph_view else self.vrt2_glyph_model
        target_view = self.current_active_view
        
        flat_idx = target_model.get_flat_index_of_char_key(char_to_find)
        if flat_idx is not None:
            self._select_char_in_view_and_emit(char_to_find, target_view, target_model) # 変更: emitを伴う選択
            self.search_input.clear()
            return
        
        QMessageBox.information(self, "検索結果", f"文字 '{search_text}' は現在表示中のリストに見つかりません。")

    def _on_filter_changed(self, checked: bool):
        self.std_glyph_model.set_filter_written_only(checked)
        self.vrt2_glyph_model.set_filter_written_only(checked)
        self._try_to_restore_selection_or_select_first(self.current_active_view)

    def _on_tab_changed(self, index: int):
        if self._is_selecting_programmatically:
            return

        if index == 0:
            self.current_active_view = self.std_glyph_view
        else:
            self.current_active_view = self.vrt2_glyph_view
        
        self.current_active_view.setFocus() 
        self.search_input.clear()
        self._try_to_restore_selection_or_select_first(self.current_active_view)

    def _on_item_clicked(self, index: QModelIndex):
        if self._is_selecting_programmatically or not index.isValid():
            return
        
        view = self.sender()
        if not isinstance(view, QTableView):
            return
        
        view.setFocus() # ★★★ クリックされたビューにフォーカスをセット ★★★
        
        model = view.model()
        if not isinstance(model, GlyphTableModel):
            return
            
        char_key = model.data(index, Qt.UserRole)
        if char_key:
            is_vrt2_source = (view == self.vrt2_glyph_view)
            self._emit_selection_signal(char_key, is_vrt2_source)


    def _on_current_item_changed_in_view(self, current: QModelIndex, previous: QModelIndex):
        if self._is_selecting_programmatically:
            return

        active_view = self.current_active_view 
        model = active_view.model() # current_active_view のモデルを使用
        if not isinstance(model, GlyphTableModel):
            return

        if not current.isValid(): # 新しいカレントインデックスが無効な場合
            is_vrt2_source_for_clear = (active_view == self.vrt2_glyph_view)
            self._emit_selection_signal("", is_vrt2_source_for_clear)
            return
        
        # current QModelIndex から char_key を取得
        char_key_from_current_idx = model.data(current, Qt.UserRole)
        if char_key_from_current_idx: # 新しいカレントインデックスから文字キーが取れた場合
            is_vrt2_source = (active_view == self.vrt2_glyph_view)
            self._emit_selection_signal(char_key_from_current_idx, is_vrt2_source)


    def _emit_selection_signal(self, char_key: str, is_vrt2_source: bool):
        if is_vrt2_source:
            self.vrt2_glyph_selected_signal.emit(char_key)
        else:
            self.glyph_selected_signal.emit(char_key)


    def set_non_rotated_vrt2_chars(self, nrvg_chars: Set[str]):
        self.non_rotated_vrt2_chars = nrvg_chars
        self.std_glyph_model.non_rotated_vrt2_chars_for_highlight = nrvg_chars
        self.std_glyph_delegate.non_rotated_vrt2_chars = nrvg_chars 
        if self.std_glyph_view.model() is not None: 
            self.std_glyph_view.viewport().update()

    def clear_grid_and_models(self):
        self.std_glyph_model.set_character_data([])
        self.vrt2_glyph_model.set_character_data([])

    def populate_models(self, 
                        std_char_data: List[Tuple[str, bool]], 
                        vrt2_char_data: List[Tuple[str, bool]]):
        self._is_selecting_programmatically = True # モデル変更中はシグナル発行を抑制
        try:
            self.std_glyph_model.set_character_data(std_char_data)
            self.vrt2_glyph_model.set_character_data(vrt2_char_data)
        finally:
            self._is_selecting_programmatically = False

        # populate直後の選択は MainWindow の _select_initial_glyph_after_full_load で行う
        # ここでは何もしないか、あるいは _try_to_restore_selection_or_select_first を呼ぶ
        # ただし、_try_to_restore はシグナルを発行する可能性があるため、
        # 初期ロード時は MainWindow 側で制御するのが望ましい。
        # self._try_to_restore_selection_or_select_first(self.current_active_view)

    def set_active_glyph(self, character: Optional[str], is_vrt2_source: bool = False):
        """
        指定されたグリフをアクティブにし、表示を更新する。
        このメソッドは主にMainWindowから呼び出され、グリッドの表示を同期させる。
        シグナルは発行しない（循環呼び出しを避けるため）。
        """
        if self._is_selecting_programmatically: 
            return

        self._is_selecting_programmatically = True
        try:
            target_view = self.vrt2_glyph_view if is_vrt2_source else self.std_glyph_view
            target_model = self.vrt2_glyph_model if is_vrt2_source else self.std_glyph_model

            target_tab_index = 1 if is_vrt2_source else 0
            if self.tab_widget.currentIndex() != target_tab_index:
                # _on_tab_changed が呼ばれ、そこで current_active_view とフォーカスが設定される
                self.tab_widget.setCurrentIndex(target_tab_index) 
            else:
                # タブが変わらない場合でも、current_active_view を更新し、フォーカスを当てる
                self.current_active_view = target_view
                # target_view.setFocus() # ★★★ フォーカスを設定 ★★★
                # → _on_tab_changed が呼ばれない場合、ここでフォーカスが必要

            if character is None or not character:
                target_view.selectionModel().clear()
                # target_view.setFocus() # フォーカスはタブ変更時かアイテムクリック時に
                self._is_selecting_programmatically = False # 早期リターンの前にフラグを解除
                return

            flat_idx = target_model.get_flat_index_of_char_key(character)
            if flat_idx is not None:
                row, col = divmod(flat_idx, target_model.columnCount())
                q_model_idx = target_model.index(row, col)
                if q_model_idx.isValid():
                    if target_view.currentIndex() != q_model_idx:
                         target_view.selectionModel().setCurrentIndex(q_model_idx, QItemSelectionModel.ClearAndSelect)
                    target_view.scrollTo(q_model_idx, QAbstractItemView.ScrollHint.EnsureVisible)
                else:
                    target_view.selectionModel().clear()
            else:
                target_view.selectionModel().clear()
            
            # 最終的にフォーカスを当てるのは、実際に操作対象となるビュー
            # self.current_active_view.setFocus() # current_active_view はタブ変更で更新される
            # タブ変更がない場合は target_view が current_active_view と同じはず
            if self.tab_widget.currentIndex() == target_tab_index: # タブ変更がなかった場合
                 target_view.setFocus()
            # タブ変更があった場合は、_on_tab_changed でフォーカスが設定される

        finally:
            self._is_selecting_programmatically = False


    def _select_char_in_view_and_emit(self, character: Optional[str], view: QTableView, model: GlyphTableModel):
        """
        指定された文字を選択し、必要であれば選択シグナルを発行する。
        主にユーザー操作（キー入力、検索）から呼び出される。
        """
        if self._is_selecting_programmatically:
            return

        self._is_selecting_programmatically = True
        try:
            current_selection_model = view.selectionModel()
            if not current_selection_model:
                return

            old_idx = current_selection_model.currentIndex()
            old_char_key = model.data(old_idx, Qt.UserRole) if old_idx.isValid() else None

            new_q_model_idx = QModelIndex() # デフォルトは無効なインデックス

            if character is None or not character:
                if old_idx.isValid(): # 実際に選択がクリアされる場合のみ
                    current_selection_model.clear()
                    # _on_current_item_changed_in_view が処理する
            else:
                flat_idx = model.get_flat_index_of_char_key(character)
                if flat_idx is not None:
                    row, col = divmod(flat_idx, model.columnCount())
                    q_model_idx = model.index(row, col)
                    if q_model_idx.isValid():
                        new_q_model_idx = q_model_idx
                        if old_idx != new_q_model_idx:
                             current_selection_model.setCurrentIndex(new_q_model_idx, QItemSelectionModel.ClearAndSelect)
                        view.scrollTo(new_q_model_idx, QAbstractItemView.ScrollHint.EnsureVisible)
                    else: # モデルインデックスが無効
                        if old_idx.isValid(): current_selection_model.clear()
                else: # 文字が見つからない
                    if old_idx.isValid(): current_selection_model.clear()
            
            view.setFocus()
            # setCurrentIndex または clear を呼ぶと _on_current_item_changed_in_view がトリガーされ、
            # そこで _emit_selection_signal が呼ばれるので、ここでは直接呼ばない。
        finally:
            self._is_selecting_programmatically = False


    def _try_to_restore_selection_or_select_first(self, view_to_update: QTableView):
        if self._is_selecting_programmatically:
            return

        model_to_update = view_to_update.model()
        if not isinstance(model_to_update, GlyphTableModel): return

        current_selection_model_idx = view_to_update.currentIndex()
        current_char_key: Optional[str] = None
        
        if current_selection_model_idx.isValid():
            current_char_key = model_to_update.data(current_selection_model_idx, Qt.UserRole)
        
        if current_char_key and model_to_update.get_flat_index_of_char_key(current_char_key) is not None:
            self._select_char_in_view_and_emit(current_char_key, view_to_update, model_to_update)
        elif model_to_update.get_metadata_count() > 0:
            first_item_char_key = model_to_update.get_char_key_at_flat_index(0)
            if first_item_char_key:
                 self._select_char_in_view_and_emit(first_item_char_key, view_to_update, model_to_update)
        else: 
            self._select_char_in_view_and_emit(None, view_to_update, model_to_update)


    # ... (update_glyph_preview, get_first_navigable_glyph_info, get_navigable_glyphs_info_for_active_tab は変更なし) ...
    def update_glyph_preview(self, character: str, pixmap: Optional[QPixmap], is_vrt2_source: bool):
        model_to_update = self.vrt2_glyph_model if is_vrt2_source else self.std_glyph_model
        model_to_update.update_glyph_pixmap(character, pixmap)

    def get_first_navigable_glyph_info(self) -> Optional[Tuple[str, bool]]:
        # このメソッドは外部から呼ばれるので、現在のUI状態に基づいて情報を返す
        active_view_for_info = self.current_active_view # タブ変更直後でも正しいビューを参照
        active_model = active_view_for_info.model()
        is_active_vrt2 = (active_view_for_info == self.vrt2_glyph_view)
        
        if isinstance(active_model, GlyphTableModel) and active_model.get_metadata_count() > 0:
            char_key = active_model.get_char_key_at_flat_index(0)
            if char_key:
                return char_key, is_active_vrt2
        
        other_view = self.vrt2_glyph_view if active_view_for_info == self.std_glyph_view else self.std_glyph_view
        other_model = other_view.model()
        is_other_vrt2 = (other_view == self.vrt2_glyph_view)
        if isinstance(other_model, GlyphTableModel) and other_model.get_metadata_count() > 0:
            char_key = other_model.get_char_key_at_flat_index(0)
            if char_key:
                return char_key, is_other_vrt2
        return None

    def get_navigable_glyphs_info_for_active_tab(self) -> List[Tuple[str, bool]]:
        nav_list: List[Tuple[str, bool]] = []
        active_model = self.current_active_view.model()
        is_vrt2_tab = (self.current_active_view == self.vrt2_glyph_view)
        if isinstance(active_model, GlyphTableModel):
            for i in range(active_model.get_metadata_count()): 
                char_key = active_model.get_char_key_at_flat_index(i)
                if char_key:
                    nav_list.append((char_key, is_vrt2_tab))
        return nav_list


    def keyPressEvent(self, event: QKeyEvent):
        if self._is_selecting_programmatically:
            event.ignore()
            return

        view = self.current_active_view
        model = view.model()
        if not isinstance(model, GlyphTableModel):
            super().keyPressEvent(event)
            return

        key = event.key()
        
        if event.modifiers() != Qt.NoModifier and event.modifiers() != Qt.KeypadModifier:
            if key in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]:
                super().keyPressEvent(event)
                if event.isAccepted():
                    return
                return
            else: 
                super().keyPressEvent(event)
                return
        
        if key not in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]:
            super().keyPressEvent(event)
            return

        current_model_idx = view.currentIndex()
        
        num_cols = model.columnCount()
        total_visible_items = model.get_metadata_count()

        if num_cols == 0 or total_visible_items == 0:
            super().keyPressEvent(event) 
            return

        current_flat_idx = -1
        if current_model_idx.isValid():
            current_row_from_idx = current_model_idx.row()
            current_col_from_idx = current_model_idx.column()
            current_flat_idx = current_row_from_idx * num_cols + current_col_from_idx
            
            if not (0 <= current_flat_idx < total_visible_items):
                current_flat_idx = -1 
        
        if current_flat_idx == -1: 
            if total_visible_items > 0:
                first_item_char_key = model.get_char_key_at_flat_index(0)
                if first_item_char_key:
                    self._select_char_in_view_and_emit(first_item_char_key, view, model)
                event.accept()
            else:
                super().keyPressEvent(event)
            return
        
        target_flat_idx = current_flat_idx 

        if key == Qt.Key_Left:
            if current_flat_idx > 0:
                target_flat_idx = current_flat_idx - 1
        
        elif key == Qt.Key_Right:
            if current_flat_idx < total_visible_items - 1:
                target_flat_idx = current_flat_idx + 1
        
        elif key == Qt.Key_Up:
            potential_target_idx = current_flat_idx - num_cols
            if potential_target_idx >= 0:
                target_flat_idx = potential_target_idx
        
        elif key == Qt.Key_Down:
            potential_target_idx = current_flat_idx + num_cols
            if potential_target_idx < total_visible_items:
                target_flat_idx = potential_target_idx

        if target_flat_idx != current_flat_idx:
            if 0 <= target_flat_idx < total_visible_items:
                char_to_select = model.get_char_key_at_flat_index(target_flat_idx)
                if char_to_select: 
                    self._select_char_in_view_and_emit(char_to_select, view, model)
        
        event.accept()
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
        font_name_layout = QHBoxLayout(); font_name_layout.addWidget(QLabel("フォント名:"))
        self.font_name_input = QLineEdit(); self.font_name_input.setPlaceholderText("例: MyCustomFont")
        self.font_name_input.editingFinished.connect(self._emit_font_name_change) 
        font_name_layout.addWidget(self.font_name_input); layout.addLayout(font_name_layout)
        font_weight_layout = QHBoxLayout(); font_weight_layout.addWidget(QLabel("ウェイト:"))
        self.font_weight_combobox = QComboBox(); self.font_weight_combobox.addItems(FONT_WEIGHT_OPTIONS)
        self.font_weight_combobox.setCurrentText(DEFAULT_FONT_WEIGHT) 
        self.font_weight_combobox.currentTextChanged.connect(self._emit_font_weight_change)
        font_weight_layout.addWidget(self.font_weight_combobox); font_weight_layout.addStretch(1) 
        layout.addLayout(font_weight_layout); layout.addSpacing(10)
        layout.addWidget(QLabel("プロジェクトの文字セット:"))
        self.char_set_text_edit = QTextEdit(); self.char_set_text_edit.setPlaceholderText("例: あいうえお漢字...")
        self.char_set_text_edit.setFixedHeight(200); layout.addWidget(self.char_set_text_edit)
        apply_char_set_button = QPushButton("文字セットを適用")
        apply_char_set_button.clicked.connect(self._apply_char_set_changes); layout.addWidget(apply_char_set_button)
        layout.addSpacing(10); layout.addWidget(QLabel("回転縦書きグリフ (単純回転):"))
        self.r_vrt2_text_edit = QTextEdit(); self.r_vrt2_text_edit.setPlaceholderText("例: （）「」…")
        self.r_vrt2_text_edit.setFixedHeight(60); layout.addWidget(self.r_vrt2_text_edit)
        apply_r_vrt2_button = QPushButton("回転縦書き文字セットを適用")
        apply_r_vrt2_button.clicked.connect(self._apply_r_vrt2_changes); layout.addWidget(apply_r_vrt2_button)
        layout.addSpacing(10); layout.addWidget(QLabel("非回転縦書き文字 (専用グリフ):"))
        self.nr_vrt2_text_edit = QTextEdit(); self.nr_vrt2_text_edit.setPlaceholderText("例: 、。‘’“”…")
        self.nr_vrt2_text_edit.setFixedHeight(60); layout.addWidget(self.nr_vrt2_text_edit)
        apply_nr_vrt2_button = QPushButton("非回転縦書き文字セットを適用")
        apply_nr_vrt2_button.clicked.connect(self._apply_nr_vrt2_changes); layout.addWidget(apply_nr_vrt2_button)
        layout.addSpacing(20) 
        self.export_font_button = QPushButton("フォントを書き出す")
        self.export_font_button.clicked.connect(self.export_font_signal); layout.addWidget(self.export_font_button)
        layout.addStretch(1) 

    def _emit_font_name_change(self): self.font_name_changed_signal.emit(self.font_name_input.text())
    def _emit_font_weight_change(self, weight_text: str): self.font_weight_changed_signal.emit(weight_text)
    def _apply_char_set_changes(self): self.character_set_changed_signal.emit(self.char_set_text_edit.toPlainText())
    def _apply_r_vrt2_changes(self): self.rotated_vrt2_set_changed_signal.emit(self.r_vrt2_text_edit.toPlainText())
    def _apply_nr_vrt2_changes(self): self.non_rotated_vrt2_set_changed_signal.emit(self.nr_vrt2_text_edit.toPlainText())

    def load_font_name(self, name: str):
        self.font_name_input.blockSignals(True); self.font_name_input.setText(name); self.font_name_input.blockSignals(False)
    def load_font_weight(self, weight: str):
        self.font_weight_combobox.blockSignals(True)
        if weight in FONT_WEIGHT_OPTIONS: self.font_weight_combobox.setCurrentText(weight)
        else: self.font_weight_combobox.setCurrentText(DEFAULT_FONT_WEIGHT) 
        self.font_weight_combobox.blockSignals(False)
    def load_character_set(self, char_string: str): self.char_set_text_edit.setText(char_string)
    def load_r_vrt2_set(self, char_string: str): self.r_vrt2_text_edit.setText(char_string)
    def load_nr_vrt2_set(self, char_string: str): self.nr_vrt2_text_edit.setText(char_string)
    def set_enabled_controls(self, enabled: bool):
        self.font_name_input.setEnabled(enabled); self.font_weight_combobox.setEnabled(enabled)
        self.char_set_text_edit.setEnabled(enabled); self.r_vrt2_text_edit.setEnabled(enabled)
        self.nr_vrt2_text_edit.setEnabled(enabled)
        apply_buttons = [btn for btn in self.findChildren(QPushButton) if btn != self.export_font_button]
        for btn in apply_buttons: btn.setEnabled(enabled)
        self.export_font_button.setEnabled(enabled)

# --- Batch Advance Width Dialog (変更なし) ---
class BatchAdvanceWidthDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("文字送り幅一括編集"); self.setMinimumWidth(350) 
        layout = QVBoxLayout(self); form_layout = QGridLayout()
        form_layout.addWidget(QLabel("適用する文字 (例: あいう / U+3042-U+304A / .notdef):"), 0, 0)
        self.char_spec_input = QLineEdit()
        self.char_spec_input.setPlaceholderText("あいうえお or U+XXXX-U+YYYY or .notdef")
        form_layout.addWidget(self.char_spec_input, 0, 1)
        form_layout.addWidget(QLabel("新しい文字送り幅 (0-1000):"), 1, 0)
        self.advance_width_spinbox = QSpinBox(); self.advance_width_spinbox.setRange(0, 1000) 
        self.advance_width_spinbox.setValue(DEFAULT_ADVANCE_WIDTH) 
        form_layout.addWidget(self.advance_width_spinbox, 1, 1); layout.addLayout(form_layout)
        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.button_box.button(QDialogButtonBox.Ok).setText("適用")
        self.button_box.button(QDialogButtonBox.Cancel).setText("キャンセル")
        self.button_box.accepted.connect(self.accept); self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)
    def get_values(self) -> Tuple[Optional[str], Optional[int]]:
        char_spec = self.char_spec_input.text().strip()
        adv_width = self.advance_width_spinbox.value()
        return char_spec if char_spec else None, adv_width

# --- MainWindow (Modified for new GlyphGridWidget and data flow) ---
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
        self.project_glyph_chars_cache: Set[str] = set() 
        # self._all_standard_glyphs_pixmap_cache: List[Tuple[str, Optional[QPixmap]]] = [] # No longer primary source for KV
        # self._all_nrvg_pixmap_cache: List[Tuple[str, Optional[QPixmap]]] = [] # No longer primary source for KV

        self.export_process: Optional[QProcess] = None
        self.original_export_button_state: bool = False 
        
        self._project_loading_in_progress: bool = False
        self._last_active_glyph_char_from_load: Optional[str] = None
        self._last_active_glyph_is_vrt2_from_load: bool = False


        self.drawing_editor_widget = DrawingEditorWidget()
        self.glyph_grid_widget = GlyphGridWidget(self.db_manager) # Pass db_manager
        # self.glyph_grid_widget.setFixedWidth(GLYPH_GRID_WIDTH) # Width is now dynamic
        self.properties_widget = PropertiesWidget()
        self.properties_widget.setFixedWidth(PROPERTIES_WIDTH)

        self.kanji_viewer_panel_widget: Optional[QWidget] = None
        self.kanji_viewer_font_combo: Optional[QComboBox] = None
        self.bookmark_font_button: Optional[QPushButton] = None
        self.kanji_viewer_display_label: Optional[QLabel] = None
        self.kanji_viewer_related_tabs: Optional[VerticalTabWidget] = None
        self.kanji_radicals_data = None; self.radical_to_kanji_data = None
        self.KANJI_TO_DATA_FILENAME = "kanji2element.json"; self.DATA_TO_KANJI_FILENAME = "element2kanji.json"
        self.related_kanji_label_style = "QLabel { border: 1px solid #DDDDDD; background-color: white; color: black; }"
        self.related_kanji_worker: Optional[RelatedKanjiWorker] = None
        self.current_related_kanji_process_id = 0
        self._worker_management_mutex = QMutex() 
        self._kanji_viewer_data_loaded_successfully = False
        self.kv_resize_timer: Optional[QTimer] = None
        self._kv_initial_char_to_display = "永" 
        self._kv_available_fonts: List[str] = [] 
        self._kv_char_to_update: Optional[str] = None 
        self._kv_deferred_update_timer: Optional[QTimer] = None
        self._kv_update_delay_ms = REFERENCE_GLYPH_DISPLAY_DELAY
        self.kv_display_mode = MainWindow.KV_MODE_FONT_DISPLAY 
        self.kv_mode_font_button: Optional[QPushButton] = None
        self.kv_mode_written_button: Optional[QPushButton] = None
        self.kv_mode_hide_button: Optional[QPushButton] = None
        self.kv_mode_button_group: Optional[QButtonGroup] = None
        self.font_bookmarks: List[str] = []
        self._load_font_bookmarks()
        
        # This dictionary will store char_key -> QPixmap for written glyphs found for the current KV update.
        # Used for rendering in KV_MODE_WRITTEN_GLYPHS.
        self.temp_written_kv_pixmaps: Dict[str, QPixmap] = {}


        central_widget = QWidget(); self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5); main_layout.setSpacing(5)
        self._init_ui_for_kanji_viewer_panel()
        if self.kanji_viewer_panel_widget: main_layout.addWidget(self.kanji_viewer_panel_widget, 0)
        main_layout.addWidget(self.glyph_grid_widget, 0) # No fixed width for grid
        main_layout.addWidget(self.drawing_editor_widget, 1)
        main_layout.addWidget(self.properties_widget, 0)

        self.file_menu: Optional[QMenu] = None; self.new_project_action: Optional[QAction] = None
        self.open_project_action: Optional[QAction] = None; self.exit_action: Optional[QAction] = None
        self.edit_menu: Optional[QMenu] = None; self.batch_adv_width_action: Optional[QAction] = None
        self.batch_import_glyphs_action: Optional[QAction] = None
        self.batch_import_reference_images_action: Optional[QAction] = None
        self._create_menus(); self.setStatusBar(QStatusBar(self))
        self._kv_deferred_update_timer = QTimer(self)
        self._kv_deferred_update_timer.setSingleShot(True)
        self._kv_deferred_update_timer.timeout.connect(self._process_deferred_kv_update)

        if self.kanji_viewer_font_combo:
            self.kanji_viewer_font_combo.currentTextChanged.connect(self._handle_kv_font_combo_changed)
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
        self._update_ui_for_project_state() 
        self._load_kanji_viewer_data_and_fonts() 
        QTimer.singleShot(0, self._kv_initial_display_setup) 
        QTimer.singleShot(100, self._update_bookmark_button_state) 

    def _get_font_bookmarks_path(self) -> str:
        script_dir = os.path.dirname(os.path.abspath(__file__)); return os.path.join(script_dir, FONT_BOOKMARKS_FILENAME)
    def _load_font_bookmarks(self):
        bookmarks_path = self._get_font_bookmarks_path()
        if os.path.exists(bookmarks_path):
            try:
                with open(bookmarks_path, 'r', encoding='utf-8') as f: loaded_bookmarks = json.load(f)
                if isinstance(loaded_bookmarks, list) and all(isinstance(item, str) for item in loaded_bookmarks):
                    self.font_bookmarks = loaded_bookmarks
                else: self.font_bookmarks = []
            except (json.JSONDecodeError, Exception): self.font_bookmarks = []
        else: self.font_bookmarks = [] 
    def _save_font_bookmarks(self):
        bookmarks_path = self._get_font_bookmarks_path()
        try:
            with open(bookmarks_path, 'w', encoding='utf-8') as f:
                json.dump(self.font_bookmarks, f, ensure_ascii=False, indent=2)
        except Exception as e: QMessageBox.warning(self, "ブックマーク保存エラー", f"フォントブックマークの保存に失敗しました: {e}")
    def _get_actual_font_name(self, display_name: str) -> str:
        return display_name[2:] if display_name.startswith("★ ") else display_name
    def _handle_kv_font_combo_changed(self, selected_display_name: str):
        self._update_bookmark_button_state() 
        actual_font_name = self._get_actual_font_name(selected_display_name)
        if actual_font_name and self.current_project_path and not self._project_loading_in_progress: 
            self.save_gui_setting_async(SETTING_KV_CURRENT_FONT, actual_font_name)
        char_to_update_kv_with = self.drawing_editor_widget.canvas.current_glyph_character or self._kv_initial_char_to_display
        self._trigger_kanji_viewer_update_for_current_glyph(char_to_update_kv_with)

    def _init_ui_for_kanji_viewer_panel(self):
        self.kanji_viewer_panel_widget = QWidget()
        self.kanji_viewer_panel_widget.setFixedWidth(350)
        self.kanji_viewer_panel_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        panel_layout = QVBoxLayout(self.kanji_viewer_panel_widget); panel_layout.setContentsMargins(5,5,5,5)
        font_selection_layout = QHBoxLayout(); font_selection_layout.addWidget(QLabel("参照:"))
        self.kanji_viewer_font_combo = QComboBox(); self.kanji_viewer_font_combo.setMinimumWidth(150)
        font_selection_layout.addWidget(self.kanji_viewer_font_combo, 1)
        self.bookmark_font_button = QPushButton("★"); self.bookmark_font_button.setToolTip("現在のフォントをブックマークに追加/削除")
        fm = self.fontMetrics(); button_width = fm.horizontalAdvance("★") + fm.horizontalAdvance("  ") 
        self.bookmark_font_button.setFixedWidth(button_width); self.bookmark_font_button.setCheckable(True) 
        self.bookmark_font_button.clicked.connect(self._toggle_font_bookmark)
        font_selection_layout.addWidget(self.bookmark_font_button); panel_layout.addLayout(font_selection_layout)
        self.kanji_viewer_display_label = QLabel(); self.kanji_viewer_display_label.setFixedSize(300, 300) 
        self.kanji_viewer_display_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.kanji_viewer_display_label.setStyleSheet("border: 2px solid #CCCCCC; background-color: white; color: black;")
        default_font = QFont(); default_font.setPointSize(10) 
        self.kanji_viewer_display_label.setFont(default_font)
        self.kanji_viewer_display_label.setText(self._kv_initial_char_to_display)
        panel_layout.addWidget(self.kanji_viewer_display_label, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        self.kanji_viewer_related_tabs = VerticalTabWidget()
        self.kanji_viewer_related_tabs.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        panel_layout.addWidget(self.kanji_viewer_related_tabs, 1)
        kv_mode_button_layout = QHBoxLayout()
        self.kv_mode_button_group = QButtonGroup(self); self.kv_mode_button_group.setExclusive(True)
        self.kv_mode_font_button = QPushButton("外部フォントを参照"); self.kv_mode_font_button.setCheckable(True)
        self.kv_mode_font_button.setToolTip("選択フォントで関連漢字を表示")
        kv_mode_button_layout.addWidget(self.kv_mode_font_button)
        self.kv_mode_button_group.addButton(self.kv_mode_font_button, MainWindow.KV_MODE_FONT_DISPLAY)
        self.kv_mode_written_button = QPushButton("プロジェクトを参照"); self.kv_mode_written_button.setCheckable(True)
        self.kv_mode_written_button.setToolTip("書き込み済みの関連グリフを一覧表示")
        kv_mode_button_layout.addWidget(self.kv_mode_written_button)
        self.kv_mode_button_group.addButton(self.kv_mode_written_button, MainWindow.KV_MODE_WRITTEN_GLYPHS)
        self.kv_mode_hide_button = QPushButton("非表示"); self.kv_mode_hide_button.setCheckable(True)
        self.kv_mode_hide_button.setToolTip("関連漢字表示を隠す")
        kv_mode_button_layout.addWidget(self.kv_mode_hide_button)
        self.kv_mode_button_group.addButton(self.kv_mode_hide_button, MainWindow.KV_MODE_HIDDEN)
        panel_layout.addLayout(kv_mode_button_layout)
        self.kv_mode_button_group.buttonToggled.connect(self._on_kv_display_mode_button_toggled)
        if self.kv_mode_written_button: self.kv_mode_written_button.setChecked(True)

    def _load_kanji_viewer_data_and_fonts(self): # (変更なし)
        error_title = "関連漢字データ読み込みエラー"
        def show_error_and_log(msg_key: str, is_critical: bool = False):
            full_msg = f"{msg_key} が見つからないか、読み込めませんでした。"
            if self.kanji_viewer_display_label: self.kanji_viewer_display_label.setText("関連データ\n読込失敗")
            if is_critical: QMessageBox.critical(self, error_title, f"{full_msg}\nアプリケーションの関連漢字機能は利用できません。")
            else: QMessageBox.warning(self, error_title, f"{full_msg}\n関連漢字機能は利用できません。")
            return False
        kanji_to_data_filepath = get_data_file_path(self.KANJI_TO_DATA_FILENAME)
        if not kanji_to_data_filepath: self._kanji_viewer_data_loaded_successfully = False; show_error_and_log(self.KANJI_TO_DATA_FILENAME); self._populate_kv_fonts(); return
        self.kanji_radicals_data = load_json_data(kanji_to_data_filepath)
        if not self.kanji_radicals_data: self._kanji_viewer_data_loaded_successfully = False; show_error_and_log(self.KANJI_TO_DATA_FILENAME); self._populate_kv_fonts(); return
        data_to_kanji_filepath = get_data_file_path(self.DATA_TO_KANJI_FILENAME)
        if not data_to_kanji_filepath: self._kanji_viewer_data_loaded_successfully = False; show_error_and_log(self.DATA_TO_KANJI_FILENAME); self._populate_kv_fonts(); return
        self.radical_to_kanji_data = load_json_data(data_to_kanji_filepath)
        if not self.radical_to_kanji_data: self._kanji_viewer_data_loaded_successfully = False; show_error_and_log(self.DATA_TO_KANJI_FILENAME); self._populate_kv_fonts(); return
        self._kanji_viewer_data_loaded_successfully = True; self._populate_kv_fonts()

    def _populate_kv_fonts(self): # (変更なし)
        if not self.kanji_viewer_font_combo: return
        current_selected_display_name = self.kanji_viewer_font_combo.currentText()
        current_selected_actual_name = self._get_actual_font_name(current_selected_display_name)
        self.kanji_viewer_font_combo.blockSignals(True); self.kanji_viewer_font_combo.clear(); self._kv_available_fonts.clear() 
        system_families = QFontDatabase.families(); valid_bookmarked_fonts = []; updated_bookmarks_list = False 
        for bookmarked_font in list(self.font_bookmarks): 
            if bookmarked_font in system_families: valid_bookmarked_fonts.append(bookmarked_font)
            else: self.font_bookmarks.remove(bookmarked_font); updated_bookmarks_list = True
        if updated_bookmarks_list: self._save_font_bookmarks()
        unbookmarked_system_fonts = [f for f in system_families if f not in valid_bookmarked_fonts]
        valid_bookmarked_fonts.sort(); unbookmarked_system_fonts.sort(); display_items = []
        for font_name in valid_bookmarked_fonts: display_items.append(f"★ {font_name}"); self._kv_available_fonts.append(font_name) 
        for font_name in unbookmarked_system_fonts: display_items.append(font_name); self._kv_available_fonts.append(font_name)
        if display_items:
            self.kanji_viewer_font_combo.addItems(display_items); new_index_to_select = -1
            prospective_display_name_bookmarked = f"★ {current_selected_actual_name}"
            if current_selected_actual_name in valid_bookmarked_fonts and prospective_display_name_bookmarked in display_items:
                new_index_to_select = display_items.index(prospective_display_name_bookmarked)
            elif current_selected_actual_name in display_items: 
                 try: new_index_to_select = display_items.index(current_selected_actual_name)
                 except ValueError: pass 
            if new_index_to_select != -1: self.kanji_viewer_font_combo.setCurrentIndex(new_index_to_select)
            elif self.kanji_viewer_font_combo.count() > 0: 
                default_selection_candidates = ["Yu Gothic", "Meiryo", "MS Gothic", "Noto Sans CJK JP", "ヒラギノ角ゴ ProN W3"]
                selected_idx_default = -1
                for pref_font_actual in default_selection_candidates:
                    pref_font_display_bookmarked = f"★ {pref_font_actual}"
                    if pref_font_display_bookmarked in display_items: selected_idx_default = display_items.index(pref_font_display_bookmarked); break
                    elif pref_font_actual in display_items: selected_idx_default = display_items.index(pref_font_actual); break
                if selected_idx_default != -1: self.kanji_viewer_font_combo.setCurrentIndex(selected_idx_default)
                else: self.kanji_viewer_font_combo.setCurrentIndex(0) 
        else:
            if self.kanji_viewer_display_label: self.kanji_viewer_display_label.setText("フォント\nなし"); self.kanji_viewer_display_label.setFont(QFont())
        self.kanji_viewer_font_combo.blockSignals(False); self._update_bookmark_button_state() 

    def _update_bookmark_button_state(self): # (変更なし)
        if not self.bookmark_font_button or not self.kanji_viewer_font_combo: return
        current_display_name = self.kanji_viewer_font_combo.currentText()
        if not current_display_name: self.bookmark_font_button.setChecked(False); self.bookmark_font_button.setEnabled(False); return
        actual_font_name = self._get_actual_font_name(current_display_name); is_bookmarked = actual_font_name in self.font_bookmarks
        self.bookmark_font_button.blockSignals(True); self.bookmark_font_button.setChecked(is_bookmarked); self.bookmark_font_button.blockSignals(False)
        tooltip_action = "削除" if is_bookmarked else "追加"
        self.bookmark_font_button.setToolTip(f"フォント「{actual_font_name}」をブックマークから{tooltip_action}")
        self.bookmark_font_button.setEnabled(bool(actual_font_name) and self.kanji_viewer_font_combo.count() > 0)
        if is_bookmarked: self.bookmark_font_button.setStyleSheet("QPushButton { background-color: palette(highlight); color: palette(highlighted-text); border: 1px solid palette(dark);} QPushButton:hover {background-color: palette(highlight); }")
        else: self.bookmark_font_button.setStyleSheet("QPushButton { } QPushButton:hover { background-color: palette(button); }")

    def _toggle_font_bookmark(self): # (変更なし)
        if not self.kanji_viewer_font_combo: return
        current_display_name = self.kanji_viewer_font_combo.currentText();
        if not current_display_name: return
        actual_font_name = self._get_actual_font_name(current_display_name)
        if not actual_font_name: return
        if actual_font_name in self.font_bookmarks:
            self.font_bookmarks.remove(actual_font_name)
            self.statusBar().showMessage(f"フォント「{actual_font_name}」をブックマークから削除しました。", 3000)
        else:
            self.font_bookmarks.append(actual_font_name); self.font_bookmarks.sort() 
            self.statusBar().showMessage(f"フォント「{actual_font_name}」をブックマークに追加しました。", 3000)
        self._save_font_bookmarks(); self._populate_kv_fonts() 

    def _kv_initial_display_setup(self): # (変更なし)
        if self.kanji_viewer_font_combo and self.kanji_viewer_font_combo.count() > 0:
            if self.kanji_viewer_font_combo.currentIndex() == -1: self.kanji_viewer_font_combo.setCurrentIndex(0) 
        elif self.kanji_viewer_display_label: 
            self.kanji_viewer_display_label.setText("利用可能な\nフォントが\nありません"); self.kanji_viewer_display_label.setFont(QFont())
        current_glyph = self.drawing_editor_widget.canvas.current_glyph_character
        char_to_display_initially = current_glyph or self._kv_initial_char_to_display
        with QMutexLocker(self._worker_management_mutex): self._kv_char_to_update = char_to_display_initially
        if self.current_project_path or not self._project_loading_in_progress: 
            self._kv_deferred_update_timer.start(self._kv_update_delay_ms)

    def _kv_calculate_optimal_font_size(self, char: str, rect: QRect, family: str, margin: float = 0.8) -> int: # (変更なし)
        if not char or rect.isEmpty() or not family: return 1 
        font = QFont(family); low = 1; high = min(max(1, rect.height()), 1200); best_size = 1
        iterations = 0; max_iterations = 100; target_width = rect.width() * margin; target_height = rect.height() * margin
        while low <= high and iterations < max_iterations:
            iterations += 1; mid = (low + high) // 2
            if mid == 0: low = 1; continue 
            font.setPixelSize(mid); fm = QFontMetrics(font); current_bound_rect = fm.boundingRect(char)
            if current_bound_rect.isNull() or current_bound_rect.isEmpty(): high = mid -1; continue
            if current_bound_rect.width() <= target_width and current_bound_rect.height() <= target_height:
                best_size = mid; low = mid + 1 
            else: high = mid - 1 
        return max(1, best_size)

    def _kv_set_label_font_and_text(self, label: QLabel, char: str, family: str, rect: QRect, margin: float = 0.9): # (変更なし)
        if not label: return
        if not char or not family: label.setText(""); label.setFont(QFont()); return
        try:
            px_size = self._kv_calculate_optimal_font_size(char, rect, family, margin)
            font = QFont(family); font.setPixelSize(px_size)
            label.setFont(font); label.setText(char); label.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        except Exception as e: label.setText("ERR"); label.setFont(QFont())

    @Slot()
    def _process_deferred_kv_update(self): # (変更なし)
        char_to_process = None
        with QMutexLocker(self._worker_management_mutex):
            if self._kv_char_to_update: char_to_process = self._kv_char_to_update
        if char_to_process: self._trigger_kanji_viewer_update_for_current_glyph(char_to_process)

    @Slot(QAbstractButton, bool)
    def _on_kv_display_mode_button_toggled(self, button: QAbstractButton, checked: bool): # (変更なし)
        if not checked: return 
        new_mode = self.kv_mode_button_group.id(button)
        if new_mode == self.kv_display_mode and self.current_project_path and not self._project_loading_in_progress : return 
        self.kv_display_mode = new_mode
        if self.current_project_path and not self._project_loading_in_progress: 
            self.save_gui_setting_async(SETTING_KV_DISPLAY_MODE, str(new_mode))
        current_char_for_kv = self.drawing_editor_widget.canvas.current_glyph_character or self._kv_initial_char_to_display
        self._trigger_kanji_viewer_update_for_current_glyph(current_char_for_kv)

    def _trigger_kanji_viewer_update_for_current_glyph(self, current_char: str): # (変更なし)
        if self._project_loading_in_progress: return 
        if not self.kanji_viewer_display_label or not self.kanji_viewer_related_tabs or not self.kanji_viewer_font_combo: return 
        font_family_display_name = self.kanji_viewer_font_combo.currentText()
        font_family_for_main_display = self._get_actual_font_name(font_family_display_name) 
        if not font_family_for_main_display and self._kv_available_fonts: font_family_for_main_display = self._kv_available_fonts[0] 
        elif not font_family_for_main_display and not self._kv_available_fonts: 
             if self.kanji_viewer_display_label: self.kanji_viewer_display_label.setText("フォント\n選択不可"); self.kanji_viewer_display_label.setFont(QFont())
             if self.kanji_viewer_related_tabs: self.kanji_viewer_related_tabs.clear()
             return 
        if self.kv_display_mode == MainWindow.KV_MODE_HIDDEN:
            if self.kanji_viewer_related_tabs: self.kanji_viewer_related_tabs.clear()
            if self.kanji_viewer_display_label:
                if current_char and len(current_char) == 1: 
                    self._kv_set_label_font_and_text(self.kanji_viewer_display_label, current_char, font_family_for_main_display, self.kanji_viewer_display_label.rect(), 0.85)
                else: self.kanji_viewer_display_label.setText(""); self.kanji_viewer_display_label.setFont(QFont())
            return 
        if not current_char or len(current_char) != 1: 
            if self.kanji_viewer_display_label: self.kanji_viewer_display_label.setText(""); self.kanji_viewer_display_label.setFont(QFont())
            if self.kanji_viewer_related_tabs: self.kanji_viewer_related_tabs.clear()
            with QMutexLocker(self._worker_management_mutex): 
                if self.related_kanji_worker and self.related_kanji_worker.isRunning(): self.related_kanji_worker.cancel()
            return
        if self.kanji_viewer_display_label:
            self._kv_set_label_font_and_text(self.kanji_viewer_display_label, current_char, font_family_for_main_display, self.kanji_viewer_display_label.rect(), 0.85)
            self.kanji_viewer_display_label.setStyleSheet("border: 2px solid #CCCCCC; background-color: white; color: black;")
        if not self._kanji_viewer_data_loaded_successfully:
            if self.kanji_viewer_related_tabs:
                self.kanji_viewer_related_tabs.clear()
                if self.kanji_viewer_related_tabs.count() == 0: 
                    no_data_label = QLabel("関連漢字データが\n読み込まれていません。"); no_data_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    self.kanji_viewer_related_tabs.addTab(no_data_label, "情報")
            return
        with QMutexLocker(self._worker_management_mutex):
            self.current_related_kanji_process_id += 1; process_id = self.current_related_kanji_process_id
            if self.related_kanji_worker and self.related_kanji_worker.isRunning(): self.related_kanji_worker.cancel() 
            tab_bar_instance = self.kanji_viewer_related_tabs.tabBar()
            tab_bar_width = tab_bar_instance.tab_fixed_width if hasattr(tab_bar_instance, 'tab_fixed_width') else 60
            content_area_width = self.kanji_viewer_related_tabs.width() - tab_bar_width - 20 
            if content_area_width <= 0: content_area_width = self.kanji_viewer_panel_widget.width() - tab_bar_width - 30 if self.kanji_viewer_panel_widget else 250 - tab_bar_width - 30
            if content_area_width <= 0: content_area_width = 250
            ideal_item_width_for_3_cols = content_area_width / 3 - 10 
            font_px_size_for_related = max(1, int(ideal_item_width_for_3_cols * 0.7)) 
            font_px_size_for_related = min(font_px_size_for_related, 50); font_px_size_for_related = max(font_px_size_for_related, 16) 
            effective_font_family_for_worker = font_family_for_main_display 
            new_worker = RelatedKanjiWorker(process_id, current_char, self.kanji_radicals_data, self.radical_to_kanji_data, effective_font_family_for_worker, font_px_size_for_related, self)
            new_worker.result_ready.connect(self._handle_kv_related_kanji_result)
            new_worker.error_occurred.connect(self._handle_kv_worker_error)
            new_worker.finished.connect(self._on_kv_worker_finished); new_worker.finished.connect(new_worker.deleteLater)
            self.related_kanji_worker = new_worker; self.related_kanji_worker.start()


    @Slot(int, dict, str, int)
    def _handle_kv_related_kanji_result(self, process_id: int, results_dict: dict, font_family_from_worker: str, font_px_size: int):
        with QMutexLocker(self._worker_management_mutex):
            if process_id != self.current_related_kanji_process_id:
                return
        if not self.kanji_viewer_related_tabs:
            return

        current_tab_text_before_clear = ""
        current_tab_idx = self.kanji_viewer_related_tabs.currentIndex()
        if self.kanji_viewer_related_tabs.count() > 0 and current_tab_idx != -1:
            current_tab_text_before_clear = self.kanji_viewer_related_tabs.tabText(current_tab_idx)
        
        self.kanji_viewer_related_tabs.clear()
        self.temp_written_kv_pixmaps.clear() # Clear previous temporary cache

        char_for_msg_display = self.drawing_editor_widget.canvas.current_glyph_character
        if not char_for_msg_display: 
            if self.related_kanji_worker and self.related_kanji_worker.input_char:
                char_for_msg_display = self.related_kanji_worker.input_char
            else:
                char_for_msg_display = "選択文字"

        effective_results_dict = results_dict
        
        if self.kv_display_mode == MainWindow.KV_MODE_WRITTEN_GLYPHS:
            if not self.current_project_path: 
                no_project_label = QLabel("プロジェクトがロードされていません。")
                no_project_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.kanji_viewer_related_tabs.addTab(no_project_label, "エラー")
                return

            filtered_dict = {}
            for radical, kanji_list in results_dict.items():
                written_kanji_in_list = []
                for k_char in kanji_list:
                    is_written = False
                    pixmap_to_store: Optional[QPixmap] = None

                    # Check standard glyphs
                    if self.glyph_grid_widget.std_glyph_model.is_glyph_written(k_char):
                        is_written = True
                        if k_char in self.glyph_grid_widget.std_glyph_model._pixmap_cache:
                            pixmap_to_store = self.glyph_grid_widget.std_glyph_model._pixmap_cache[k_char]
                        else: # Not in model's RAM cache, try direct DB load for KV
                            pixmap_to_store = self.db_manager.load_glyph_image(k_char, is_vrt2=False)
                    
                    # Check VRT2 glyphs (non-rotated) if not found as written standard glyph
                    elif k_char in self.non_rotated_vrt2_chars and \
                         self.glyph_grid_widget.vrt2_glyph_model.is_glyph_written(k_char):
                        is_written = True
                        if k_char in self.glyph_grid_widget.vrt2_glyph_model._pixmap_cache:
                            pixmap_to_store = self.glyph_grid_widget.vrt2_glyph_model._pixmap_cache[k_char]
                        else: # Not in model's RAM cache, try direct DB load for KV
                            pixmap_to_store = self.db_manager.load_glyph_image(k_char, is_vrt2=True)
                    
                    if is_written:
                        written_kanji_in_list.append(k_char)
                        if pixmap_to_store and k_char not in self.temp_written_kv_pixmaps:
                             self.temp_written_kv_pixmaps[k_char] = pixmap_to_store.copy() # Store a copy
                
                if written_kanji_in_list:
                    filtered_dict[radical] = sorted(list(set(written_kanji_in_list))) 
            
            effective_results_dict = filtered_dict
            if not effective_results_dict: 
                 msg_text = f"「{char_for_msg_display}」に関連する書き込み済みグリフは\n見つかりませんでした。" 
                 no_results_label = QLabel(msg_text)
                 no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter); no_results_label.setWordWrap(True)
                 container_widget = QWidget(); layout = QVBoxLayout(container_widget)
                 layout.addWidget(no_results_label); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                 self.kanji_viewer_related_tabs.addTab(container_widget, "") 
                 return

        if not effective_results_dict:
            msg_text = f"「{char_for_msg_display}」の構成部首データがないか、\n関連する漢字が見つかりませんでした。"
            if self.kv_display_mode == MainWindow.KV_MODE_WRITTEN_GLYPHS: 
                 msg_text = f"「{char_for_msg_display}」に関連する書き込み済みグリフは\n見つかりませんでした。"
            elif self.kanji_radicals_data and char_for_msg_display in self.kanji_radicals_data and \
                 not self.kanji_radicals_data.get(char_for_msg_display): 
                msg_text = f"「{char_for_msg_display}」には構成部首が登録されていません。"
            elif self.kanji_radicals_data and char_for_msg_display not in self.kanji_radicals_data:
                 msg_text = f"「{char_for_msg_display}」のデータがありません。"
            no_results_label = QLabel(msg_text)
            no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter); no_results_label.setWordWrap(True)
            container_widget = QWidget(); layout = QVBoxLayout(container_widget)
            layout.addWidget(no_results_label); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.kanji_viewer_related_tabs.addTab(container_widget, "")
            return

        tab_bar_instance = self.kanji_viewer_related_tabs.tabBar()
        tab_bar_width = tab_bar_instance.tab_fixed_width if hasattr(tab_bar_instance, 'tab_fixed_width') else 60
        content_area_width = self.kanji_viewer_related_tabs.width() - tab_bar_width - \
                             self.kanji_viewer_related_tabs.contentsMargins().left() - \
                             self.kanji_viewer_related_tabs.contentsMargins().right() - 10 
        if content_area_width <= 0: content_area_width = self.kanji_viewer_panel_widget.width() - tab_bar_width - 30 if self.kanji_viewer_panel_widget else 250 - tab_bar_width -30
        if content_area_width <= 0: content_area_width = 250
        item_side_length = font_px_size + 12 
        item_side_length = max(item_side_length, 36); item_side_length = min(item_side_length, 80) 
        preview_display_size = QSize(item_side_length - 4, item_side_length - 4) 
        MAX_COLS = max(1, content_area_width // (item_side_length + 5)) 
        any_tabs_added = False; new_selected_index_to_restore = -1
        
        related_kanji_font: QFont 
        if self.kv_display_mode == MainWindow.KV_MODE_FONT_DISPLAY:
            related_kanji_font = QFont(font_family_from_worker) 
            related_kanji_font.setPixelSize(font_px_size)

        sorted_radicals = sorted(effective_results_dict.keys())
        for radical in sorted_radicals:
            kanji_list = effective_results_dict[radical]
            if not kanji_list: continue
            any_tabs_added = True; tab_content_widget = QWidget()
            tab_content_widget.setStyleSheet(self.related_kanji_label_style) 
            tab_layout = QGridLayout(tab_content_widget)
            tab_layout.setSpacing(5); tab_layout.setContentsMargins(5, 5, 5, 5)
            row, col = 0, 0
            for kanji_char_to_display in kanji_list:
                kanji_label = QLabel()
                kanji_label.setFixedSize(item_side_length, item_side_length)
                kanji_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

                if self.kv_display_mode == MainWindow.KV_MODE_WRITTEN_GLYPHS:
                    glyph_pixmap = self.temp_written_kv_pixmaps.get(kanji_char_to_display)
                    if glyph_pixmap and not glyph_pixmap.isNull():
                        scaled_pixmap = glyph_pixmap.scaled(preview_display_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                        kanji_label.setPixmap(scaled_pixmap)
                    else: 
                        kanji_label.setText(kanji_char_to_display) 
                        font_for_fallback = QFont(self.font().family()) 
                        font_for_fallback.setPixelSize(max(10, int(item_side_length * 0.6)))
                        kanji_label.setFont(font_for_fallback)
                else: # KV_MODE_FONT_DISPLAY
                    kanji_label.setFont(related_kanji_font)
                    kanji_label.setText(kanji_char_to_display)
                
                tab_layout.addWidget(kanji_label, row, col)
                col += 1
                if col >= MAX_COLS: col = 0; row += 1
            
            if col > 0 and col < MAX_COLS :
                for c_fill in range(col, MAX_COLS):
                    spacer_item = QSpacerItem(item_side_length, item_side_length, QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
                    tab_layout.addItem(spacer_item, row, c_fill)
            tab_layout.setRowStretch(row + 1, 1); tab_layout.setColumnStretch(MAX_COLS, 1) 
            scroll_area = QScrollArea(); scroll_area.setWidgetResizable(True); scroll_area.setWidget(tab_content_widget)
            scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded); scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            tab_idx = self.kanji_viewer_related_tabs.addTab(scroll_area, radical)
            if radical == current_tab_text_before_clear: new_selected_index_to_restore = tab_idx

        if not any_tabs_added and effective_results_dict:
            no_results_label = QLabel(f"「{char_for_msg_display}」の各構成部首を共有する\n他の漢字は見つかりませんでした。")
            if self.kv_display_mode == MainWindow.KV_MODE_WRITTEN_GLYPHS:
                 no_results_label.setText(f"「{char_for_msg_display}」の各構成部首を共有する\n書き込み済みグリフは見つかりませんでした。")
            no_results_label.setAlignment(Qt.AlignmentFlag.AlignCenter); no_results_label.setWordWrap(True)
            container_widget = QWidget(); layout = QVBoxLayout(container_widget); layout.addWidget(no_results_label); layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.kanji_viewer_related_tabs.addTab(container_widget, "")
            
        if new_selected_index_to_restore != -1: self.kanji_viewer_related_tabs.setCurrentIndex(new_selected_index_to_restore)
        elif self.kanji_viewer_related_tabs.count() > 0: self.kanji_viewer_related_tabs.setCurrentIndex(0)
        
        self.temp_written_kv_pixmaps.clear()


    @Slot(int, str)
    def _handle_kv_worker_error(self, process_id: int, error_message: str): # (変更なし)
        with QMutexLocker(self._worker_management_mutex):
            if process_id != self.current_related_kanji_process_id: return 
        if not self.kanji_viewer_related_tabs: return
        self.kanji_viewer_related_tabs.clear(); lbl = QLabel(f"関連漢字の取得エラー:\n{error_message}")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter); lbl.setWordWrap(True)
        cw = QWidget(); lo = QVBoxLayout(cw); lo.addWidget(lbl); self.kanji_viewer_related_tabs.addTab(cw, "エラー")

    @Slot()
    def _on_kv_worker_finished(self): # (変更なし)
        sender_worker = self.sender(); 
        if not isinstance(sender_worker, RelatedKanjiWorker): return 
        with QMutexLocker(self._worker_management_mutex):
            if self.related_kanji_worker is sender_worker: self.related_kanji_worker = None 

    def _create_menus(self): # (変更なし)
        menu_bar = self.menuBar(); self.file_menu = menu_bar.addMenu("&ファイル")
        self.new_project_action = QAction("&新規プロジェクト...", self); self.new_project_action.triggered.connect(self.new_project)
        self.file_menu.addAction(self.new_project_action)
        self.open_project_action = QAction("&プロジェクトを開く...", self); self.open_project_action.triggered.connect(self.open_project)
        self.file_menu.addAction(self.open_project_action); self.file_menu.addSeparator()
        self.exit_action = QAction("&終了", self); self.exit_action.triggered.connect(self.close)
        self.file_menu.addAction(self.exit_action); self.edit_menu = menu_bar.addMenu("&編集")
        self.batch_adv_width_action = QAction("文字送り幅一括編集...", self); self.batch_adv_width_action.triggered.connect(self.open_batch_advance_width_dialog)
        self.edit_menu.addAction(self.batch_adv_width_action); self.edit_menu.addSeparator()
        self.batch_import_glyphs_action = QAction("グリフの一括読み込み...", self); self.batch_import_glyphs_action.triggered.connect(self.batch_import_glyphs)
        self.edit_menu.addAction(self.batch_import_glyphs_action)
        self.batch_import_reference_images_action = QAction("下書きの一括読み込み...", self); self.batch_import_reference_images_action.triggered.connect(self.batch_import_reference_images)
        self.edit_menu.addAction(self.batch_import_reference_images_action)

    def _set_project_loading_state(self, loading: bool): # (glyph_grid_widget.setEnabled -> .set_search_and_filter_enabled)
        self._project_loading_in_progress = loading
        self.drawing_editor_widget.setEnabled(not loading and self.current_project_path is not None)
        self.properties_widget.setEnabled(not loading and self.current_project_path is not None)
        self.glyph_grid_widget.set_search_and_filter_enabled(not loading and self.current_project_path is not None) # Changed
        if self.batch_adv_width_action: self.batch_adv_width_action.setEnabled(not loading and self.current_project_path is not None)
        if self.batch_import_glyphs_action: self.batch_import_glyphs_action.setEnabled(not loading and self.current_project_path is not None)
        if self.batch_import_reference_images_action: self.batch_import_reference_images_action.setEnabled(not loading and self.current_project_path is not None)
        if self.new_project_action: self.new_project_action.setEnabled(not loading)
        if self.open_project_action: self.open_project_action.setEnabled(not loading)
        if loading: self.statusBar().showMessage("プロジェクト処理中...", 0) 
        else:
            if self.current_project_path: self.statusBar().showMessage(f"プロジェクト '{os.path.basename(self.current_project_path)}' がロードされました。", 5000)
            else: self.statusBar().clearMessage()
        self._update_ui_for_project_state() 

    def _update_ui_for_project_state(self): # (glyph_grid_widget.clear_grid -> .clear_grid_and_models, etc.)
        project_loaded_and_not_processing = self.current_project_path is not None and not self._project_loading_in_progress
        self.drawing_editor_widget.set_enabled_controls(project_loaded_and_not_processing)
        self.properties_widget.set_enabled_controls(project_loaded_and_not_processing)
        self.glyph_grid_widget.set_search_and_filter_enabled(project_loaded_and_not_processing) # Changed
        if self.batch_adv_width_action: self.batch_adv_width_action.setEnabled(project_loaded_and_not_processing)
        if self.batch_import_glyphs_action: self.batch_import_glyphs_action.setEnabled(project_loaded_and_not_processing)
        if self.batch_import_reference_images_action: self.batch_import_reference_images_action.setEnabled(project_loaded_and_not_processing)
        kv_data_ok = self._kanji_viewer_data_loaded_successfully and bool(self.kanji_viewer_font_combo and self.kanji_viewer_font_combo.count() > 0)
        kv_panel_enabled = kv_data_ok and not self._project_loading_in_progress 
        if self.kanji_viewer_panel_widget: self.kanji_viewer_panel_widget.setEnabled(kv_panel_enabled)
        if self.bookmark_font_button: self.bookmark_font_button.setEnabled(kv_panel_enabled and self.kanji_viewer_font_combo.count() > 0)
        kv_buttons_enabled_base = kv_panel_enabled 
        if self.kanji_viewer_font_combo: self.kanji_viewer_font_combo.setEnabled(kv_panel_enabled) 
        if self.kv_mode_button_group:
            buttons = self.kv_mode_button_group.buttons() 
            for btn in buttons: btn.setEnabled(kv_buttons_enabled_base) 
        if project_loaded_and_not_processing:
            self.setWindowTitle(f"P-Glyph - {os.path.basename(self.current_project_path)}")
            if kv_buttons_enabled_base and self.kv_mode_button_group: 
                current_checked_button_id = self.kv_mode_button_group.checkedId()
                current_mode_to_set = self.kv_display_mode 
                if current_checked_button_id == -1 or current_checked_button_id != current_mode_to_set:
                    button_to_set = self.kv_mode_button_group.button(current_mode_to_set)
                    if button_to_set: self.kv_mode_button_group.blockSignals(True); button_to_set.setChecked(True); self.kv_mode_button_group.blockSignals(False)
                    elif self.kv_mode_written_button: self.kv_mode_button_group.blockSignals(True); self.kv_mode_written_button.setChecked(True); self.kv_mode_button_group.blockSignals(False)
        else: 
            self.setWindowTitle("P-Glyph")
            if not self._project_loading_in_progress: 
                self.drawing_editor_widget.set_rotated_vrt2_chars(set()) 
                self.glyph_grid_widget.set_non_rotated_vrt2_chars(set())
                self.glyph_grid_widget.clear_grid_and_models() # Changed
                # self._all_standard_glyphs_pixmap_cache.clear() # No longer needed as primary source
                # self._all_nrvg_pixmap_cache.clear() # No longer needed as primary source
                self.drawing_editor_widget.canvas.reference_image = None
                self.drawing_editor_widget.canvas.set_reference_image_opacity(DEFAULT_REFERENCE_IMAGE_OPACITY)
                self.drawing_editor_widget.canvas.load_glyph("", None, None, DEFAULT_ADVANCE_WIDTH, is_vrt2=False) 
                self.properties_widget.load_character_set(""); self.properties_widget.load_r_vrt2_set("")
                self.properties_widget.load_nr_vrt2_set(""); self.properties_widget.load_font_name(DEFAULT_FONT_NAME)
                self.properties_widget.load_font_weight(DEFAULT_FONT_WEIGHT)
                self.glyph_grid_widget.set_active_glyph(None)
                self.drawing_editor_widget.update_unicode_display(None)
                self.project_glyph_chars_cache.clear(); self.non_rotated_vrt2_chars.clear()
                if self.kv_mode_button_group and self.kv_mode_written_button: 
                    self.kv_mode_button_group.blockSignals(True); self.kv_mode_written_button.setChecked(True); self.kv_mode_button_group.blockSignals(False)

                self.kv_display_mode = MainWindow.KV_MODE_WRITTEN_GLYPHS
                if self.kanji_viewer_display_label:
                    default_kv_font = ""
                    if self.kanji_viewer_font_combo and self.kanji_viewer_font_combo.count() > 0:
                        default_kv_font_display_name = self.kanji_viewer_font_combo.itemText(0)
                        if default_kv_font_display_name: # この行が修正点です
                            default_kv_font = self._get_actual_font_name(default_kv_font_display_name)
                    self._kv_set_label_font_and_text(self.kanji_viewer_display_label, self._kv_initial_char_to_display, default_kv_font, self.kanji_viewer_display_label.rect(), 0.85)
                if self.kanji_viewer_related_tabs: self.kanji_viewer_related_tabs.clear()
        self._update_bookmark_button_state() 

    def _load_font_settings_txt(self): # (変更なし)
        script_dir = os.path.dirname(os.path.abspath(__file__)); settings_file_path = os.path.join(script_dir, FONT_SETTINGS_FILENAME)
        char_string = DEFAULT_CHAR_SET 
        try:
            if os.path.exists(settings_file_path):
                with open(settings_file_path, 'r', encoding='utf-8') as f: content = f.read().strip() 
                if content: char_string = content
            else: 
                try:
                    with open(settings_file_path, 'w', encoding='utf-8') as f: f.write(DEFAULT_CHAR_SET)
                except IOError as e: pass
        except IOError as e: pass
        seen = set(); unique_ordered_chars = []
        for c in char_string:
            if len(c) == 1 and c not in seen : unique_ordered_chars.append(c); seen.add(c)
        return sorted(unique_ordered_chars, key=ord)

    def _load_vrt2_settings_txt(self, filename: str, default_set: str): # (変更なし)
        script_dir = os.path.dirname(os.path.abspath(__file__)); settings_file_path = os.path.join(script_dir, filename)
        char_string = default_set
        try:
            if os.path.exists(settings_file_path):
                with open(settings_file_path, 'r', encoding='utf-8') as f: content = f.read().strip()
                if content: char_string = content
            else:
                try:
                    with open(settings_file_path, 'w', encoding='utf-8') as f: f.write(default_set)
                except IOError as e: pass
        except IOError as e: pass
        seen = set(); unique_ordered_chars = []
        for c in char_string:
            if len(c) == 1 and c not in seen: unique_ordered_chars.append(c); seen.add(c)
        return sorted(unique_ordered_chars, key=ord)

    def new_project(self): # (glyph_grid_widget.clear_grid -> .clear_grid_and_models)
        if self._project_loading_in_progress: QMessageBox.information(self, "処理中", "プロジェクトの読み込みまたは作成処理が進行中です。"); return
        filepath, _ = QFileDialog.getSaveFileName(self, "新規プロジェクトを作成", "", "Font Project Files (*.fontproj)")
        if filepath:
            if not filepath.endswith(".fontproj"): filepath += ".fontproj"
            self._set_project_loading_state(True) 
            initial_chars = self._load_font_settings_txt()
            r_vrt2_chars = self._load_vrt2_settings_txt(R_VERT_FILENAME, DEFAULT_R_VERT_CHARS)
            nr_vrt2_chars = self._load_vrt2_settings_txt(VERT_FILENAME, DEFAULT_VERT_CHARS)
            if not initial_chars and DEFAULT_CHAR_SET: 
                initial_chars = sorted(list(set(c for c in DEFAULT_CHAR_SET if len(c) == 1)), key=ord)
            create_worker = CreateProjectWorker(filepath, initial_chars, r_vrt2_chars, nr_vrt2_chars)
            create_worker.signals.project_created.connect(self._on_project_created_and_start_load)
            create_worker.signals.error.connect(self._on_project_create_error)
            create_worker.signals.finished.connect(self._check_and_finalize_loading_state_after_create) 
            self.thread_pool.start(create_worker)

    @Slot()
    def _check_and_finalize_loading_state_after_create(self): pass # (変更なし)

    @Slot(str)
    def _on_project_created_and_start_load(self, filepath: str): # (glyph_grid_widget.clear_grid -> .clear_grid_and_models, LoadProjectWorker signal connect change)
        self.current_project_path = filepath
        self.db_manager.connect_db(filepath) 
        # self._all_standard_glyphs_pixmap_cache.clear(); self._all_nrvg_pixmap_cache.clear() # No longer needed
        self.glyph_grid_widget.clear_grid_and_models() # Changed
        load_worker = LoadProjectWorker(self.current_project_path)
        load_worker.signals.basic_info_loaded.connect(self._on_project_basic_info_loaded)
        # load_worker.signals.glyph_batch_loaded.connect(self._on_glyph_batch_loaded) # No longer used by grid directly
        load_worker.signals.load_progress.connect(self._on_load_progress)
        load_worker.signals.error.connect(self._on_project_load_error)
        load_worker.signals.finished.connect(self._check_and_finalize_loading_state) 
        self.thread_pool.start(load_worker)

    @Slot(str)
    def _on_project_create_error(self, error_message: str): # (変更なし)
        QMessageBox.critical(self, "プロジェクト作成エラー", error_message)
        self.current_project_path = None; self.project_glyph_chars_cache.clear(); self.non_rotated_vrt2_chars.clear()
        self._set_project_loading_state(False) 

    def open_project(self): # (glyph_grid_widget.clear_grid -> .clear_grid_and_models, LoadProjectWorker signal connect change)
        if self._project_loading_in_progress: QMessageBox.information(self, "処理中", "プロジェクトの読み込みまたは作成処理が進行中です。"); return
        filepath, _ = QFileDialog.getOpenFileName(self, "プロジェクトを開く", "", "Font Project Files (*.fontproj)")
        if filepath:
            self._set_project_loading_state(True) 
            self.current_project_path = filepath; self.db_manager.connect_db(filepath) 
            # self._all_standard_glyphs_pixmap_cache.clear(); self._all_nrvg_pixmap_cache.clear() # No longer needed
            self.glyph_grid_widget.clear_grid_and_models() # Changed
            worker = LoadProjectWorker(self.current_project_path)
            worker.signals.basic_info_loaded.connect(self._on_project_basic_info_loaded)
            # worker.signals.glyph_batch_loaded.connect(self._on_glyph_batch_loaded) # No longer used
            worker.signals.load_progress.connect(self._on_load_progress)
            worker.signals.error.connect(self._on_project_load_error)
            worker.signals.finished.connect(self._check_and_finalize_loading_state) 
            self.thread_pool.start(worker)

    @Slot()
    def _check_and_finalize_loading_state(self): # (変更なし)
        if self._project_loading_in_progress: 
            self._set_project_loading_state(False)
            self._select_initial_glyph_after_full_load()
            if self.current_project_path: 
                self.statusBar().showMessage(f"プロジェクト '{os.path.basename(self.current_project_path)}' の読み込み完了。", 5000)

    @Slot(dict)
    def _on_project_basic_info_loaded(self, basic_data: dict):
        if not self.current_project_path: return 
        self.properties_widget.load_character_set("".join(basic_data['char_set_list']))
        self.properties_widget.load_r_vrt2_set("".join(basic_data['r_vrt2_list']))
        self.drawing_editor_widget.set_rotated_vrt2_chars(set(basic_data['r_vrt2_list']))
        self.properties_widget.load_nr_vrt2_set("".join(basic_data['nr_vrt2_list']))
        self.non_rotated_vrt2_chars = set(basic_data['nr_vrt2_list'])
        self.glyph_grid_widget.set_non_rotated_vrt2_chars(self.non_rotated_vrt2_chars)
        self.properties_widget.load_font_name(basic_data['font_name']); self.properties_widget.load_font_weight(basic_data['font_weight'])
        self.drawing_editor_widget.apply_gui_settings(basic_data['gui_settings'])
        
        # Update GlyphGridWidget models with character lists and their initial image state
        self.glyph_grid_widget.populate_models(
            basic_data['char_set_list_with_img_info'],
            basic_data['nr_vrt2_list_with_img_info']
        )
        # Update internal caches for KV and other functions
        self.project_glyph_chars_cache = set(basic_data['char_set_list']) # Raw list for this cache
        # For _all_standard_glyphs_pixmap_cache etc., these are for written glyphs.
        # KV will now directly use db_manager or model's cache.
        # self._all_standard_glyphs_pixmap_cache.clear() # No longer needed
        # self._all_nrvg_pixmap_cache.clear() # No longer needed


        if self.kanji_viewer_font_combo: # (KV font logic unchanged)
            self._populate_kv_fonts() 
            kv_font_actual_name = basic_data['kv_font_actual_name']
            if kv_font_actual_name:
                target_display_name_bookmarked = f"★ {kv_font_actual_name}"; target_display_name_plain = kv_font_actual_name
                found_idx = self.kanji_viewer_font_combo.findText(target_display_name_bookmarked)
                if found_idx == -1: found_idx = self.kanji_viewer_font_combo.findText(target_display_name_plain)
                if found_idx != -1: self.kanji_viewer_font_combo.blockSignals(True); self.kanji_viewer_font_combo.setCurrentIndex(found_idx); self.kanji_viewer_font_combo.blockSignals(False)
                elif self.kanji_viewer_font_combo.count() > 0: 
                    if self.kanji_viewer_font_combo.currentIndex() == -1: self.kanji_viewer_font_combo.setCurrentIndex(0)
                    new_default_font_display = self.kanji_viewer_font_combo.currentText()
                    if new_default_font_display:
                        new_default_font_actual = self._get_actual_font_name(new_default_font_display)
                        if kv_font_actual_name != new_default_font_actual: self.save_gui_setting_async(SETTING_KV_CURRENT_FONT, new_default_font_actual)
            elif self.kanji_viewer_font_combo.count() > 0 : 
                if self.kanji_viewer_font_combo.currentIndex() == -1: self.kanji_viewer_font_combo.setCurrentIndex(0)
                current_font_display = self.kanji_viewer_font_combo.currentText()
                if current_font_display:
                    current_font_actual = self._get_actual_font_name(current_font_display)
                    self.save_gui_setting_async(SETTING_KV_CURRENT_FONT, current_font_actual)
        self.kv_display_mode = basic_data['kv_display_mode_val'] 
        if self.kv_mode_button_group:
            button_to_check = self.kv_mode_button_group.button(self.kv_display_mode)
            if button_to_check: self.kv_mode_button_group.blockSignals(True); button_to_check.setChecked(True); self.kv_mode_button_group.blockSignals(False)
            elif self.kv_mode_written_button: self.kv_mode_button_group.blockSignals(True); self.kv_mode_written_button.setChecked(True); self.kv_mode_button_group.blockSignals(False)
        self._last_active_glyph_char_from_load = basic_data.get('last_active_glyph_char')
        self._last_active_glyph_is_vrt2_from_load = basic_data.get('last_active_glyph_is_vrt2', False)


    def _select_initial_glyph_after_full_load(self):
        if not self.current_project_path or self._project_loading_in_progress: return
        char_to_load_initially: Optional[str] = None
        load_as_vrt2_initially = False

        # Try to load last active glyph
        if self._last_active_glyph_char_from_load:
            char_key = self._last_active_glyph_char_from_load
            is_vrt2 = self._last_active_glyph_is_vrt2_from_load
            # Check if this char is still valid in the respective model
            model_to_check = self.glyph_grid_widget.vrt2_glyph_model if is_vrt2 else self.glyph_grid_widget.std_glyph_model
            if model_to_check.get_flat_index_of_char_key(char_key) is not None:
                char_to_load_initially = char_key
                load_as_vrt2_initially = is_vrt2
        
        if not char_to_load_initially: # Fallback to first navigable glyph
            first_nav_info = self.glyph_grid_widget.get_first_navigable_glyph_info()
            if first_nav_info:
                char_to_load_initially, load_as_vrt2_initially = first_nav_info
        
        self.drawing_editor_widget.update_vrt2_controls(False, False)
        if char_to_load_initially:
            self.load_glyph_for_editing(char_to_load_initially, is_vrt2_edit_mode=load_as_vrt2_initially)
        else: 
            adv_width = DEFAULT_ADVANCE_WIDTH 
            self.drawing_editor_widget.canvas.load_glyph("", None, None, adv_width, is_vrt2=False)
            self.glyph_grid_widget.set_active_glyph(None)
            self.drawing_editor_widget.update_unicode_display(None)
            self.drawing_editor_widget._update_adv_width_ui_no_signal(adv_width)
        self._update_bookmark_button_state()

    @Slot(int, str)
    def _on_load_progress(self, progress: int, message: str): # (変更なし)
        if self._project_loading_in_progress: self.statusBar().showMessage(f"{message} ({progress}%)", 0) 

    @Slot(str)
    def _on_project_load_error(self, error_message: str): # (キャッシュクリア変更)
        QMessageBox.critical(self, "プロジェクト読み込みエラー", f"プロジェクトの読み込みに失敗しました:\n{error_message}")
        self.current_project_path = None; self.db_manager.db_path = None 
        self.project_glyph_chars_cache.clear(); 
        # self._all_standard_glyphs_pixmap_cache.clear() # No longer needed
        # self._all_nrvg_pixmap_cache.clear() # No longer needed
        self.non_rotated_vrt2_chars.clear()
        self.statusBar().showMessage("プロジェクトの読み込みに失敗しました。", 5000)

    def _save_current_advance_width_sync(self, character: str, advance_width: int): # (変更なし)
        if not self.current_project_path or not character: return
        try:
            self.db_manager.save_glyph_advance_width(character, advance_width)
        except Exception as e: pass

    @Slot(str) 
    @Slot(str, bool) 
    def load_glyph_for_editing(self, character: str, is_vrt2_edit_mode: bool = False): # (SETTING_LAST_ACTIVE_GLYPH_IS_VRT2 save added)
        if self._project_loading_in_progress: return
        current_canvas_char = self.drawing_editor_widget.canvas.current_glyph_character
        current_canvas_adv_width = self.drawing_editor_widget.adv_width_spinbox.value() 
        if current_canvas_char: 
            if current_canvas_char != character or \
               (current_canvas_char == character and is_vrt2_edit_mode != self.drawing_editor_widget.canvas.editing_vrt2_glyph):
                 if not self.drawing_editor_widget.canvas.editing_vrt2_glyph:
                     self._save_current_advance_width_sync(current_canvas_char, current_canvas_adv_width)
        if self._kanji_viewer_data_loaded_successfully and character and len(character) == 1:
            with QMutexLocker(self._worker_management_mutex): self._kv_char_to_update = character 
            self._kv_deferred_update_timer.start(self._kv_update_delay_ms) 
        elif self.kanji_viewer_display_label : 
            self.kanji_viewer_display_label.setText(character if character else ""); self.kanji_viewer_display_label.setFont(QFont()) 
            if self.kanji_viewer_related_tabs: self.kanji_viewer_related_tabs.clear()
            if self._kv_deferred_update_timer.isActive(): self._kv_deferred_update_timer.stop()
            with QMutexLocker(self._worker_management_mutex): self._kv_char_to_update = None
        if not self.current_project_path or not character: 
            adv_width = DEFAULT_ADVANCE_WIDTH
            self.drawing_editor_widget.canvas.load_glyph("", None, None, adv_width, is_vrt2=False)
            self.drawing_editor_widget.set_enabled_controls(False)
            self.glyph_grid_widget.set_active_glyph(None)
            self.drawing_editor_widget.update_unicode_display(None)
            self.drawing_editor_widget._update_adv_width_ui_no_signal(adv_width)
            self.drawing_editor_widget.update_vrt2_controls(False, False); return
        adv_width = self.db_manager.load_glyph_advance_width(character)
        pixmap = self.db_manager.load_glyph_image(character, is_vrt2=is_vrt2_edit_mode)
        reference_pixmap = None
        if is_vrt2_edit_mode: reference_pixmap = self.db_manager.load_vrt2_glyph_reference_image(character)
        else: reference_pixmap = self.db_manager.load_reference_image(character)
        self.drawing_editor_widget.canvas.load_glyph(character, pixmap, reference_pixmap, adv_width, is_vrt2=is_vrt2_edit_mode)
        self.drawing_editor_widget.set_enabled_controls(True)
        self.glyph_grid_widget.set_active_glyph(character, is_vrt2_source=is_vrt2_edit_mode)
        self.drawing_editor_widget.update_unicode_display(character)
        self.drawing_editor_widget._update_adv_width_ui_no_signal(adv_width)
        is_char_in_nr_vrt2_set = character in self.non_rotated_vrt2_chars
        self.drawing_editor_widget.update_vrt2_controls(is_char_in_nr_vrt2_set, is_editing_vrt2=is_vrt2_edit_mode)
        editor_is_generally_enabled = self.drawing_editor_widget.pen_button.isEnabled()
        self.drawing_editor_widget.vrt2_toggle_button.setEnabled(editor_is_generally_enabled and is_char_in_nr_vrt2_set)
        self.drawing_editor_widget.transfer_to_vrt2_button.setEnabled(editor_is_generally_enabled and is_char_in_nr_vrt2_set)
        self.drawing_editor_widget.delete_ref_button.setEnabled(editor_is_generally_enabled and reference_pixmap is not None)
        self.drawing_editor_widget.load_ref_button.setEnabled(editor_is_generally_enabled)
        if hasattr(self.drawing_editor_widget, 'glyph_to_ref_reset_button'):
            current_glyph_has_content_on_canvas = self.drawing_editor_widget.canvas.image and not self.drawing_editor_widget.canvas.image.isNull() 
            self.drawing_editor_widget.glyph_to_ref_reset_button.setEnabled(editor_is_generally_enabled and current_glyph_has_content_on_canvas)
        if character and not self._project_loading_in_progress:
            self.save_gui_setting_async(SETTING_LAST_ACTIVE_GLYPH, character)
            self.save_gui_setting_async(SETTING_LAST_ACTIVE_GLYPH_IS_VRT2, str(is_vrt2_edit_mode))# Save VRT2 state
        # self.drawing_editor_widget.canvas.setFocus() 

    @Slot(str, QPixmap, bool)
    def handle_glyph_modification_from_canvas(self, character: str, pixmap: QPixmap, is_vrt2: bool): # (変更なし)
        if not self.current_project_path or self._project_loading_in_progress: return
        worker = SaveGlyphWorker(self.current_project_path, character, pixmap, is_vrt2_glyph=is_vrt2)
        worker.signals.result.connect(self.on_glyph_save_success)
        worker.signals.error.connect(self.on_glyph_save_error)
        self.thread_pool.start(worker)
        if hasattr(self.drawing_editor_widget, 'glyph_to_ref_reset_button'):
            has_content = pixmap and not pixmap.isNull() 
            can_enable_button = self.drawing_editor_widget.pen_button.isEnabled() and has_content
            self.drawing_editor_widget.glyph_to_ref_reset_button.setEnabled(can_enable_button)

    @Slot(str, QPixmap, bool)
    def on_glyph_save_success(self, character: str, saved_pixmap: QPixmap, is_vrt2_glyph: bool): # (Uses new grid update method)
        if self._project_loading_in_progress: return 
        if self.current_project_path: 
             self.glyph_grid_widget.update_glyph_preview(character, saved_pixmap, is_vrt2_source=is_vrt2_glyph)

    @Slot(str)
    def on_glyph_save_error(self, error_message: str): # (変更なし)
        QMessageBox.warning(self, "保存エラー", f"グリフの保存中にエラーが発生しました:\n{error_message}")
        self.statusBar().showMessage(f"グリフ保存エラー: {error_message[:100]}...", 5000)

    @Slot(str, QPixmap, bool) 
    def save_reference_image_async(self, character: str, pixmap: QPixmap, is_vrt2: bool): # (変更なし)
        if self.current_project_path and character and not self._project_loading_in_progress:
            worker = SaveReferenceImageWorker(self.current_project_path, character, pixmap, is_vrt2_glyph=is_vrt2)
            worker.signals.result.connect(self.on_reference_image_save_success)
            worker.signals.error.connect(self.on_reference_image_save_error)
            self.thread_pool.start(worker)

    @Slot(str, bool) 
    def handle_delete_reference_image_async(self, character: str, is_vrt2: bool): # (変更なし)
        if self.current_project_path and character and not self._project_loading_in_progress:
            worker = SaveReferenceImageWorker(self.current_project_path, character, None, is_vrt2_glyph=is_vrt2) 
            worker.signals.result.connect(self.on_reference_image_save_success)
            worker.signals.error.connect(self.on_reference_image_save_error)
            self.thread_pool.start(worker)

    @Slot(str, QPixmap, bool) 
    def on_reference_image_save_success(self, character: str, saved_pixmap: Optional[QPixmap], is_vrt2_saved: bool): # (変更なし)
        if self._project_loading_in_progress: return
        current_canvas_char = self.drawing_editor_widget.canvas.current_glyph_character
        current_canvas_is_vrt2_editing = self.drawing_editor_widget.canvas.editing_vrt2_glyph
        if character == current_canvas_char and is_vrt2_saved == current_canvas_is_vrt2_editing:
            self.drawing_editor_widget.canvas.reference_image = saved_pixmap.copy() if saved_pixmap else None
            self.drawing_editor_widget.canvas.update()
            self.drawing_editor_widget.delete_ref_button.setEnabled(self.drawing_editor_widget.load_ref_button.isEnabled() and (saved_pixmap is not None))
        status_message = f"下書き画像 '{character}' " + ("保存完了" if saved_pixmap else "削除完了")
        self.statusBar().showMessage(status_message, 3000)

    @Slot(str)
    def on_reference_image_save_error(self, error_message: str): # (変更なし)
        QMessageBox.warning(self, "下書き画像保存エラー", f"下書き画像の保存中にエラーが発生しました:\n{error_message}")
        self.statusBar().showMessage(f"下書き画像保存エラー: {error_message[:100]}...", 5000)

    @Slot(str, str)
    def save_gui_setting_async(self, key: str, value: str): # (変更なし)
        if self.current_project_path and not self._project_loading_in_progress:
            worker = SaveGuiStateWorker(self.current_project_path, key, value)
            worker.signals.error.connect(self.on_gui_save_error) 
            self.thread_pool.start(worker)
    
    @Slot(str)
    def on_gui_save_error(self, error_message: str): # (変更なし)
        self.statusBar().showMessage(f"GUI設定保存エラー: {error_message[:100]}...", 3000)

    @Slot(str, int)
    def save_glyph_advance_width_async(self, character: str, advance_width: int): # (変更なし)
        if self.current_project_path and character and not self._project_loading_in_progress:
            worker = SaveAdvanceWidthWorker(self.current_project_path, character, advance_width)
            worker.signals.error.connect(self.on_gui_save_error) 
            self.thread_pool.start(worker)

    def _process_char_set_string(self, char_string: str) -> List[str]: # (変更なし)
        seen = set(); unique_ordered_chars = []
        for c in char_string:
            if len(c) == 1 and c not in seen: unique_ordered_chars.append(c); seen.add(c)
        return sorted(unique_ordered_chars, key=ord)

    @Slot(str)
    def update_project_character_set(self, new_char_string: str): # (Reload logic adapted)
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。"); return
        processed_chars = self._process_char_set_string(new_char_string)
        try:
            self.db_manager.update_project_character_set(processed_chars) 
            self._set_project_loading_state(True) 
            # self._all_standard_glyphs_pixmap_cache.clear(); self._all_nrvg_pixmap_cache.clear() # No longer needed
            self.glyph_grid_widget.clear_grid_and_models() # Changed
            worker = LoadProjectWorker(self.current_project_path)
            worker.signals.basic_info_loaded.connect(self._on_project_basic_info_loaded)
            # worker.signals.glyph_batch_loaded.connect(self._on_glyph_batch_loaded) # No longer needed
            worker.signals.load_progress.connect(self._on_load_progress)
            worker.signals.error.connect(self._on_project_load_error)
            worker.signals.finished.connect(self._check_and_finalize_loading_state)
            self.thread_pool.start(worker)
            QMessageBox.information(self, "文字セット更新", "プロジェクトの文字セットが更新されました。データ再読み込み中です。")
        except Exception as e: 
            QMessageBox.critical(self, "文字セット更新エラー", f"文字セットの更新に失敗しました: {e}")
            self._set_project_loading_state(False) 

    @Slot(str)
    def update_rotated_vrt2_set(self, new_char_string: str): # (Reload logic adapted)
        if not self.current_project_path or self._project_loading_in_progress: 
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。"); return
        processed_chars = self._process_char_set_string(new_char_string)
        try:
            self.db_manager.update_rotated_vrt2_character_set(processed_chars)
            self._set_project_loading_state(True)
            # self._all_standard_glyphs_pixmap_cache.clear(); self._all_nrvg_pixmap_cache.clear() # No longer needed
            self.glyph_grid_widget.clear_grid_and_models() # Changed
            worker = LoadProjectWorker(self.current_project_path)
            worker.signals.basic_info_loaded.connect(self._on_project_basic_info_loaded)
            worker.signals.load_progress.connect(self._on_load_progress)
            worker.signals.error.connect(self._on_project_load_error)
            worker.signals.finished.connect(self._check_and_finalize_loading_state)
            self.thread_pool.start(worker)
            QMessageBox.information(self, "縦書き文字セット更新", "回転縦書き文字セットが更新されました。データ再読み込み中です。")
        except Exception as e: 
            QMessageBox.critical(self, "縦書き文字セット更新エラー", f"セットの更新に失敗: {e}")
            self._set_project_loading_state(False)

    @Slot(str)
    def update_non_rotated_vrt2_set(self, new_char_string: str): # (Reload logic adapted)
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。"); return
        processed_chars = self._process_char_set_string(new_char_string)
        new_nr_vrt2_set = set(processed_chars)
        r_vrt2_set_from_db = set(self.db_manager.get_rotated_vrt2_character_set())
        conflicts_resolved_r_vrt2 = list(r_vrt2_set_from_db - new_nr_vrt2_set) 
        try:
            self.db_manager.update_non_rotated_vrt2_character_set(list(new_nr_vrt2_set))
            if len(conflicts_resolved_r_vrt2) != len(r_vrt2_set_from_db): 
                self.db_manager.update_rotated_vrt2_character_set(conflicts_resolved_r_vrt2)
            self._set_project_loading_state(True)
            # self._all_standard_glyphs_pixmap_cache.clear(); self._all_nrvg_pixmap_cache.clear() # No longer needed
            self.glyph_grid_widget.clear_grid_and_models() # Changed
            worker = LoadProjectWorker(self.current_project_path)
            worker.signals.basic_info_loaded.connect(self._on_project_basic_info_loaded)
            worker.signals.load_progress.connect(self._on_load_progress)
            worker.signals.error.connect(self._on_project_load_error)
            worker.signals.finished.connect(self._check_and_finalize_loading_state)
            self.thread_pool.start(worker)
            QMessageBox.information(self, "非回転縦書き文字セット更新", "非回転縦書き文字セットが更新されました。データ再読み込み中です。")
        except Exception as e: 
            QMessageBox.critical(self, "非回転縦書き文字セット更新エラー", f"セットの更新に失敗: {e}")
            self._set_project_loading_state(False)

    @Slot(str)
    def update_font_name(self, name: str): # (変更なし)
        if not self.current_project_path or self._project_loading_in_progress: return
        self.save_gui_setting_async(SETTING_FONT_NAME, name)
    @Slot(str)
    def update_font_weight(self, weight: str): # (変更なし)
        if not self.current_project_path or self._project_loading_in_progress: return
        self.save_gui_setting_async(SETTING_FONT_WEIGHT, weight)

    @Slot(bool)
    def handle_vrt2_edit_mode_toggle(self, is_editing_vrt2: bool): # (変更なし)
        if self._project_loading_in_progress: return
        current_char = self.drawing_editor_widget.canvas.current_glyph_character
        if current_char: self.load_glyph_for_editing(current_char, is_vrt2_edit_mode=is_editing_vrt2)

    @Slot()
    def handle_transfer_to_vrt2(self): # (変更なし)
        current_char = self.drawing_editor_widget.canvas.current_glyph_character
        if not current_char or not self.current_project_path or self._project_loading_in_progress: return
        current_adv_width_from_ui = self.drawing_editor_widget.adv_width_spinbox.value()
        if not self.drawing_editor_widget.canvas.editing_vrt2_glyph: 
            self._save_current_advance_width_sync(current_char, current_adv_width_from_ui)
        self.drawing_editor_widget.transfer_to_vrt2_button.setEnabled(False)
        self.drawing_editor_widget.vrt2_toggle_button.setEnabled(False); QApplication.processEvents() 
        standard_pixmap = self.db_manager.load_glyph_image(current_char, is_vrt2=False)
        if not standard_pixmap: 
            standard_pixmap = QPixmap(self.drawing_editor_widget.canvas.image_size); standard_pixmap.fill(QColor(Qt.white))
        try:
            pixmap_for_worker = standard_pixmap.copy() 
            vrt2_save_worker = SaveGlyphWorker(self.current_project_path, current_char, pixmap_for_worker, is_vrt2_glyph=True)
            vrt2_save_worker.signals.result.connect(self._handle_transfer_to_vrt2_result)
            vrt2_save_worker.signals.error.connect(self._handle_transfer_to_vrt2_error)
            vrt2_save_worker.signals.finished.connect(self._reenable_vrt2_buttons_after_transfer) 
            self.thread_pool.start(vrt2_save_worker)
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "転送エラー", f"'{current_char}' の縦書きグリフへのデータ転送開始時にエラーが発生しました: {e}\n{traceback.format_exc()}")
            self._reenable_vrt2_buttons_after_transfer() 

    def _handle_transfer_to_vrt2_result(self, character: str, saved_pixmap: QPixmap, is_vrt2_glyph: bool): # (Uses new grid update)
        if not is_vrt2_glyph or self._project_loading_in_progress: return 
        if not self.current_project_path: return
        self.glyph_grid_widget.update_glyph_preview(character, saved_pixmap, is_vrt2_source=True) # Changed
        if self.drawing_editor_widget.canvas.editing_vrt2_glyph and \
           self.drawing_editor_widget.canvas.current_glyph_character == character:
            self.load_glyph_for_editing(character, is_vrt2_edit_mode=True)
        self.statusBar().showMessage(f"'{character}' の縦書きグリフへの転送成功。", 3000)

    def _handle_transfer_to_vrt2_error(self, error_message: str): # (変更なし)
        current_char_display = self.drawing_editor_widget.canvas.current_glyph_character or "選択中の文字"
        QMessageBox.critical(self, "転送エラー", f"'{current_char_display}' の縦書きグリフへのデータ転送中にエラーが発生しました: {error_message}")

    def _reenable_vrt2_buttons_after_transfer(self): # (変更なし)
        is_editor_generally_enabled = self.drawing_editor_widget.pen_button.isEnabled()
        char_for_vrt2_check = self.drawing_editor_widget.canvas.current_glyph_character
        is_char_in_nr_set_and_valid = (self.current_project_path is not None and char_for_vrt2_check and char_for_vrt2_check in self.non_rotated_vrt2_chars)
        if hasattr(self.drawing_editor_widget, 'transfer_to_vrt2_button'): 
            self.drawing_editor_widget.transfer_to_vrt2_button.setEnabled(is_editor_generally_enabled and is_char_in_nr_set_and_valid)
        if hasattr(self.drawing_editor_widget, 'vrt2_toggle_button'):
            self.drawing_editor_widget.vrt2_toggle_button.setEnabled(is_editor_generally_enabled and is_char_in_nr_set_and_valid)
        current_status_msg = self.statusBar().currentMessage(); char_display_name = char_for_vrt2_check or "操作"
        if not (current_status_msg.endswith("転送成功。") or "転送中にエラーが発生しました" in current_status_msg or "データ転送開始時にエラーが発生しました" in current_status_msg):
            self.statusBar().showMessage(f"'{char_display_name}' の縦書きグリフ転送処理完了。", 3000)

    @Slot(bool) 
    def handle_glyph_to_reference_and_reset(self, is_vrt2_target: bool): # (変更なし)
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。"); return
        canvas = self.drawing_editor_widget.canvas; current_char = canvas.current_glyph_character
        if not current_char: QMessageBox.information(self, "操作不可", "グリフが選択されていません。"); return
        if canvas.editing_vrt2_glyph != is_vrt2_target:
            QMessageBox.critical(self, "内部エラー", "グリフを下書きへ転送操作でモード不一致。"); return
        current_glyph_pixmap = canvas.get_current_image()
        if current_glyph_pixmap.isNull(): QMessageBox.information(self, "情報", "グリフに描画内容がありません。"); return
        self.drawing_editor_widget.glyph_to_ref_reset_button.setEnabled(False); QApplication.processEvents()
        ref_worker = SaveReferenceImageWorker(self.current_project_path, current_char, current_glyph_pixmap.copy(), is_vrt2_glyph=is_vrt2_target)
        ref_worker.signals.result.connect(self._on_glyph_to_ref_transfer_ref_save_success)
        ref_worker.signals.error.connect(lambda err: self._on_glyph_to_ref_transfer_error("下書き保存", err))
        ref_worker.signals.finished.connect(self._check_glyph_to_ref_reset_completion); self.thread_pool.start(ref_worker)
        blank_pixmap = QPixmap(canvas.image_size); blank_pixmap.fill(QColor(Qt.white))
        canvas.image = blank_pixmap.copy(); canvas._save_state_to_undo_stack(); canvas.update() 
        glyph_worker = SaveGlyphWorker(self.current_project_path, current_char, blank_pixmap.copy(), is_vrt2_glyph=is_vrt2_target)
        glyph_worker.signals.result.connect(self._on_glyph_to_ref_transfer_glyph_reset_success)
        glyph_worker.signals.error.connect(lambda err: self._on_glyph_to_ref_transfer_error("グリフ白紙化", err))
        glyph_worker.signals.finished.connect(self._check_glyph_to_ref_reset_completion); self.thread_pool.start(glyph_worker)
        self._glyph_to_ref_reset_op_count = 2 

    _glyph_to_ref_reset_op_count = 0 

    def _on_glyph_to_ref_transfer_ref_save_success(self, character: str, saved_ref_pixmap: Optional[QPixmap], is_vrt2_ref_saved: bool): # (変更なし)
        if self._project_loading_in_progress: return
        canvas = self.drawing_editor_widget.canvas
        if canvas.current_glyph_character == character and canvas.editing_vrt2_glyph == is_vrt2_ref_saved:
            canvas.reference_image = saved_ref_pixmap.copy() if saved_ref_pixmap else None; canvas.update()
            self.drawing_editor_widget.delete_ref_button.setEnabled(saved_ref_pixmap is not None)

    def _on_glyph_to_ref_transfer_glyph_reset_success(self, character: str, reset_glyph_pixmap: QPixmap, is_vrt2_glyph_reset: bool): # (Uses new grid update)
        if self._project_loading_in_progress: return
        self.glyph_grid_widget.update_glyph_preview(character, reset_glyph_pixmap, is_vrt2_source=is_vrt2_glyph_reset) # Changed
        if hasattr(self.drawing_editor_widget, 'glyph_to_ref_reset_button'):
             self.drawing_editor_widget.glyph_to_ref_reset_button.setEnabled(False) 

    def _on_glyph_to_ref_transfer_error(self, operation_name: str, error_message: str): # (変更なし)
        QMessageBox.warning(self, f"{operation_name}エラー", f"{operation_name}中にエラーが発生しました:\n{error_message}")

    def _check_glyph_to_ref_reset_completion(self): # (変更なし)
        self._glyph_to_ref_reset_op_count -= 1
        if self._glyph_to_ref_reset_op_count == 0:
            self.statusBar().showMessage("グリフを下書きへ転送しリセットしました。", 3000)
            if hasattr(self.drawing_editor_widget, 'glyph_to_ref_reset_button'):
                is_editor_enabled = self.drawing_editor_widget.pen_button.isEnabled()
                self.drawing_editor_widget.glyph_to_ref_reset_button.setEnabled(is_editor_enabled and False)

    @Slot()
    def handle_export_font(self): # (変更なし)
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。"); return
        if self.export_process and self.export_process.state() != QProcess.NotRunning:
            QMessageBox.information(self, "情報", "フォント書き出し処理が既に実行中です。"); return
        self.original_export_button_state = self.properties_widget.export_font_button.isEnabled()
        self.properties_widget.export_font_button.setEnabled(False)
        self.statusBar().showMessage("フォント書き出し中...", 0); QApplication.processEvents() 
        try:
            self.export_process = QProcess(self); self.export_process.setProcessChannelMode(QProcess.MergedChannels)
            script_dir = Path(sys.argv[0]).resolve().parent; db2otf_script_path = script_dir / "DB2OTF.py"
            if not db2otf_script_path.exists():
                QMessageBox.critical(self, "エラー", f"スクリプト {db2otf_script_path} が見つかりません。")
                self._cleanup_after_export(); return
            python_executable = sys.executable 
            arguments = [str(db2otf_script_path), "--db_path", self.current_project_path]
            self.export_process.finished.connect(self._on_export_process_finished)
            self.export_process.errorOccurred.connect(self._on_export_process_error)
            self.export_process.start(python_executable, arguments)
        except Exception as e:
            import traceback
            QMessageBox.critical(self, "書き出しエラー", f"フォント書き出しの開始中に予期せぬエラーが発生しました: {e}")
            self._cleanup_after_export()

    @Slot(int, QProcess.ExitStatus)
    def _on_export_process_finished(self, exitCode: int, exitStatus: QProcess.ExitStatus): # (変更なし)
        if not self.export_process: return
        output_bytes = self.export_process.readAllStandardOutput() 
        try: output = output_bytes.data().decode(sys.stdout.encoding if sys.stdout.encoding else 'utf-8', errors='replace')
        except UnicodeDecodeError: output = output_bytes.data().decode('latin-1', errors='replace') 
        if exitStatus == QProcess.NormalExit and exitCode == 0:
            QMessageBox.information(self, "書き出し完了", "フォントの書き出しが完了しました。")
            self.statusBar().showMessage("フォントの書き出しが完了しました。", 5000)
        else:
            error_reason = "クラッシュしました" if exitStatus == QProcess.CrashExit else f"エラー終了コード: {exitCode}"
            msg_box = QMessageBox(self); msg_box.setIcon(QMessageBox.Warning); msg_box.setWindowTitle("書き出しエラー")
            msg_box.setText(f"フォント書き出し中にエラーが発生しました ({error_reason})。")
            msg_box.setInformativeText(f"終了コード: {exitCode}, 終了ステータス: {exitStatus.name}") 
            msg_box.setDetailedText(output if output.strip() else "出力はありませんでした。"); msg_box.exec()
            self.statusBar().showMessage(f"フォント書き出しエラー ({error_reason})", 5000)
        self._cleanup_after_export()

    @Slot(QProcess.ProcessError)
    def _on_export_process_error(self, error: QProcess.ProcessError): # (変更なし)
        if not self.export_process: return
        error_string = self.export_process.errorString()
        QMessageBox.critical(self, "書き出しプロセスエラー", f"フォント書き出しプロセスの実行に失敗しました。\nエラータイプ: {error.name}\n詳細: {error_string}")
        self.statusBar().showMessage("フォント書き出しプロセス失敗。", 5000); self._cleanup_after_export()

    def _cleanup_after_export(self): # (変更なし)
        project_still_loaded = self.current_project_path is not None and not self._project_loading_in_progress
        can_be_enabled_after_export = project_still_loaded and self.original_export_button_state
        self.properties_widget.export_font_button.setEnabled(can_be_enabled_after_export)
        if self.statusBar().currentMessage() == "フォント書き出し中...": self.statusBar().clearMessage()
        if self.export_process:
            try: self.export_process.finished.disconnect(self._on_export_process_finished)
            except RuntimeError: pass 
            try: self.export_process.errorOccurred.disconnect(self._on_export_process_error)
            except RuntimeError: pass
            self.export_process.deleteLater(); self.export_process = None


    def keyPressEvent(self, event: QKeyEvent):
        if self._project_loading_in_progress:
            event.ignore()
            return

        # Standard Undo/Redo shortcuts
        if event.matches(QKeySequence.StandardKey.Undo):
            if self.drawing_editor_widget.undo_button.isEnabled():
                self.drawing_editor_widget.canvas.undo()
            event.accept(); return
        if event.matches(QKeySequence.StandardKey.Redo):
            if self.drawing_editor_widget.redo_button.isEnabled():
                self.drawing_editor_widget.canvas.redo()
            event.accept(); return

        focus_widget = QApplication.focusWidget()

        if isinstance(focus_widget, (QLineEdit, QTextEdit, QSpinBox)):
            super().keyPressEvent(event); return

        if self.drawing_editor_widget.pen_button.isEnabled() and \
           self.drawing_editor_widget.canvas.current_glyph_character:
            if not event.isAutoRepeat():
                key = event.key()
                if key == Qt.Key_E: self.drawing_editor_widget.eraser_button.click(); event.accept(); return
                if key == Qt.Key_B: self.drawing_editor_widget.pen_button.click(); event.accept(); return
                if key == Qt.Key_V: self.drawing_editor_widget.move_button.click(); event.accept(); return

        key = event.key()
        if key in [Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down]:
            # Check if focus is on one of the QTableViews or their parent GlyphGridWidget
            if focus_widget == self.glyph_grid_widget.current_active_view or \
               focus_widget == self.glyph_grid_widget:
                # Pass the event to GlyphGridWidget's custom handler
                self.glyph_grid_widget.keyPressEvent(event)
                if event.isAccepted():
                    return
        
        super().keyPressEvent(event)

    def open_batch_advance_width_dialog(self): # (変更なし)
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。"); return
        dialog = BatchAdvanceWidthDialog(self)
        if dialog.exec() == QDialog.Accepted:
            char_spec, adv_width = dialog.get_values()
            if char_spec is not None and adv_width is not None: self._apply_batch_advance_width(char_spec, adv_width)

    def _parse_char_specification(self, char_spec: str) -> Tuple[Optional[List[str]], Optional[str]]: # (変更なし)
        target_chars_set: Set[str] = set(); original_spec = char_spec; temp_spec = char_spec 
        if ".notdef" in temp_spec.lower():
            target_chars_set.add(".notdef"); temp_spec = re.sub(r"\.notdef", "", temp_spec, flags=re.IGNORECASE) 
        range_matches = list(re.finditer(r"U\+([0-9A-Fa-f]{4,6})-U\+([0-9A-Fa-f]{4,6})", temp_spec, re.IGNORECASE))
        for match in reversed(range_matches): 
            try:
                start_hex, end_hex = match.group(1), match.group(2); start_ord, end_ord = int(start_hex, 16), int(end_hex, 16)
                if start_ord > end_ord: return None, f"Unicode範囲の開始値が終了値より大きいです: U+{start_hex}-U+{end_hex}"
                for i in range(start_ord, end_ord + 1): target_chars_set.add(chr(i))
                temp_spec = temp_spec[:match.start()] + temp_spec[match.end():] 
            except (ValueError, OverflowError): return None, f"無効なUnicodeコードポイントが範囲内に含まれています: U+{match.group(0)}"
        unicode_matches = list(re.finditer(r"U\+([0-9A-Fa-f]{4,6})", temp_spec, re.IGNORECASE))
        for match in reversed(unicode_matches): 
            try:
                hex_code = match.group(1); target_chars_set.add(chr(int(hex_code, 16)))
                temp_spec = temp_spec[:match.start()] + temp_spec[match.end():]
            except (ValueError, OverflowError): return None, f"無効なUnicodeコードポイントが含まれています: U+{match.group(0)}"
        cleaned_literals = re.sub(r"[\s,;\t]+", "", temp_spec) 
        for char_val in cleaned_literals:
            if len(char_val) == 1 : target_chars_set.add(char_val) 
        if not target_chars_set and original_spec: return None, "有効な文字またはUnicode指定が見つかりませんでした。"
        return sorted(list(target_chars_set), key=lambda x: (-1, x) if x == '.notdef' else (ord(x) if len(x)==1 else float('inf'),x )), None

    def _apply_batch_advance_width(self, char_spec: str, new_adv_width: int): # (変更なし)
        if not self.current_project_path or self._project_loading_in_progress: return
        target_chars_list, error_msg = self._parse_char_specification(char_spec)
        if error_msg: QMessageBox.warning(self, "入力エラー", error_msg); return
        if not target_chars_list: QMessageBox.information(self, "情報", "適用対象の有効な文字が見つかりませんでした。"); return
        if not self.project_glyph_chars_cache: 
            all_glyphs_data_raw = self.db_manager.get_all_glyphs_with_preview_data() 
            self.project_glyph_chars_cache = {char_data[0] for char_data in all_glyphs_data_raw}
            if not self.project_glyph_chars_cache and target_chars_list: 
                QMessageBox.warning(self, "エラー", "プロジェクトにグリフが読み込まれていないか空です。"); return
        updated_count = 0; skipped_count = 0
        current_char_in_editor = self.drawing_editor_widget.canvas.current_glyph_character
        conn = self.db_manager._get_connection(); cursor = conn.cursor()
        try:
            for char_to_update in target_chars_list:
                if char_to_update in self.project_glyph_chars_cache: 
                    cursor.execute("UPDATE glyphs SET advance_width = ? WHERE character = ?", (new_adv_width, char_to_update))
                    if cursor.rowcount > 0: updated_count += 1
                else: skipped_count += 1
            conn.commit()
        except Exception as e: conn.rollback(); QMessageBox.critical(self, "一括更新エラー", f"データベース更新中にエラーが発生しました: {e}"); return
        finally: conn.close()
        summary_message = f"{updated_count} 文字の送り幅を {new_adv_width} に更新しました。"
        if skipped_count > 0: summary_message += f"\n{skipped_count} 文字はプロジェクトに含まれていないためスキップされました。"
        QMessageBox.information(self, "一括更新完了", summary_message)
        if current_char_in_editor and not self.drawing_editor_widget.canvas.editing_vrt2_glyph and \
           current_char_in_editor in target_chars_list and current_char_in_editor in self.project_glyph_chars_cache: 
            self.drawing_editor_widget.canvas.current_glyph_advance_width = new_adv_width
            self.drawing_editor_widget._update_adv_width_ui_no_signal(new_adv_width) 
            self.drawing_editor_widget.canvas.update() 

    def batch_import_glyphs(self): self._process_batch_image_import(import_type="glyph") # (grid update changed)
    def batch_import_reference_images(self): self._process_batch_image_import(import_type="reference") # (grid update changed)

    def _process_batch_image_import(self, import_type: str): # (Grid update changed)
        if not self.current_project_path or self._project_loading_in_progress:
            QMessageBox.warning(self, "エラー", "プロジェクトが開かれていないか、処理中です。"); return
        file_dialog_title = "グリフ画像の一括読み込み" if import_type == "glyph" else "下書き画像の一括読み込み"
        file_paths, _ = QFileDialog.getOpenFileNames(self, file_dialog_title, "", "画像ファイル (*.png *.jpg *.jpeg *.bmp)")
        if not file_paths: return
        target_image_size = QSize(CANVAS_IMAGE_WIDTH, CANVAS_IMAGE_HEIGHT)
        if not self.project_glyph_chars_cache: 
            all_glyphs_data_raw = self.db_manager.get_all_glyphs_with_preview_data()
            self.project_glyph_chars_cache = {char_data[0] for char_data in all_glyphs_data_raw}
        if import_type == "reference" and not self.non_rotated_vrt2_chars: 
            self.non_rotated_vrt2_chars = set(self.db_manager.get_non_rotated_vrt2_character_set())
        processed_count = 0; skipped_filename_format = 0; skipped_unicode_conversion = 0
        skipped_char_not_in_project = 0; skipped_image_load_error = 0; db_update_errors = 0
        total_files = len(file_paths)
        self.statusBar().showMessage(f"{file_dialog_title} を開始します ({total_files} ファイル)...", 0); QApplication.processEvents()
        for idx, file_path_str in enumerate(file_paths):
            file_path = Path(file_path_str)
            self.statusBar().showMessage(f"処理中 ({idx+1}/{total_files}): {file_path.name}", 0); QApplication.processEvents()
            stem = file_path.stem; is_vrt2_ref_candidate = False; hex_code = "" 
            if import_type == "reference":
                match_vert = re.fullmatch(r"uni([0-9A-Fa-f]{4,6})vert", stem, re.IGNORECASE)
                if not match_vert:
                     match_vert_uplus = re.fullmatch(r"U\+([0-9A-Fa-f]{4,6})vert", stem, re.IGNORECASE)
                     if match_vert_uplus: hex_code = match_vert_uplus.group(1); is_vrt2_ref_candidate = True
                else: hex_code = match_vert.group(1); is_vrt2_ref_candidate = True
            if not is_vrt2_ref_candidate: 
                match = re.fullmatch(r"uni([0-9A-Fa-f]{4,6})", stem, re.IGNORECASE)
                if not match: 
                    match_uplus = re.fullmatch(r"U\+([0-9A-Fa-f]{4,6})", stem, re.IGNORECASE)
                    if not match_uplus: skipped_filename_format += 1; continue
                    hex_code = match_uplus.group(1)
                else: hex_code = match.group(1)
            if not hex_code: skipped_filename_format += 1; continue
            try: unicode_val = int(hex_code, 16); character = chr(unicode_val)
            except (ValueError, OverflowError): skipped_unicode_conversion += 1; continue
            target_is_nrvg_for_ref = False 
            if import_type == "reference" and is_vrt2_ref_candidate:
                if character not in self.non_rotated_vrt2_chars: skipped_char_not_in_project +=1; continue
                target_is_nrvg_for_ref = True
            elif character not in self.project_glyph_chars_cache: skipped_char_not_in_project += 1; continue
            qimage = QImage(file_path_str)
            if qimage.isNull(): skipped_image_load_error += 1; continue
            scaled_image = qimage.scaled(target_image_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            final_qimage = QImage(target_image_size, QImage.Format_ARGB32_Premultiplied); final_qimage.fill(QColor(Qt.white)) 
            painter = QPainter(final_qimage)
            x_offset = (target_image_size.width() - scaled_image.width()) // 2
            y_offset = (target_image_size.height() - scaled_image.height()) // 2
            painter.drawImage(x_offset, y_offset, scaled_image); painter.end()
            processed_pixmap = QPixmap.fromImage(final_qimage) 
            byte_array = QByteArray(); buffer = QBuffer(byte_array)
            buffer.open(QIODevice.WriteOnly); final_qimage.save(buffer, "PNG"); image_data_bytes = byte_array.data()
            success = False
            if import_type == "glyph": success = self.db_manager.update_glyph_image_data_bytes(character, image_data_bytes)
            elif import_type == "reference":
                if target_is_nrvg_for_ref: success = self.db_manager.update_vrt2_glyph_reference_image_data_bytes(character, image_data_bytes)
                else: success = self.db_manager.update_glyph_reference_image_data_bytes(character, image_data_bytes)
            if success:
                processed_count += 1
                if import_type == "glyph": # Standard glyph image updated
                    self.glyph_grid_widget.update_glyph_preview(character, processed_pixmap, is_vrt2_source=False)
                current_editor_char = self.drawing_editor_widget.canvas.current_glyph_character
                is_editor_vrt2_editing = self.drawing_editor_widget.canvas.editing_vrt2_glyph
                if character == current_editor_char:
                    if import_type == "glyph" and not is_editor_vrt2_editing: 
                        self.load_glyph_for_editing(character, is_vrt2_edit_mode=False) 
                    elif import_type == "reference":
                        if target_is_nrvg_for_ref and is_editor_vrt2_editing: 
                            self.drawing_editor_widget.canvas.reference_image = processed_pixmap.copy()
                            self.drawing_editor_widget.canvas.update(); self.drawing_editor_widget.delete_ref_button.setEnabled(True)
                        elif not target_is_nrvg_for_ref and not is_editor_vrt2_editing: 
                            self.drawing_editor_widget.canvas.reference_image = processed_pixmap.copy()
                            self.drawing_editor_widget.canvas.update(); self.drawing_editor_widget.delete_ref_button.setEnabled(True)
            else: db_update_errors +=1
        self.statusBar().showMessage(f"{file_dialog_title} 完了", 5000)
        summary_parts = [f"{processed_count} 件の画像を処理・保存しました。"]
        if skipped_filename_format > 0: summary_parts.append(f"{skipped_filename_format} 件: ファイル名形式エラー")
        if skipped_unicode_conversion > 0: summary_parts.append(f"{skipped_unicode_conversion} 件: Unicode変換エラー")
        if skipped_char_not_in_project > 0: summary_parts.append(f"{skipped_char_not_in_project} 件: 文字がプロジェクトに未登録/不適切")
        if skipped_image_load_error > 0: summary_parts.append(f"{skipped_image_load_error} 件: 画像読み込みエラー")
        if db_update_errors > 0: summary_parts.append(f"{db_update_errors} 件: DB更新エラー")
        QMessageBox.information(self, "一括読み込み結果", "\n".join(summary_parts))

    def closeEvent(self, event: QEvent): # (変更なし)
        if self._project_loading_in_progress:
            reply = QMessageBox.question(self, '確認', "プロジェクトの読み込み処理が進行中です。強制終了しますか？\n（データが破損する可能性があります）", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No: event.ignore(); return
        reply = QMessageBox.question(self, '確認', "アプリケーションを終了しますか？", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if not self._project_loading_in_progress and self.current_project_path and \
               self.drawing_editor_widget.canvas.current_glyph_character and not self.drawing_editor_widget.canvas.editing_vrt2_glyph: 
                current_char = self.drawing_editor_widget.canvas.current_glyph_character
                adv_width = self.drawing_editor_widget.adv_width_spinbox.value()
                self._save_current_advance_width_sync(current_char, adv_width)
            if self._kv_deferred_update_timer and self._kv_deferred_update_timer.isActive(): self._kv_deferred_update_timer.stop()
            with QMutexLocker(self._worker_management_mutex):
                if self.related_kanji_worker and self.related_kanji_worker.isRunning(): self.related_kanji_worker.cancel() 
            if self.export_process and self.export_process.state() != QProcess.NotRunning:
                self.export_process.terminate() 
                if not self.export_process.waitForFinished(3000): self.export_process.kill(); self.export_process.waitForFinished(1000) 
            self.thread_pool.waitForDone(-1); event.accept()
        else: event.ignore()

    def resizeEvent(self, event: QResizeEvent): # (変更なし)
        super().resizeEvent(event)
        if self._project_loading_in_progress: return
        if self.kv_resize_timer and self.kv_resize_timer.isActive(): self.kv_resize_timer.stop()
        if not self.kv_resize_timer: 
            self.kv_resize_timer = QTimer(self); self.kv_resize_timer.setSingleShot(True)
            self.kv_resize_timer.timeout.connect(self._on_kv_resize_finished)
        self.kv_resize_timer.start(250) 

    def _on_kv_resize_finished(self): # (変更なし)
        if self._project_loading_in_progress: return
        if self._kanji_viewer_data_loaded_successfully and self.drawing_editor_widget.canvas.current_glyph_character:
            char_to_update = self.drawing_editor_widget.canvas.current_glyph_character
            if char_to_update and len(char_to_update) == 1: 
                with QMutexLocker(self._worker_management_mutex): self._kv_char_to_update = char_to_update
                self._kv_deferred_update_timer.start(self._kv_update_delay_ms)

if __name__ == "__main__":
    QApplication.setApplicationName("P-Glyph")
    app = QApplication(sys.argv)
    icon_path = "ico.ico" 
    if os.path.exists(icon_path): app.setWindowIcon(QIcon(icon_path))
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        icon_path_data = os.path.join(script_dir, "data", "ico.ico")
        if os.path.exists(icon_path_data): app.setWindowIcon(QIcon(icon_path_data))
        else:
            parent_dir = os.path.dirname(script_dir)
            icon_path_parent_data = os.path.join(parent_dir, "data", "ico.ico")
            if os.path.exists(icon_path_parent_data): app.setWindowIcon(QIcon(icon_path_parent_data))
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
