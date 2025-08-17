import sys
import os
import sqlite3
import argparse
import json
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Any

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTextEdit, QPushButton, QSlider, QLabel, QScrollArea, QMessageBox,
    QSizePolicy, QCheckBox
)
from PySide6.QtGui import (
    QPainter, QPixmap, QColor, QPen, QResizeEvent, QImage
)
from PySide6.QtCore import (
    Qt, QRectF, QSize, QByteArray, QTimer, Signal, QPoint # SignalとQPointを追加
)

# P-GlyphのAPIクライアントからselect_glyph関数をインポート
# 注: api_usage_sample.py が同じディレクトリにあるか、Pythonのパスが通っている必要があります。
from api_usage_sample import select_glyph

# --- 定数 ---
CANVAS_IMAGE_WIDTH = 500
CANVAS_IMAGE_HEIGHT = 500
EM_SQUARE_UNITS = 1000
SETTINGS_FILE = "metric_settings.json"

# --- データベース管理 ---
class MetricsDBManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        if not os.path.exists(self.db_path):
            raise FileNotFoundError(f"データベースファイルが見つかりません: {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        return conn

    def load_glyph_image_and_advance(self, character: str) -> Optional[Tuple[Optional[QPixmap], int, str]]:
        """画像がなくてもレコードがあればadvanceとlast_modifiedを返すように修正"""
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            is_pua = False
            try:
                if 0xE000 <= ord(character) <= 0xF8FF:
                    is_pua = True
            except TypeError: pass
            
            table = "pua_glyphs" if is_pua else "glyphs"
            cursor.execute(f"SELECT image_data, advance_width, last_modified FROM {table} WHERE character = ?", (character,))
            row = cursor.fetchone()

            if row:
                pixmap = None
                if row['image_data']:
                    pixmap = QPixmap()
                    pixmap.loadFromData(row['image_data'])
                
                advance_width = row['advance_width'] if row['advance_width'] is not None else EM_SQUARE_UNITS
                last_modified = row['last_modified'] if row['last_modified'] else ""
                return pixmap, advance_width, last_modified
            
            return None
        except sqlite3.OperationalError:
            return None
        finally:
            conn.close()

    def get_current_timestamps(self, characters: List[str]) -> Dict[str, str]:
        """指定された文字リストの現在のlast_modifiedタイムスタンプを一括で取得する"""
        if not characters:
            return {}
        
        timestamps = {}
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            placeholders = ','.join('?' for _ in characters)
            
            query = f"SELECT character, last_modified FROM glyphs WHERE character IN ({placeholders})"
            cursor.execute(query, characters)
            for row in cursor.fetchall():
                timestamps[row['character']] = row['last_modified']
            
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='pua_glyphs'")
            if cursor.fetchone():
                query_pua = f"SELECT character, last_modified FROM pua_glyphs WHERE character IN ({placeholders})"
                cursor.execute(query_pua, characters)
                for row in cursor.fetchall():
                    timestamps[row['character']] = row['last_modified']

        except sqlite3.OperationalError as e:
            print(f"DB access error in get_current_timestamps: {e}")
        finally:
            conn.close()
        return timestamps

# --- グリフ表示ウィジェット ---
class MetricsDisplayWidget(QWidget):
    # クリックされたグリフの情報を伝えるためのシグナル
    # 引数: character (str), is_pua (bool)
    glyphClicked = Signal(str, bool) 

    def __init__(self, db_manager: MetricsDBManager, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._db_manager = db_manager
        self._text_to_display: str = ""
        self._scale_factor: float = 0.25
        self._glyph_cache: Dict[str, Dict[str, Any]] = {}
        
        self._show_advance_guide: bool = True
        self._auto_wrap_text: bool = False
        self._is_inverted: bool = False
        
        self._character_spacing: int = 0
        self._line_spacing_factor: float = 1.0
        
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)

    def set_text(self, text: str):
        if self._text_to_display == text: return
        self._text_to_display = text
        self._update_glyph_cache()
        self._recalculate_and_update_geometry()
        self.update()

    def set_scale(self, scale: float):
        self._scale_factor = max(0.05, scale)
        self._recalculate_and_update_geometry()
        self.update()

    def set_show_advance_guide(self, show: bool):
        if self._show_advance_guide != show:
            self._show_advance_guide = show
            self.update()

    def set_auto_wrap(self, wrap: bool):
        if self._auto_wrap_text != wrap:
            self._auto_wrap_text = wrap
            self._recalculate_and_update_geometry()
            self.update()

    def set_color_inversion(self, invert: bool):
        if self._is_inverted != invert:
            self._is_inverted = invert
            self.update()
            
    def set_character_spacing(self, spacing: int):
        if self._character_spacing != spacing:
            self._character_spacing = spacing
            self._recalculate_and_update_geometry()
            self.update()

    def set_line_spacing(self, factor: float):
        if self._line_spacing_factor != factor:
            self._line_spacing_factor = factor
            self._recalculate_and_update_geometry()
            self.update()

    def check_and_refresh_visible_glyphs(self):
        """表示中グリフの更新をDBに問い合わせて、変更があれば再描画する。画像が新規追加された場合も検知。"""
        visible_chars = list(set(c for c in self._text_to_display if c != '\n'))
        if not visible_chars:
            return

        current_timestamps_from_db = self._db_manager.get_current_timestamps(visible_chars)
        changed_chars = []

        for char in visible_chars:
            db_timestamp = current_timestamps_from_db.get(char, "")
            cached_data = self._glyph_cache.get(char)
            
            if not cached_data or cached_data.get("last_modified") != db_timestamp:
                changed_chars.append(char)

        if changed_chars:
            unique_changed_chars = sorted(list(set(changed_chars)))
            print(f"  > Invalidating cache for: {', '.join(unique_changed_chars)}")
            self.invalidate_specific_glyphs(unique_changed_chars)

    def invalidate_specific_glyphs(self, characters: List[str]):
        needs_redraw = False
        current_chars_in_view = set(self._text_to_display.replace('\n', ''))
        for char in characters:
            if char in self._glyph_cache:
                del self._glyph_cache[char]
                if char in current_chars_in_view:
                    needs_redraw = True
        
        if needs_redraw:
            self._update_glyph_cache()
            self.update()

    def _get_pixel_advance(self, char: str) -> float:
        cached_data = self._glyph_cache.get(char)
        if cached_data:
            return (cached_data.get('advance', EM_SQUARE_UNITS) / EM_SQUARE_UNITS) * CANVAS_IMAGE_WIDTH
        return (EM_SQUARE_UNITS / EM_SQUARE_UNITS) * CANVAS_IMAGE_WIDTH

    def _get_char_total_width(self, char: str) -> float:
        advance_width = self._get_pixel_advance(char) * self._scale_factor
        scaled_spacing = (self._character_spacing / EM_SQUARE_UNITS) * CANVAS_IMAGE_WIDTH * self._scale_factor
        return advance_width + scaled_spacing

    def _update_glyph_cache(self):
        """画像なしでもタイムスタンプ等をキャッシュする"""
        unique_chars = set(self._text_to_display.replace('\n', ''))
        for char in unique_chars:
            if char not in self._glyph_cache:
                loaded_data = self._db_manager.load_glyph_image_and_advance(char)
                if loaded_data:
                    pixmap, advance, last_modified = loaded_data
                    
                    normal_pixmap, inverted_pixmap = None, None
                    if pixmap:
                        normal_pixmap = pixmap
                        qimage = pixmap.toImage()
                        qimage.invertPixels(QImage.InvertMode.InvertRgb)
                        inverted_pixmap = QPixmap.fromImage(qimage)

                    self._glyph_cache[char] = {
                        "normal": normal_pixmap, 
                        "inverted": inverted_pixmap, 
                        "advance": advance,
                        "last_modified": last_modified
                    }
                else:
                    self._glyph_cache[char] = {
                        "normal": None, "inverted": None,
                        "advance": EM_SQUARE_UNITS,
                        "last_modified": ""
                    }

    def _recalculate_and_update_geometry(self):
        margin_size = (CANVAS_IMAGE_HEIGHT * self._scale_factor) * 0.5
        scaled_line_height_with_spacing = int(CANVAS_IMAGE_HEIGHT * self._scale_factor * self._line_spacing_factor)
        
        if self._auto_wrap_text:
            self.setMinimumWidth(0)
            drawable_width = max(0, self.width() - margin_size * 2)
            lines = self._calculate_drawable_lines(drawable_width)
            height = (len(lines) * scaled_line_height_with_spacing) + margin_size * 2
            self.setMinimumHeight(max(150, int(height)))
        else:
            lines = self._calculate_drawable_lines(0)
            height = (len(lines) * scaled_line_height_with_spacing) + margin_size * 2
            max_text_width = 0
            if lines:
                max_text_width = max(sum(self._get_char_total_width(c) for c in line) for line in lines)
            width = max_text_width + margin_size * 2
            self.setMinimumSize(QSize(max(200, int(width)), max(150, int(height))))
        
        self.updateGeometry()

    def resizeEvent(self, event: QResizeEvent):
        super().resizeEvent(event)
        if self._auto_wrap_text:
            self._recalculate_and_update_geometry()

    def _calculate_drawable_lines(self, max_width: float) -> List[str]:
        drawable_lines = []
        for line_segment in self._text_to_display.split('\n'):
            if not self._auto_wrap_text or not line_segment:
                drawable_lines.append(line_segment)
                continue
            
            current_line = ""
            current_width = 0
            for char in line_segment:
                char_width = self._get_char_total_width(char)
                if max_width > 0 and current_width + char_width > max_width and len(current_line) > 0:
                    drawable_lines.append(current_line)
                    current_line = ""
                    current_width = 0
                current_line += char
                current_width += char_width
            
            if current_line or not drawable_lines or drawable_lines[-1] != "":
                drawable_lines.append(current_line)
        
        return drawable_lines

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        bg_color = Qt.GlobalColor.black if self._is_inverted else Qt.GlobalColor.white
        painter.fillRect(self.rect(), bg_color)
        
        margin_size = (CANVAS_IMAGE_HEIGHT * self._scale_factor) * 0.5
        drawable_width = max(0, self.width() - margin_size * 2)
        lines_to_draw = self._calculate_drawable_lines(drawable_width)
        
        scaled_line_height_with_spacing = int(CANVAS_IMAGE_HEIGHT * self._scale_factor * self._line_spacing_factor)
        base_scaled_line_height = int(CANVAS_IMAGE_HEIGHT * self._scale_factor)
        y_cursor = margin_size

        for line_text in lines_to_draw:
            self._paint_line(painter, line_text, margin_size, y_cursor, base_scaled_line_height)
            y_cursor += scaled_line_height_with_spacing
        
        painter.end()

    def _paint_line(self, painter: QPainter, line_to_draw: str, start_x: float, start_y: float, line_height: float):
        """【変更】文字間隔 > 0 の場合、左側にもガイドを描画するように変更"""
        x_cursor = start_x
        scaled_spacing = (self._character_spacing / EM_SQUARE_UNITS) * CANVAS_IMAGE_WIDTH * self._scale_factor
        
        for char in line_to_draw:
            pixel_advance = self._get_pixel_advance(char)
            scaled_width = pixel_advance * self._scale_factor
            cached_data = self._glyph_cache.get(char)
            
            pixmap_key = "inverted" if self._is_inverted else "normal"
            pixmap = cached_data.get(pixmap_key) if cached_data else None

            if not pixmap:
                if self._is_inverted:
                    painter.setPen(QPen(Qt.GlobalColor.magenta))
                    painter.setBrush(QColor(80, 0, 80))
                else:
                    painter.setPen(QPen(Qt.GlobalColor.red))
                    painter.setBrush(QColor(255, 200, 200))
                painter.drawRect(QRectF(x_cursor, start_y, scaled_width, line_height))
                
                # ガイド描画（未定義グリフにも適用）
                if self._show_advance_guide:
                    guide_color = Qt.GlobalColor.cyan if self._is_inverted else Qt.GlobalColor.blue
                    pen = QPen(guide_color, 1, Qt.PenStyle.DashLine)
                    painter.setPen(pen)
                    if self._character_spacing > 0:
                        painter.drawLine(int(x_cursor), int(start_y), int(x_cursor), int(start_y + line_height))
                    right_line_x = int(x_cursor + scaled_width)
                    painter.drawLine(right_line_x, int(start_y), right_line_x, int(start_y + line_height))

                x_cursor += scaled_width + scaled_spacing
                continue

            source_rect = QRectF(0, 0, pixel_advance, CANVAS_IMAGE_HEIGHT)
            dest_rect = QRectF(x_cursor, start_y, scaled_width, line_height)
            painter.drawPixmap(dest_rect, pixmap, source_rect)

            # --- ガイド描画ロジック ---
            if self._show_advance_guide:
                guide_color = Qt.GlobalColor.cyan if self._is_inverted else Qt.GlobalColor.blue
                pen = QPen(guide_color, 1, Qt.PenStyle.DashLine)
                painter.setPen(pen)

                # 【変更ここから】文字間隔 > 0 の場合、左側（描画開始位置）にもガイドを描画
                if self._character_spacing > 0:
                    left_line_x = int(x_cursor)
                    painter.drawLine(left_line_x, int(start_y), left_line_x, int(start_y + line_height))

                # 右側のガイド（文字送り幅の位置）は常に表示
                right_line_x = int(x_cursor + scaled_width)
                painter.drawLine(right_line_x, int(start_y), right_line_x, int(start_y + line_height))
                # 【変更ここまで】
            
            x_cursor += scaled_width + scaled_spacing

    def mousePressEvent(self, event):
        """マウスがクリックされたときに、クリックされたグリフを特定しシグナルを発行する"""
        if event.button() == Qt.MouseButton.LeftButton:
            clicked_char_info = self._get_char_at_pixel_pos(event.pos())
            if clicked_char_info:
                char, is_pua = clicked_char_info
                print(f"Clicked on character: '{char}' (PUA: {is_pua})")
                self.glyphClicked.emit(char, is_pua)
        super().mousePressEvent(event)

    def _get_char_at_pixel_pos(self, pos: QPoint) -> Optional[Tuple[str, bool]]:
        """
        クリックされたピクセル座標に対応する文字を特定する
        
        このロジックは、paintEvent の描画ロジックと密接に関連しているため、
        描画処理と同期して座標計算を行う必要があります。
        """
        margin_size = (CANVAS_IMAGE_HEIGHT * self._scale_factor) * 0.5
        drawable_width = max(0, self.width() - margin_size * 2)
        lines_to_draw = self._calculate_drawable_lines(drawable_width)
        
        scaled_line_height_with_spacing = int(CANVAS_IMAGE_HEIGHT * self._scale_factor * self._line_spacing_factor)
        
        y_cursor = margin_size

        for line_text in lines_to_draw:
            # 各行の描画領域を計算 (クリック位置がこの行内にあるかを確認するため)
            line_height = scaled_line_height_with_spacing # paintEvent で使われるline_height
            line_rect = QRectF(margin_size, y_cursor, drawable_width if self._auto_wrap_text else self.width() - margin_size * 2, line_height)
            
            if line_rect.contains(pos): # クリックが現在の行内にあるかチェック
                x_cursor = margin_size
                scaled_spacing = (self._character_spacing / EM_SQUARE_UNITS) * CANVAS_IMAGE_WIDTH * self._scale_factor
                
                for char in line_text:
                    pixel_advance = self._get_pixel_advance(char)
                    scaled_width = pixel_advance * self._scale_factor
                    
                    char_total_width = scaled_width + scaled_spacing # 文字の描画幅 + 文字間隔
                    char_rect = QRectF(x_cursor, y_cursor, char_total_width, line_height)
                    
                    if char_rect.contains(pos): # クリックが現在の文字内にあるかチェック
                        is_pua = False
                        try:
                            if 0xE000 <= ord(char) <= 0xF8FF: # PUA判定ロジック
                                is_pua = True
                        except TypeError: # char が単一文字でない場合など
                            pass
                        return char, is_pua
                    
                    x_cursor += char_total_width
            
            y_cursor += scaled_line_height_with_spacing
        
        return None # どの文字もクリックされなかった場合


# --- メインウィンドウ ---
class MetricsViewerWindow(QMainWindow):
    def __init__(self, db_path: str):
        super().__init__()
        self.setWindowTitle(f"フォントメトリックビューア - {os.path.basename(db_path)}")
        self.setGeometry(100, 100, 1200, 800)

        try:
            self._db_manager = MetricsDBManager(db_path)
        except FileNotFoundError as e:
            QMessageBox.critical(self, "エラー", str(e)); sys.exit(1)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self._display_widget = MetricsDisplayWidget(self._db_manager)
        # グリフクリックシグナルをスロットに接続
        self._display_widget.glyphClicked.connect(self._on_glyph_clicked) 

        self._scroll_area = QScrollArea()
        self._scroll_area.setWidget(self._display_widget)
        self._scroll_area.setWidgetResizable(True) 
        self._scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        main_layout.addWidget(self._scroll_area, 3)

        controls_widget = QWidget(); controls_widget.setFixedWidth(250)
        controls_layout = QVBoxLayout(controls_widget)

        controls_layout.addWidget(QLabel("表示するテキスト:"))
        self._text_input = QTextEdit()
        self._text_input.setPlaceholderText("ここに表示したいテキストを入力...\n(Ctrl+Enterで更新)")
        self._text_input.setAcceptRichText(False)
        self._text_input.keyPressEvent = self.text_input_key_press
        controls_layout.addWidget(self._text_input, 1)

        update_button = QPushButton("表示を更新 (Ctrl+Enter)")
        update_button.clicked.connect(self._update_display)
        controls_layout.addWidget(update_button)
        
        controls_layout.addSpacing(15)

        self._guide_checkbox = QCheckBox("文字送り幅ガイド(青線)を表示")
        self._guide_checkbox.setChecked(True)
        self._guide_checkbox.toggled.connect(self._display_widget.set_show_advance_guide)
        controls_layout.addWidget(self._guide_checkbox)

        self._wrap_checkbox = QCheckBox("ウィンドウ幅で自動改行")
        self._wrap_checkbox.setChecked(True)
        self._wrap_checkbox.toggled.connect(self._on_wrap_mode_changed)
        controls_layout.addWidget(self._wrap_checkbox)

        self._invert_color_checkbox = QCheckBox("表示色を反転する (黒背景)")
        self._invert_color_checkbox.setChecked(False)
        self._invert_color_checkbox.toggled.connect(self._on_invert_color_changed)
        controls_layout.addWidget(self._invert_color_checkbox)
        
        controls_layout.addSpacing(15)
        
        self._scale_label = QLabel("表示倍率: 25%")
        controls_layout.addWidget(self._scale_label)
        self._scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._scale_slider.setRange(5, 200); self._scale_slider.setValue(25)
        self._scale_slider.valueChanged.connect(self._on_scale_changed)
        controls_layout.addWidget(self._scale_slider)
        
        controls_layout.addSpacing(15)

        self._char_spacing_label = QLabel("文字間隔: 0")
        controls_layout.addWidget(self._char_spacing_label)
        self._char_spacing_slider = QSlider(Qt.Orientation.Horizontal)
        self._char_spacing_slider.setRange(0, 500); self._char_spacing_slider.setValue(0)
        self._char_spacing_slider.valueChanged.connect(self._on_char_spacing_changed)
        controls_layout.addWidget(self._char_spacing_slider)

        self._line_spacing_label = QLabel("行間: 100%")
        controls_layout.addWidget(self._line_spacing_label)
        self._line_spacing_slider = QSlider(Qt.Orientation.Horizontal)
        self._line_spacing_slider.setRange(100, 300); self._line_spacing_slider.setValue(100)
        self._line_spacing_slider.valueChanged.connect(self._on_line_spacing_changed)
        controls_layout.addWidget(self._line_spacing_slider)
        
        controls_layout.addStretch(1)
        main_layout.addWidget(controls_widget)

        self._load_settings()

        self._on_wrap_mode_changed(self._wrap_checkbox.isChecked())
        self._on_scale_changed(self._scale_slider.value())
        self._on_char_spacing_changed(self._char_spacing_slider.value())
        self._on_line_spacing_changed(self._line_spacing_slider.value())
        
        if not self._text_input.toPlainText():
            self._text_input.setText("the quick brown fox jumps over the lazy dog.\n\nTHE QUICK BROWN FOX JUMPS OVER THE LAZY DOG.\n \nあらさじと　うちかへすらし　をやまだの　なはしろみづに　ぬれてつくるあ　めもはるに　ゆきまもあをく　なりにけり　いまこそのべに　わかなつみてめ　つくばやま　さけるさくらの　にほひをぞ　いりてをらねど　よそながらみつ　ちぐさにも　ほころぶはなの　しげきかな　いづらあをやぎ　ぬひしいとすぢ")
        
        self._update_display()
        
        self._db_watcher_timer = QTimer(self)
        self._db_watcher_timer.timeout.connect(self._trigger_glyph_check)
        self._db_watcher_timer.start(2000)
        
    def _trigger_glyph_check(self):
        """2秒ごとに表示ウィジェットに更新チェックを依頼する"""
        if self.isVisible():
            self._display_widget.check_and_refresh_visible_glyphs()

    def text_input_key_press(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and \
           event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._update_display(); event.accept(); return
        QTextEdit.keyPressEvent(self._text_input, event)

    def _update_display(self):
        self._display_widget.set_text(self._text_input.toPlainText())

    def _on_scale_changed(self, value: int):
        self._scale_label.setText(f"表示倍率: {value}%")
        self._display_widget.set_scale(value / 100.0)
        
    def _on_wrap_mode_changed(self, checked: bool):
        self._scroll_area.setWidgetResizable(checked)
        self._display_widget.set_auto_wrap(checked)

    def _on_invert_color_changed(self, checked: bool):
        self._display_widget.set_color_inversion(checked)

    def _on_char_spacing_changed(self, value: int):
        self._char_spacing_label.setText(f"文字間隔: {value}")
        self._display_widget.set_character_spacing(value)

    def _on_line_spacing_changed(self, value: int):
        self._line_spacing_label.setText(f"行間: {value}%")
        self._display_widget.set_line_spacing(value / 100.0)

    def _on_glyph_clicked(self, char: str, is_pua: bool):
        """グリフがクリックされたときにP-Glyphでそのグリフを選択する"""
        print(f"API経由でP-Glyphにグリフ '{char}' (PUA: {is_pua}) の選択をリクエスト中...")
        # api_usage_sample.py からインポートした select_glyph 関数を呼び出す
        # 現在のMetricビューアは縦書きの情報を持たないため、is_vrt2はFalseとする
        select_glyph(char, is_vrt2=False, is_pua=is_pua)
        # 成功/失敗のメッセージをQMessageBoxなどで表示することも可能です
        # 例: QMessageBox.information(self, "グリフ選択", f"P-Glyphに'{char}'の選択をリクエストしました。")


    def _save_settings(self):
        settings = {
            "window_geometry": self.saveGeometry().toHex().data().decode('utf-8'),
            "text": self._text_input.toPlainText(),
            "show_guide": self._guide_checkbox.isChecked(),
            "auto_wrap": self._wrap_checkbox.isChecked(),
            "invert_colors": self._invert_color_checkbox.isChecked(),
            "scale": self._scale_slider.value(),
            "char_spacing": self._char_spacing_slider.value(),
            "line_spacing": self._line_spacing_slider.value(),
        }
        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
        except IOError as e:
            print(f"設定の保存中にエラーが発生しました: {e}")

    def _load_settings(self):
        try:
            if not os.path.exists(SETTINGS_FILE):
                return

            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)

            geom_hex = settings.get("window_geometry", "").encode('utf-8')
            self.restoreGeometry(QByteArray.fromHex(geom_hex))
            
            self._text_input.setText(settings.get("text", ""))
            self._guide_checkbox.setChecked(settings.get("show_guide", True))
            self._wrap_checkbox.setChecked(settings.get("auto_wrap", True))
            self._invert_color_checkbox.setChecked(settings.get("invert_colors", False))
            self._scale_slider.setValue(settings.get("scale", 25))
            self._char_spacing_slider.setValue(settings.get("char_spacing", 0))
            self._line_spacing_slider.setValue(settings.get("line_spacing", 100))

        except (IOError, json.JSONDecodeError) as e:
            print(f"設定の読み込み中にエラーが発生しました: {e}")

    def closeEvent(self, event):
        self._db_watcher_timer.stop()
        self._save_settings()
        super().closeEvent(event)

def main():
    parser = argparse.ArgumentParser(description="P-Glyph フォントメトリックビューア")
    parser.add_argument("--db_path", type=str, required=True, help="プロジェクトデータベースファイル（.fontproj）のパス")
    args = parser.parse_args()
    app = QApplication(sys.argv)
    if not os.path.isfile(args.db_path):
        QMessageBox.critical(None, "起動エラー", f"指定されたデータベースファイルが見つかりません:\n{args.db_path}")
        sys.exit(1)
    window = MetricsViewerWindow(db_path=args.db_path)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
