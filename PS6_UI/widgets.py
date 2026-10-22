"""Small custom widgets: thumbnails, hover preview, elided button."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QGridLayout, QLabel, QPushButton, QSizePolicy, QToolButton,
    QVBoxLayout, QWidget,
)

THUMB_SIZE = 48


# ---------------------------------------------------------------------------
# Thumbnail widget with an overlaid remove button (mirrors the Flet thumbs)
# ---------------------------------------------------------------------------
class _ImagePreview(QWidget):

    MAX_SIZE = 600

    def __init__(self, path: Path, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_ShowWithoutActivating, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_StyledBackground, True)   # paint stylesheet bg
        self.setObjectName("imagePreview")
        self.setStyleSheet(
            "QWidget#imagePreview { background: #16181a;"
            " border: 1px solid #6FA8FF; border-radius: 6px; }")

        label = QLabel(self)
        label.setAlignment(Qt.AlignCenter)
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            label.setText("?")
        else:
            label.setPixmap(pixmap.scaled(
                self.MAX_SIZE, self.MAX_SIZE,
                Qt.KeepAspectRatio, Qt.SmoothTransformation))

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addWidget(label)
        self.adjustSize()

    def show_near(self, anchor: QWidget) -> None:
        """Place the preview beside the anchor, clamped to the screen."""
        self.adjustSize()
        screen = anchor.screen()
        if screen is None:
            self.show()
            return
        geo = screen.availableGeometry()
        top_left = anchor.mapToGlobal(QPoint(0, 0))
        x = top_left.x() + anchor.width() + 10
        if x + self.width() > geo.right() + 1:
            x = top_left.x() - self.width() - 10
        y = top_left.y() + (anchor.height() - self.height()) // 2
        x = max(geo.left() + 4, min(x, geo.right() - self.width() - 3))
        y = max(geo.top() + 4, min(y, geo.bottom() - self.height() - 3))
        self.move(x, y)
        self.show()


class _Thumb(QWidget):

    remove = Signal(object)                      # carries the Path
    selected = Signal(object)                    # carries the Path

    def __init__(self, path: Path, size: int = THUMB_SIZE,
                 border: str | None = None, removable: bool = True,
                 selectable: bool = False, parent=None):
        super().__init__(parent)
        self.path = path
        self.setFixedSize(size, size)
        self._selectable = selectable
        self._border = border
        self._selected = False
        self._preview: _ImagePreview | None = None
        self._hover_timer = QTimer(self)
        self._hover_timer.setSingleShot(True)
        self._hover_timer.setInterval(250)   # brief dwell avoids popup flashes
        self._hover_timer.timeout.connect(self._show_preview)

        pixmap = QPixmap(str(path))
        self._label = QLabel(self)
        self._label.setFixedSize(size, size)
        self._label.setAlignment(Qt.AlignCenter)
        # Let clicks fall through to this widget so the whole tile selects.
        self._label.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        if not pixmap.isNull():
            self._label.setPixmap(pixmap.scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:                                     # unreadable image
            self._label.setText("?")

        lay = QGridLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self._label, 0, 0)

        self._badge = QLabel("REF", self)
        self._badge.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self._badge.setStyleSheet(
            "QLabel { background: #4A6FA5; color: white; font-size: 8px;"
            "  font-weight: bold; padding: 1px 3px; border-radius: 3px; }")
        self._badge.adjustSize()
        lay.addWidget(self._badge, 0, 0, Qt.AlignBottom | Qt.AlignHCenter)
        self._badge.hide()
        self._label.installEventFilter(self)
        self._badge.installEventFilter(self)

        if removable:
            btn = QToolButton(self)
            btn.setText("x")
            btn.setFixedSize(16, 16)
            btn.setStyleSheet(
                "QToolButton { background: rgba(0,0,0,60%); color: #e0e0e0;"
                "  border: none; font-size: 9px; }"
                "QToolButton:hover { background: rgba(180,50,50,90%); }")
            btn.clicked.connect(lambda: self.remove.emit(self.path))
            lay.addWidget(btn, 0, 0, Qt.AlignTop | Qt.AlignRight)
        self._base_tip = "Click to select as REF"
        if selectable:
            self.setCursor(Qt.PointingHandCursor)
            self.setToolTip(self._base_tip)
        self._apply_style()

    def _apply_style(self):
        border = "#6FA8FF" if self._selected else self._border
        style = ""
        if border:
            width = 3 if self._selected else 2
            style = f"border: {width}px solid {border}; border-radius: 4px;"
        self._label.setStyleSheet(f"background: #2a2d30; {style}")
        self._badge.setVisible(self._selected)

    def set_selected(self, selected: bool):
        self._selected = bool(selected)
        self._apply_style()
        # While selected, suppress the native tooltip so it cannot pop up on
        # top of the large preview; the hover preview takes over.
        self.setToolTip("" if self._selected else self._base_tip)
        if not self._selected:
            self._hide_preview()
        elif self.underMouse():
            # Selected via click without moving the cursor: behave as if the
            # cursor had just entered the tile.
            self._hover_timer.start()

    # -- hover preview for the selected REF image -------------------------
    def _show_preview(self):
        if not (self._selectable and self._selected and self.isVisible()):
            return
        if self._preview is None:
            self._preview = _ImagePreview(self.path, self)
        self._preview.show_near(self)

    def _hide_preview(self):
        if self._preview is not None and self._preview.isVisible():
            self._preview.hide()

    def enterEvent(self, event):
        super().enterEvent(event)
        if self._selectable and self._selected:
            self._hover_timer.start()

    def leaveEvent(self, event):
        self._hover_timer.stop()
        self._hide_preview()
        super().leaveEvent(event)

    def hideEvent(self, event):
        self._hover_timer.stop()
        self._hide_preview()
        super().hideEvent(event)

    def wheelEvent(self, event):
        # Scrolling moves the tile out from under the preview; drop it and
        # let the list scroll normally.
        self._hover_timer.stop()
        self._hide_preview()
        event.ignore()

    def eventFilter(self, obj, event):
        # The image label covers this widget, so handle its clicks too.
        if (self._selectable and obj in (self._label, self._badge)
                and event.type() == QEvent.MouseButtonPress
                and event.button() == Qt.LeftButton):
            self.selected.emit(self.path)
            event.accept()
            return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event):
        if self._selectable and event.button() == Qt.LeftButton:
            self.selected.emit(self.path)
            event.accept()
            return
        super().mousePressEvent(event)


class _ElidedButton(QPushButton):

    #: emitted on middle-click (used for the ghost-mesh overlay toggle)
    middleClicked = Signal()

    def __init__(self, full_text: str, parent=None):
        super().__init__(parent)
        self._full_text = full_text
        self._dot = False
        # Let the layout shrink below the text width; the label is re-elided
        # in resizeEvent instead of forcing the Outliner column wider.
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.setMinimumWidth(0)
        self._update_elision()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_elision()

    def set_dot(self, enabled: bool):
        """Show/hide the small right-aligned status circle."""
        enabled = bool(enabled)
        if enabled != self._dot:
            self._dot = enabled
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self._dot:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        radius = 3.5
        center = QPointF(self.width() - 10.0, self.height() / 2.0)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0x9a, 0xa0, 0xa6))   # same gray as children
        painter.drawEllipse(center, radius, radius)

    def mousePressEvent(self, event):
        if event.button() == Qt.MiddleButton:
            self.middleClicked.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def _update_elision(self):
        # 2x6 px stylesheet padding plus a little end margin.
        available = max(16, self.width() - 20)
        text = self.fontMetrics().elidedText(
            self._full_text, Qt.ElideMiddle, available)
        if text != self.text():
            self.setText(text)
