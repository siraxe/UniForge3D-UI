"""Dark palette and shared splitter styling."""

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QToolTip

DARK_PALETTE = QPalette()
DARK_PALETTE.setColor(QPalette.Window, QColor(0x1E, 0x20, 0x22))
DARK_PALETTE.setColor(QPalette.WindowText, QColor(0xE0, 0xE0, 0xE0))
DARK_PALETTE.setColor(QPalette.Base, QColor(0x16, 0x18, 0x1A))
DARK_PALETTE.setColor(QPalette.Text, QColor(0xE0, 0xE0, 0xE0))
DARK_PALETTE.setColor(QPalette.Button, QColor(0x2A, 0x2D, 0x30))
DARK_PALETTE.setColor(QPalette.ButtonText, QColor(0xE0, 0xE0, 0xE0))
DARK_PALETTE.setColor(QPalette.Highlight, QColor(0x4A, 0x6F, 0xA5))
DARK_PALETTE.setColor(QPalette.HighlightedText, QColor(0xFF, 0xFF, 0xFF))
DARK_PALETTE.setColor(QPalette.PlaceholderText, QColor(0x80, 0x84, 0x88))

SPLITTER_STYLE = "QSplitter::handle { background: #3a3d42; }"


def apply_tooltip_font() -> None:
    """Give the native tooltip an explicit point size.

    Widgets sized via stylesheet ``font-size`` in *px* resolve to a font
    with ``pointSize == -1``; when their tooltip shows, Qt's QTipLabel
    copies that font and warns "QFont::setPointSize: Point size <= 0 (-1)".
    Setting the tooltip font once avoids the warning (and keeps tooltips
    readable against the px-sized control text).
    """
    font = QFont()
    font.setPointSize(9)
    QToolTip.setFont(font)
