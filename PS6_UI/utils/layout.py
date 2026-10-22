"""Small layout builders shared by the side panels."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QPushButton, QVBoxLayout, QWidget


def underlined_label(text: str) -> QLabel:
    label = QLabel(text)
    font = label.font()
    font.setUnderline(True)
    label.setFont(font)
    return label


def accordion_section(container: QVBoxLayout, text: str, *,
                      first: bool = False,
                      expanded: bool = False) -> QVBoxLayout:
    """Collapsible section under an underlined header; returns its body layout."""
    if not first:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #3a3d42; background: #3a3d42;")
        line.setFixedHeight(1)
        container.addSpacing(6)
        container.addWidget(line)
        container.addSpacing(2)
    body = QWidget()
    body_lay = QVBoxLayout(body)
    body_lay.setContentsMargins(0, 0, 0, 0)
    body_lay.setSpacing(6)
    header = QPushButton(text)
    header.setFlat(True)
    header.setCheckable(True)
    header.setCursor(Qt.PointingHandCursor)
    font = header.font()
    font.setUnderline(True)
    header.setFont(font)
    header.setStyleSheet(
        "QPushButton { background: transparent; border: none;"
        " text-align: left; color: #c8cbd0; padding: 1px 0; }"
        "QPushButton:hover { color: #ffffff; }")

    def _toggle(on: bool):
        header.setText(("▾  " if on else "▸  ") + text)
        body.setVisible(on)
    header.toggled.connect(_toggle)
    container.addWidget(header)
    container.addWidget(body)
    header.blockSignals(True)
    header.setChecked(expanded)
    header.blockSignals(False)
    _toggle(expanded)
    return body_lay
