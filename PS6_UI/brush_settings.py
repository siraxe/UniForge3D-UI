"""Brush-settings store + frameless hold-Space popup (Draw mode).

Owns the handful of brush values worth exposing (the rest stay as module
constants in viewer_brush):

  Brush : radius (screen px), dab spacing (fraction of the radius)
  Smooth: umbrella strength per dab (0..3; above 1 = extra smoothing
          passes, so the slider goes stronger without overshooting)
  Flatten: flatten multiplier, flatten plane depth (fraction of the radius)
  Flag  : move brush applies only to the clicked face's polygroup
          (off by default)

Values changed in the popup apply live and are remembered for the session;
Save persists them to brush_settings.json next to the app so later
sessions start from them, and Reset restores the in-file defaults and
removes the saved file (reset supersedes save).
"""

from __future__ import annotations

import json

from PySide6.QtCore import QEvent, QObject, QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from paths import ROOT

_PATH = ROOT / "brush_settings.json"

#: key -> default; viewer_brush keeps only the non-exposed constants.
DEFAULTS = {
    "radius_px": 40.0,      # brush ring radius on screen
    "strength": 0.75,       # umbrella smooth, per dab
    "spacing": 0.3,         # dab spacing, fraction of the radius
    "hook_sharpness": 0.9,  # move-brush spike focus (tan angle, radians)
    "flatten": 0.5,         # flatten pass multiplier (0 = off)
    "flatten_depth": 0.03,  # flatten plane offset inward, fraction of R
    #: Move brush ("snake hook") deforms only the polygroup of the face
    #: that was clicked first; off = whole grabbed region, as before.
    "apply_to_polygroup": False,
}

#: key -> (min, max, decimals) for the popup sliders.
RANGES = {
    "radius_px": (4.0, 256.0, 0),
    "strength": (0.0, 3.0, 2),
    "spacing": (0.05, 1.0, 2),
    "hook_sharpness": (0.3, 1.5, 2),
    "flatten": (0.0, 1.0, 2),
    "flatten_depth": (0.0, 0.15, 3),
}

_GROUPS = [
    ("Brush", ["radius_px", "spacing"]),
    ("Smooth", ["strength", "hook_sharpness"]),
    ("Flatten", ["flatten", "flatten_depth"]),
]

#: Popup top-left relative to the cursor when it opens (logical px):
#: up and to the left so the pointer lands on the middle of the radius
#: slider instead of on the popup's top-left corner.
_OPEN_OFFSET = QPoint(-65, -45)

_STYLE = """
QWidget#brushPopup { background: #1E2022; border: 1px solid #3a3d42;
                     border-radius: 8px; }
QGroupBox { border: 1px solid #3a3d42; border-radius: 6px; margin-top: 9px;
            padding: 4px 6px 2px 6px; font-size: 10px; color: #9aa0a6; }
QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
QLabel { color: #c8ccd0; font-size: 10px; background: transparent; }
QPushButton { background: #2A2D30; border: 1px solid #3a3d42;
              border-radius: 4px; padding: 4px 16px; color: #e0e0e0;
              font-size: 11px; }
QPushButton:hover { background: #35393e; }
QCheckBox { color: #c8ccd0; font-size: 10px; background: transparent;
            spacing: 6px; }
QCheckBox::indicator { width: 12px; height: 12px; border: 1px solid #3a3d42;
                       border-radius: 3px; background: #2A2D30; }
QCheckBox::indicator:checked { background: #4A6FA5;
                               border-color: #6FA8FF; }
QSlider::groove:horizontal { height: 4px; background: #2A2D30;
                             border-radius: 2px; }
QSlider::sub-page:horizontal { background: #4A6FA5; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0;
                             background: #6FA8FF; border-radius: 6px; }
"""


def _clamp(key, value) -> float:
    lo, hi, _ = RANGES[key]
    return min(max(float(value), lo), hi)


def _coerce(key, value):
    """Slider keys clamp to their range; flag keys become booleans."""
    if key not in RANGES:
        return bool(value)
    return _clamp(key, value)


class BrushSettings(QObject):
    """The live brush values; emits :attr:`changed` on every edit."""

    changed = Signal()

    def __init__(self):
        super().__init__()
        self._values = dict(DEFAULTS)

    def __getattr__(self, key):
        # only called when normal attribute lookup fails -> the value keys
        try:
            return self.__dict__["_values"][key]
        except KeyError:
            raise AttributeError(key) from None

    # -- persistence ----------------------------------------------------------
    @classmethod
    def load(cls) -> "BrushSettings":
        """Defaults merged with whatever a previous session saved."""
        settings = cls()
        try:
            saved = json.loads(_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return settings
        for key in DEFAULTS:
            if key in saved:
                try:
                    settings._values[key] = _coerce(key, saved[key])
                except (TypeError, ValueError):
                    continue
        return settings

    def save(self):
        """Persist current values for later sessions."""
        try:
            _PATH.write_text(
                json.dumps(self._values, indent=2), encoding="utf-8")
        except OSError:
            pass

    def reset(self):
        """Back to the in-file defaults; a saved file is removed
        (reset supersedes save)."""
        self._values = dict(DEFAULTS)
        try:
            _PATH.unlink()
        except OSError:
            pass
        self.changed.emit()

    # -- mutation ---------------------------------------------------------------
    def set(self, key, value, emit: bool = True):
        """Update one value (slider keys clamp, flag keys coerce to bool)."""
        coerced = _coerce(key, value)
        if self._values.get(key) == coerced:
            return
        self._values[key] = coerced
        if emit:
            self.changed.emit()


class _SliderRow(QWidget):
    """One labelled slider + live value readout bound to a settings key."""

    def __init__(self, title, key, settings: BrushSettings):
        super().__init__()
        self._key = key
        self._settings = settings
        self._lo, self._hi, self._decimals = RANGES[key]

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(2)
        head = QHBoxLayout()
        head.setSpacing(4)
        name = QLabel(title)
        self.value_label = QLabel()
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.value_label.setStyleSheet("color: #6FA8FF;")
        self.value_label.setMinimumWidth(34)
        head.addWidget(name)
        head.addStretch(1)
        head.addWidget(self.value_label)
        lay.addLayout(head)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(1000)
        self.slider.setCursor(Qt.PointingHandCursor)
        lay.addWidget(self.slider)

        self.slider.valueChanged.connect(self._on_slider)
        settings.changed.connect(self.refresh)
        self.refresh()

    def _to_slider(self, value) -> int:
        frac = (_clamp(self._key, value) - self._lo) / (self._hi - self._lo)
        return int(round(frac * 1000))

    def refresh(self):
        """Sync slider + label from the settings (external changes too)."""
        value = float(getattr(self._settings, self._key))
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_slider(value))
        self.slider.blockSignals(False)
        self.value_label.setText(f"{value:.{self._decimals}f}")

    def _on_slider(self, tick):
        value = self._lo + (self._hi - self._lo) * (tick / 1000.0)
        value = round(value, self._decimals)
        self._settings.set(self._key, value)
        self.value_label.setText(f"{value:.{self._decimals}f}")


class BrushSettingsPopup(QWidget):
    """Frameless 2-column slider popup; closes on outside click or Space."""

    #: Emitted whenever the popup hides, however it was closed (click
    #: outside, Space release, model swap), so the owner can drop it.
    closed = Signal()

    def __init__(self, settings: BrushSettings, parent=None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.setObjectName("brushPopup")
        self.setStyleSheet(_STYLE)
        self._settings = settings

        grid = QGridLayout(self)
        grid.setContentsMargins(10, 10, 10, 8)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)

        def group(title, keys, row, col, span=1):
            box = QGroupBox(title)
            lay = QVBoxLayout(box)
            lay.setContentsMargins(6, 2, 6, 4)
            lay.setSpacing(5)
            for key in keys:
                lay.addWidget(_SliderRow(key, key, settings))
            grid.addWidget(box, row, col, 1, span)

        group("Brush", _GROUPS[0][1], 0, 0)
        group("Smooth", _GROUPS[1][1], 0, 1)
        group("Flatten", _GROUPS[2][1], 1, 0, span=2)

        self.chk_polygroup = QCheckBox("Apply to polygroup")
        self.chk_polygroup.setCursor(Qt.PointingHandCursor)
        self.chk_polygroup.setToolTip(
            "Move brush only: deform just the polygroup of the face the "
            "stroke starts on")
        self.chk_polygroup.setChecked(bool(settings.apply_to_polygroup))
        self.chk_polygroup.toggled.connect(self._on_polygroup_toggled)
        settings.changed.connect(self._refresh_polygroup)
        grid.addWidget(self.chk_polygroup, 2, 0, 1, 2)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        self.btn_reset = QPushButton("Reset")
        self.btn_save = QPushButton("Save")
        buttons.addWidget(self.btn_reset)
        buttons.addWidget(self.btn_save)
        grid.addLayout(buttons, 3, 0, 1, 2)

        self.btn_reset.clicked.connect(settings.reset)
        self.btn_save.clicked.connect(self._on_save)

    def _on_save(self):
        self._settings.save()
        # brief acknowledgement without stealing focus
        self.btn_save.setText("Saved")
        QTimer.singleShot(1200, lambda: self.btn_save.setText("Save"))

    def _on_polygroup_toggled(self, checked: bool):
        self._settings.set("apply_to_polygroup", bool(checked))

    def _refresh_polygroup(self):
        """Reset / loaded settings -> keep the checkbox in sync."""
        checked = bool(self._settings.apply_to_polygroup)
        if self.chk_polygroup.isChecked() != checked:
            self.chk_polygroup.blockSignals(True)
            self.chk_polygroup.setChecked(checked)
            self.chk_polygroup.blockSignals(False)

    # -- hold-Space behaviour -------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        # Watch the whole application while open: the popup has the
        # keyboard grab, and a focused slider/button would otherwise eat
        # the Space release that is supposed to close us.
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def hideEvent(self, event):
        app = QApplication.instance()
        if app is not None:
            app.removeEventFilter(self)
        super().hideEvent(event)
        self.closed.emit()

    def eventFilter(self, obj, event):
        if (event.type() == QEvent.KeyRelease
                and event.key() == Qt.Key_Space
                and not event.isAutoRepeat()):
            self.close()          # releasing Space is the popup's close key
            return True
        return super().eventFilter(obj, event)

    def open_at(self, global_pos=None):
        """Size, offset from the cursor, clamp to the screen, then show.

        The offset puts the pointer over the radius slider, so the user
        can immediately drag it instead of first moving onto the popup.
        """
        self.adjustSize()
        pos = QPoint(global_pos if global_pos is not None else QCursor.pos())
        pos += _OPEN_OFFSET
        screen = self.screen()
        if screen is None:
            screen = QGuiApplication.screenAt(pos)
        if screen is not None:
            geo = screen.availableGeometry()
            pos.setX(max(geo.left() + 4,
                         min(pos.x(), geo.right() + 1 - self.width() - 4)))
            pos.setY(max(geo.top() + 4,
                         min(pos.y(), geo.bottom() + 1 - self.height() - 4)))
        self.move(pos)
        self.show()
