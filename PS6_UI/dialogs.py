"""Modal dialogs and the persistent settings store.

Split out of main.py so the window class stays focused on layout and
wiring; everything user-facing that pops up as its own dialog lives here:
  - SplitNameDialog: per-polygroup file suffix prompt (Split and save)
  - SettingsDialog: the application settings (Settings menu / Ctrl+,)
  - app_settings()/stage2_rescale_enabled(): the QSettings store those
    dialogs read and write.
"""

from __future__ import annotations

import re

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout,
)

#: Characters that cannot appear in a file name on Windows.
_BAD_SUFFIX_CHARS = re.compile(r'[<>:"/\|?*\x00-\x1f]')


def split_suffix(text: str) -> str:
    """Turn user input into a safe file-name suffix (without separators)."""
    return _BAD_SUFFIX_CHARS.sub("_", text).strip().strip(".")


class SplitNameDialog(QDialog):
    """Ask for one polygroup's file suffix during Split-and-save."""

    def __init__(self, group, index, total, default_suffix, stem,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Set name")
        self.setModal(True)
        self.setMinimumWidth(340)
        layout = QVBoxLayout(self)
        title = QLabel(f"Set name for group {group}  ({index} of {total})")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)
        hint = QLabel("Suffix added to the file name after the underscore:")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        self.edit = QLineEdit(default_suffix)
        self.edit.selectAll()
        layout.addWidget(self.edit)
        self.preview = QLabel()
        self.preview.setStyleSheet("color: #888c92; font-size: 10px;")
        self.preview.setWordWrap(True)
        layout.addWidget(self.preview)
        buttons = QHBoxLayout()
        self.btn_cancel = QPushButton("Cancel split")
        self.btn_set = QPushButton("Set")
        self.btn_set.setDefault(True)
        buttons.addWidget(self.btn_cancel)
        buttons.addStretch(1)
        buttons.addWidget(self.btn_set)
        layout.addLayout(buttons)
        self._stem = stem
        self.edit.textChanged.connect(self._refresh_preview)
        self.btn_set.clicked.connect(self.accept)
        self.btn_cancel.clicked.connect(self.reject)
        self._refresh_preview()

    def _refresh_preview(self):
        suffix = self.suffix()
        self.btn_set.setEnabled(bool(suffix))
        self.preview.setText(
            f"Will save {self._stem}_{suffix}.glb" if suffix
            else "Enter a suffix to continue.")

    def suffix(self) -> str:
        return split_suffix(self.edit.text())


#: QSettings organisation/application so the Settings dialog persists between
#: sessions (Windows registry / ini depending on the platform).
_SETTINGS_ORG = "UniForge3D"
_SETTINGS_APP = "UniForge3D-UI"


def app_settings() -> QSettings:
    """The application-wide QSettings store."""
    return QSettings(_SETTINGS_ORG, _SETTINGS_APP)


def stage2_rescale_enabled() -> bool:
    """Setting 'Uniforge 2nd stage Rescale' (on by default)."""
    return app_settings().value("settings/stage2_rescale", True, type=bool)


class SettingsDialog(QDialog):
    """Application settings, opened from the Settings menu."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        box = QGroupBox("Generation")
        box_layout = QVBoxLayout(box)
        self.stage2_rescale = QCheckBox("Uniforge 2nd stage Rescale")
        self.stage2_rescale.setChecked(stage2_rescale_enabled())
        self.stage2_rescale.setToolTip(
            "Auto-rescale after 2nd stage")
        box_layout.addWidget(self.stage2_rescale)
        hint = QLabel(
            "Turn this off to keep the raw canonical-cube output of the 2nd "
            "stage (the old Scale back button behaviour is not available "
            "then).")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #888c92; font-size: 10px;")
        box_layout.addWidget(hint)
        layout.addWidget(box)

        buttons = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_ok = QPushButton("OK")
        btn_ok.setDefault(True)
        buttons.addWidget(btn_cancel)
        buttons.addStretch(1)
        buttons.addWidget(btn_ok)
        layout.addLayout(buttons)
        btn_ok.clicked.connect(self.accept)
        btn_cancel.clicked.connect(self.reject)
