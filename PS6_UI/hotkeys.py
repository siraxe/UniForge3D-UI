"""Single source of truth for every keyboard shortcut in UniForge3D.

Each hotkey is declared once in ``HOTKEYS``; the menu bar (main.py) and the
3D viewer (viewer.py) both read their sequences from here, so adding,
removing or rebinding a shortcut only ever happens in this one file.
Use ``hotkeys.seq("name")`` wherever a QKeySequence/QAction shortcut is set.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QKeySequence


@dataclass(frozen=True)
class Hotkey:
    sequence: str          # Qt-compatible string, e.g. "Ctrl+O"
    description: str       # one-line explanation for menus and tooltips


#: All application shortcuts. The key is the stable action id used with
#: :func:`seq`; never hard-code a key string anywhere else.
HOTKEYS: dict[str, Hotkey] = {
    # File / app (menu bar in main.py)
    "open_folder": Hotkey(
        "Ctrl+O", "Pick a folder; every *.glb (and any *.obj without a "
        "GLB twin) appears in the Outliner column."),
    "settings": Hotkey(
        "Ctrl+,", "Open the Settings dialog."),
    "exit": Hotkey(
        "Ctrl+Q", "Quit UniForge3D."),
    # 3D viewer (viewer.py)
    "reset_view": Hotkey(
        "F", "Frame the loaded model again (works anywhere in the app)."),
    "clear_selection": Hotkey(
        "Ctrl+D", "Clear the lasso face selection."),
    "assign_polygroup": Hotkey(
        "Ctrl+W", "Paint the selected faces with the next free polygroup."),
    "assign_polygroup_islands": Hotkey(
        "Ctrl+Shift+W", "Split the visible faces into edge-connected "
        "islands and give every island its own polygroup."),
    # Draw mode (Outliner panel's square Draw button; toggled from viewer.py)
    "draw_mode": Hotkey(
        "Q", "Toggle Draw mode (the Outliner panel's Draw button). While "
        "on, hold Shift over the mesh to sculpt with the smooth brush."),
    # Brush symmetry (Edit mesh panel's Symmetry switch; toggled from
    # viewer.py - it flips the checkbox, the brushes read it per stroke)
    "symmetry_toggle": Hotkey(
        "X", "Toggle brush symmetry (the Edit mesh panel's Symmetry "
        "switch). While on, move / smooth strokes also apply at the "
        "position mirrored across the checked axis planes."),
    # Ghost overlay (menu bar in main.py; overlay drawn by viewer.py)
    "ghost_visible": Hotkey(
        "Ctrl+1", "Hide/show the ghosted reference mesh. The ghost stays "
        "picked, so Reproject keeps using it as the donor mesh."),
    # Smooth brush (Shift-hold while hovering the mesh; viewer_brush.py
    # handles the live key events - Shift is a modifier, not a QAction)
    # No entry needed here: the brush has no QAction-style shortcut.
    # Shift shows the ring, Shift + left-drag smooths, W / S resize.
    "undo": Hotkey(
        "Ctrl+Z", "Undo the last viewer edit: smooth-brush stroke (restores "
        "the exact vertex positions it moved), lasso selection or "
        "polygroup paint/reset."),
    "redo": Hotkey(
        "Ctrl+Shift+Z", "Re-apply the most recently undone viewer edit "
        "(brush stroke, lasso selection or polygroup paint/reset)."),
}


def seq(name: str) -> str:
    """The shortcut sequence for action id *name* (e.g. ``hotkeys.seq("exit")``)."""
    try:
        return HOTKEYS[name].sequence
    except KeyError:
        raise KeyError(
            f"unknown hotkey {name!r}; declare it in hotkeys.HOTKEYS") from None


def describe(name: str) -> str:
    """The human-readable description for action id *name*."""
    try:
        return HOTKEYS[name].description
    except KeyError:
        raise KeyError(
            f"unknown hotkey {name!r}; declare it in hotkeys.HOTKEYS") from None


def key_combo(name: str):
    """QKeyCombination for a single-key hotkey, or None for combos.

    Hold-style shortcuts (plain letters such as S) cannot be QActions -
    they need live key press/release events on the widget. The viewer
    matches those events with this helper so the binding itself still
    only ever lives in ``HOTKEYS``.
    """
    if name not in HOTKEYS:
        raise KeyError(
            f"unknown hotkey {name!r}; declare it in hotkeys.HOTKEYS")
    parsed = QKeySequence(HOTKEYS[name].sequence)
    return parsed[0] if parsed.count() == 1 else None
