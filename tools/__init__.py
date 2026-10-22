"""Editable, hot-reloadable backend tools (reproject, polish, remesh).

Each subpackage is self-contained and exposes its public API through its
``__init__``.  The UI loads them via ``tools._loader.load_tool``, which
re-reads the sources from disk on every job, so the tools can be edited in
place while the app is running.
"""
