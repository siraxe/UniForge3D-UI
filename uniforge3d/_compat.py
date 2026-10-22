"""Compatibility shims that must run before optional third-party imports."""

import enum

import torch.utils._pytree as pytree

_original_register_constant = pytree.register_constant


def _register_constant(cls):
    """Skip pytree registration for Enums, which torch now handles natively.

    torchao registers its Enum configs as pytree constants when imported.
    Recent torch versions support Enums as opaque value types, so those calls
    are unnecessary and log a deprecation warning. Dropping them here avoids
    patching the installed torchao package.
    """
    if isinstance(cls, type) and issubclass(cls, enum.Enum):
        return
    return _original_register_constant(cls)


if not getattr(_original_register_constant, "_uniforge_enum_compat", False):
    _register_constant._uniforge_enum_compat = True
    pytree.register_constant = _register_constant
