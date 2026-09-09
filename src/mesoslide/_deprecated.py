"""Deprecation shims for the pre-single-WSI API.

Kept in one module so the whole lot can be deleted at 1.0. Each shim names the
replacement rather than just warning, because the migration is not a rename:
slides used to be picked out of a multi-WSI SpatialData by mangled element name
(``{image}_grid_point_patch``), and are now passed as objects or a manifest.

Three behaviours, matching the three kinds of change:

``rename``       the argument still exists under a new name.
``drop``         the argument is gone because element names are now constants;
                 accepted silently if it still holds its historical default,
                 rejected otherwise, since a non-default value cannot be honoured.
``removed``      the argument is gone and no automatic translation is possible.
"""

from __future__ import annotations

import functools
import warnings


class _Spec:
    __slots__ = ("kind", "target", "default", "hint")

    def __init__(self, kind, target=None, default=None, hint=None):
        self.kind = kind
        self.target = target
        self.default = default
        self.hint = hint


def rename(new_name: str) -> _Spec:
    """Old keyword forwards to `new_name`."""
    return _Spec("rename", target=new_name)


def drop(default, hint: str) -> _Spec:
    """Old keyword is gone; tolerated only at its historical default."""
    return _Spec("drop", default=default, hint=hint)


def removed(hint: str) -> _Spec:
    """Old keyword is gone with no automatic translation."""
    return _Spec("removed", hint=hint)


def deprecated_kwargs(**specs: _Spec):
    """Intercept removed keyword arguments, warning or raising as appropriate."""

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for old, spec in specs.items():
                if old not in kwargs:
                    continue
                value = kwargs.pop(old)

                if spec.kind == "rename":
                    warnings.warn(
                        f"{func.__name__}(): '{old}' is deprecated, use "
                        f"'{spec.target}' instead.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    kwargs.setdefault(spec.target, value)

                elif spec.kind == "drop":
                    if value != spec.default:
                        raise ValueError(
                            f"{func.__name__}(): '{old}={value!r}' can no longer be "
                            f"honoured. {spec.hint}"
                        )
                    warnings.warn(
                        f"{func.__name__}(): '{old}' is deprecated and ignored. "
                        f"{spec.hint}",
                        DeprecationWarning,
                        stacklevel=2,
                    )

                else:  # removed
                    raise TypeError(
                        f"{func.__name__}(): '{old}' has been removed. {spec.hint}"
                    )
            return func(*args, **kwargs)

        return wrapper

    return decorator


ELEMENT_NAME_HINT = (
    "Each SpatialData now holds one WSI, so element names are constants "
    "('wsi', 'tiles', 'tissues', 'tiles_table'); pass tile_key= if you renamed "
    "the tiles element."
)

SLIDES_HINT = (
    "Pass `slides` instead: a tile table, a WSIData, a sequence or mapping of "
    "either, or a cohort manifest DataFrame. See mesoslide.iter_slides / "
    "mesoslide.open_slides."
)


def check_not_spatialdata(slides, func_name: str) -> None:
    """Reject the old ``(sdata, patch_table_names, ...)`` positional call form.

    Passing a multi-WSI SpatialData positionally used to be how every selector
    was called. Left unchecked it would now silently bind to `slides` and then
    fail somewhere less obvious.
    """
    try:
        from spatialdata import SpatialData
        from wsidata import WSIData
    except ImportError:  # pragma: no cover
        return
    if isinstance(slides, SpatialData) and not isinstance(slides, WSIData):
        raise TypeError(
            f"{func_name}(): a bare SpatialData is no longer accepted. It used to "
            f"hold many WSIs, addressed by mangled element names; each store now "
            f"holds one slide. {SLIDES_HINT}"
        )
