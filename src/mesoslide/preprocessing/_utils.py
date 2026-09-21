"""Shared helpers for the preprocessing subpackage."""

from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:
    import pandas as pd


def channel_indices_from_markers(
    markers: List[str],
    marker_table: Optional["pd.DataFrame"] = None,
    marker_col: str = "marker_name",
) -> List[int]:
    """Resolve marker/channel names to integer channel indices via a marker
    table, or pass integer channels straight through.

    Parameters
    ----------
    markers : list of str
        Marker/channel names, or integer channel indices if `marker_table`
        is omitted.
    marker_table : pandas.DataFrame, optional
        Maps marker names to channel indices via `marker_table.index`, e.g.
        loaded from a markers CSV with ``pd.read_csv(...)``.
    marker_col : str, default='marker_name'
        Column in `marker_table` holding the marker name.

    Returns
    -------
    list of int

    Raises
    ------
    ValueError
        If a requested marker is not found in `marker_table[marker_col]`.
    """
    if marker_table is None:
        return [int(m) for m in markers]
    marker_ids = dict(zip(marker_table[marker_col], marker_table.index))
    missing = [m for m in markers if m not in marker_ids]
    if missing:
        raise ValueError(
            f"Markers not found in marker_table['{marker_col}']: {missing}"
        )
    return [int(marker_ids[m]) for m in markers]
