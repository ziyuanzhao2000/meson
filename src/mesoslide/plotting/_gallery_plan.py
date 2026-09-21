"""Composable multi-row patch-gallery layout.

A `GalleryPlan` describes which rows go on a patch-gallery figure -- built up
incrementally by calling one `add_*_row` method per row type (H&E, cluster
maps, CyCIF channels, ...) -- decoupled from how the result is paginated,
which `render()` decides. Rows render top-to-bottom in the order they were
added.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
import io
import numpy as np
import matplotlib.pyplot as plt

from mesoslide._slides import DEFAULT_TILE_KEY, SLIDE_ID
from mesoslide.preprocessing._extract_patches import extract_patch_images
from mesoslide.preprocessing._extract_cluster_maps import extract_cluster_maps
from mesoslide.preprocessing._utils import channel_indices_from_markers
from ._image_grid import _draw_group_border, _draw_corner_label, _group_color_lookup
from ._utils import _finish_plot, FLUOROPHORE_COLORS, MARKER_COLOR_DEFAULTS

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd
    from mesoslide.tools.segmenters import TokenClusterizer


def _paged_path(output_path: str, start_idx: int, end_idx: int) -> str:
    """Derive a per-page file path by inserting the patch range before the extension."""
    p = Path(output_path)
    return str(p.with_name(f"{p.stem}_{start_idx + 1}-{end_idx}{p.suffix}"))


def _to_float01_rgb(img: np.ndarray) -> np.ndarray:
    """Coerce an (H, W, 3[/4]) image to float RGB in [0, 1]."""
    rgb = img[..., :3]
    if np.issubdtype(rgb.dtype, np.integer):
        return rgb.astype(np.float32) / 255.0
    return rgb.astype(np.float32)


def _resolve_color(spec: Union[str, Tuple[float, float, float]]) -> Tuple[float, float, float]:
    if isinstance(spec, str):
        return FLUOROPHORE_COLORS[spec]
    return tuple(spec)


def _resolve_per_channel(value, channels: List[str]) -> Dict[str, Optional[float]]:
    """Normalise a scalar / dict / list-aligned-with-channels value to {channel: value}."""
    if value is None:
        return {ch: None for ch in channels}
    if isinstance(value, dict):
        return {ch: value.get(ch) for ch in channels}
    if isinstance(value, (list, tuple)):
        return dict(zip(channels, value))
    return {ch: value for ch in channels}


def _channel_percentile(cycif_arr, channel_idx: int, pct: float) -> float:
    if isinstance(cycif_arr, np.ndarray):
        return float(np.percentile(cycif_arr[:, channel_idx], pct))
    return float(np.percentile(
        np.concatenate([p[channel_idx].ravel() for p in cycif_arr]), pct
    ))


@dataclass
class _RowBlock:
    """One or more rows repeated once per patch."""
    label_per_row: List[str]
    n_rows: int
    frames: List[List[np.ndarray]]  # frames[patch_idx][row_idx] -> ready-to-imshow image
    border_group_ids: Optional[List] = None  # drawn on row 0 of the block only
    border_cmap: str = 'tab10'
    border_extend: float = 0.1
    border_alpha: float = 1.0


def _normalize_axes(axes, n_rows: int, n_cols: int) -> np.ndarray:
    if n_rows == 1 and n_cols == 1:
        return np.array([[axes]])
    if n_rows == 1:
        return np.asarray(axes).reshape(1, -1)
    if n_cols == 1:
        return np.asarray(axes).reshape(-1, 1)
    return np.asarray(axes)


class GalleryPlan:
    """Build a multi-row patch-gallery figure layout, then render it paginated.

    Examples
    --------
    >>> plan = GalleryPlan(patches)
    >>> plan.add_he_row(slides=slides)
    >>> plan.add_cluster_map_rows([clusterizer], model='uni2', slides=slides)
    >>> plan.render(output_path='gallery.png', patches_per_row=10, max_rows_per_page=6)
    """

    def __init__(self, patches: "ad.AnnData"):
        self.patches = patches
        self.n_patches = len(patches.obs)
        self._blocks: List[_RowBlock] = []

    def add_he_row(
        self,
        slides=None,
        *,
        tile_key: str = DEFAULT_TILE_KEY,
        label: str = "H&E",
        group_col: Optional[str] = None,
        cmap: str = 'tab10',
        border_extend: float = 0.1,
        border_alpha: float = 1.0,
        cache: bool = False,
    ) -> "GalleryPlan":
        """Add one row per patch of H&E images."""
        if group_col is not None and group_col not in self.patches.obs.columns:
            raise ValueError(f"group_col '{group_col}' not found in patches.obs")

        images = extract_patch_images(
            self.patches, slides,
            tile_key=tile_key, channel_first=False,
            progress_bar=True, skip_errors=True, cache=cache,
        )
        images = list(images) if isinstance(images, np.ndarray) else images

        border_group_ids = (
            self.patches.obs[group_col].tolist() if group_col is not None else None
        )

        self._blocks.append(_RowBlock(
            label_per_row=[label],
            n_rows=1,
            frames=[[img] for img in images],
            border_group_ids=border_group_ids,
            border_cmap=cmap,
            border_extend=border_extend,
            border_alpha=border_alpha,
        ))
        return self

    def add_cluster_map_rows(
        self,
        clusterizers: List["TokenClusterizer"],
        model=None,
        slides=None,
        *,
        tile_key: str = DEFAULT_TILE_KEY,
        cmap: str = 'viridis',
        blend_with_previous: bool = True,
        saliency_alpha_power: float = 1.0,
        batch_size: int = 16,
        cache: bool = False,
    ) -> "GalleryPlan":
        """Add one row per clusterizer, each a cluster-map overlay.

        When `blend_with_previous=True` (default), each row blends its own
        clusterizer's map onto the same-patch-column frame from the most
        recently added block (e.g. the H&E row added just before), without
        consuming or replacing that block -- it still contributes its own
        unmodified row.
        """
        if not clusterizers:
            raise ValueError("At least one clusterizer must be provided.")
        names = [c.feature_name for c in clusterizers]
        if any(not n for n in names):
            raise ValueError("Every clusterizer must have a non-empty feature_name.")
        if len(set(names)) != len(names):
            dupes = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"clusterizers must have distinct feature_name values; duplicates: {dupes}")

        per_clusterizer_maps = [
            extract_cluster_maps(
                self.patches, c, model, slides=slides,
                batch_size=batch_size, progress_bar=True, cache=cache,
            )
            for c in clusterizers
        ]
        n_clusterizers = len(clusterizers)
        is_list_result = isinstance(per_clusterizer_maps[0], list)
        if is_list_result:
            cluster_maps = [
                np.stack([per_clusterizer_maps[k][i] for k in range(n_clusterizers)], axis=0)
                for i in range(self.n_patches)
            ]
        else:
            stacked = np.stack(per_clusterizer_maps, axis=0)  # (K, N, H, W)
            cluster_maps = list(stacked.transpose(1, 0, 2, 3))  # N x (K, H, W)

        prev_block = self._blocks[-1] if self._blocks else None
        colormap = plt.get_cmap(cmap)
        n_clusters = 3  # matches the fixed cluster count used elsewhere for saliency alpha

        frames = [[] for _ in range(self.n_patches)]
        for k in range(n_clusterizers):
            for patch_idx in range(self.n_patches):
                cmap_vals = cluster_maps[patch_idx][k].astype(np.float32) / (n_clusters - 1)
                colored = colormap(np.clip(cmap_vals, 0, 1))[..., :3]

                if blend_with_previous:
                    if prev_block is None:
                        base = np.ones_like(colored)  # blank white base
                    elif prev_block.n_rows == n_clusterizers:
                        base = _to_float01_rgb(prev_block.frames[patch_idx][k])
                    else:
                        base = _to_float01_rgb(prev_block.frames[patch_idx][-1])
                    # alpha = np.clip(cmap_vals ** saliency_alpha_power, 0, 1)[..., None]
                    alpha = 0.5
                    blended = base * (1 - alpha) + colored * alpha
                else:
                    blended = colored

                frames[patch_idx].append(blended)

        self._blocks.append(_RowBlock(
            label_per_row=names,
            n_rows=n_clusterizers,
            frames=frames,
        ))
        return self

    def add_cycif_rows(
        self,
        channels: List[str],
        *,
        slides=None,
        tile_key: str = DEFAULT_TILE_KEY,
        marker_table: Optional["pd.DataFrame"] = None,
        marker_col: str = 'marker_name',
        vmin=None,
        vmax=None,
        merge: bool = True,
        merge_colors: Optional[Dict[str, Union[str, Tuple[float, float, float]]]] = None,
        single_channel_color: Union[str, Tuple[float, float, float]] = 'white',
        cache: bool = False,
    ) -> "GalleryPlan":
        """Add an optional multicolor-merge row followed by one row per CyCIF channel."""
        channel_idx = channel_indices_from_markers(channels, marker_table, marker_col)
        cycif_arr = extract_patch_images(
            self.patches, slides,
            channels=channel_idx, tile_key=tile_key,
            progress_bar=True, skip_errors=True, cache=cache,
        )

        vmin_map = _resolve_per_channel(vmin, channels)
        vmax_map = _resolve_per_channel(vmax, channels)
        for ci, ch in enumerate(channels):
            if vmin_map[ch] is None:
                vmin_map[ch] = _channel_percentile(cycif_arr, ci, 1)
            if vmax_map[ch] is None:
                vmax_map[ch] = _channel_percentile(cycif_arr, ci, 99)

        single_rgb = np.array(_resolve_color(single_channel_color))
        merge_rgb = {
            ch: np.array(_resolve_color(
                (merge_colors or {}).get(ch, MARKER_COLOR_DEFAULTS.get(ch, 'white'))
            ))
            for ch in channels
        }

        label_per_row = (["Merge"] if merge else []) + list(channels)
        frames = []
        for patch_idx in range(self.n_patches):
            channel_imgs = cycif_arr[patch_idx]  # (C, H, W)
            normalized = []
            for ci, ch in enumerate(channels):
                img = channel_imgs[ci].astype(np.float32)
                vlo, vhi = vmin_map[ch], vmax_map[ch]
                normalized.append(np.clip((img - vlo) / (vhi - vlo), 0, 1))

            rows = []
            if merge:
                colorized_for_merge = [n[..., None] * merge_rgb[ch] for n, ch in zip(normalized, channels)]
                rows.append(1 - np.prod([1 - c for c in colorized_for_merge], axis=0))
            for n in normalized:
                rows.append(n[..., None] * single_rgb)
            frames.append(rows)

        self._blocks.append(_RowBlock(
            label_per_row=label_per_row,
            n_rows=len(label_per_row),
            frames=frames,
        ))
        return self

    def render(
        self,
        *,
        output_path: Optional[str] = None,
        patches_per_row: int = 10,
        samples_per_figure: int = 100,
        max_rows_per_page: Optional[int] = None,
        patch_display_size: float = 2.0,
        margin: float = 0.02,
        dpi: int = 300,
        title: Optional[str] = None,
        show_slide_ids: bool = False,
        show_scores: bool = False,
        return_fig: bool = False,
        return_buffer: bool = False,
        progress_bar: bool = True,
    ) -> Optional[Union[Tuple[plt.Figure, np.ndarray], List[io.BytesIO]]]:
        """Paginate and render the accumulated rows.

        If `max_rows_per_page` is given, it (together with `patches_per_row`
        and the total rows contributed by all added blocks) determines how
        many patches fit per page: `blocks_per_page = max_rows_per_page //
        total_rows_per_patch`, `patches_per_page = blocks_per_page *
        patches_per_row`. Otherwise falls back to `samples_per_figure`
        patches per page with patch-bands stacked without a row cap.
        """
        if not self._blocks:
            raise ValueError("GalleryPlan has no rows to render; call an add_*_row method first.")

        patch_df = self.patches.obs
        n_patches = len(patch_df)
        for block in self._blocks:
            if len(block.frames) != n_patches:
                raise ValueError(
                    f"Row block '{block.label_per_row}' has {len(block.frames)} patches, "
                    f"expected {n_patches}."
                )

        if show_slide_ids and SLIDE_ID not in patch_df.columns:
            raise ValueError(f"show_slide_ids=True requires a '{SLIDE_ID}' column in patches.obs")
        if show_scores and 'score' not in patch_df.columns:
            raise ValueError("show_scores=True requires 'score' column in patches.obs")

        total_rows_per_patch = sum(b.n_rows for b in self._blocks)

        if max_rows_per_page is not None:
            blocks_per_page = max(1, max_rows_per_page // total_rows_per_patch)
            patches_per_page = blocks_per_page * patches_per_row
        else:
            patches_per_page = samples_per_figure

        n_pages = int(np.ceil(n_patches / patches_per_page))
        if n_pages > 1 and output_path is None and not return_buffer:
            raise ValueError(
                f"Dataset has {n_patches} patches requiring {n_pages} pages. "
                "Please provide output_path and/or return_buffer=True for multi-page figures."
            )

        patch_titles = []
        for _, row in patch_df.iterrows():
            parts = []
            if show_slide_ids:
                parts.append(str(row[SLIDE_ID]))
            if show_scores:
                parts.append(f"Score: {row.get('score', 0):.3f}")
            patch_titles.append('\n'.join(parts) if parts else None)

        row_labels = [lbl for block in self._blocks for lbl in block.label_per_row]

        buffers = [] if return_buffer else None

        for page_idx in range(n_pages):
            start = page_idx * patches_per_page
            end = min(start + patches_per_page, n_patches)
            n_samples = end - start
            n_cols = min(patches_per_row, n_samples)
            n_bands = int(np.ceil(n_samples / patches_per_row))
            n_rows_total = n_bands * total_rows_per_patch

            if progress_bar and n_pages > 1:
                print(f"Rendering page {page_idx + 1}/{n_pages} (patches {start + 1}-{end})...")

            fig, axes = plt.subplots(
                n_rows_total, n_cols,
                figsize=(patch_display_size * n_cols, patch_display_size * n_rows_total),
            )
            axes = _normalize_axes(axes, n_rows_total, n_cols)

            for i in range(n_samples):
                patch_idx = start + i
                band = i // patches_per_row
                col = i % patches_per_row
                row_base = band * total_rows_per_patch

                r = row_base
                for block in self._blocks:
                    for k in range(block.n_rows):
                        ax = axes[r, col]
                        ax.imshow(block.frames[patch_idx][k])
                        if k == 0 and block.border_group_ids is not None:
                            color = _group_color_lookup(block.border_group_ids, block.border_cmap)[
                                block.border_group_ids[patch_idx]
                            ]
                            _draw_group_border(ax, color, patch_display_size, block.border_extend, block.border_alpha)
                        if r == row_base and patch_titles[patch_idx]:
                            ax.set_title(patch_titles[patch_idx], fontsize=8)
                        ax.axis('off')
                        r += 1

            for i in range(n_samples, n_bands * n_cols):
                band = i // n_cols
                col = i % n_cols
                for rr in range(band * total_rows_per_patch, (band + 1) * total_rows_per_patch):
                    axes[rr, col].axis('off')
                    axes[rr, col].set_visible(False)

            if title and n_pages == 1:
                fig.suptitle(title, fontsize=16)

            plt.tight_layout()
            fig.subplots_adjust(left=margin, right=1 - margin, top=1 - margin, bottom=margin)

            for band in range(n_bands):
                for row_idx, label in enumerate(row_labels):
                    r = band * total_rows_per_patch + row_idx
                    y = 1 - (r + 0.5) / n_rows_total
                    fig.text(0, y, label, fontsize=12, rotation=90, va='center', ha='center')

            fp = None
            if output_path is not None:
                fp = output_path if n_pages == 1 else _paged_path(output_path, start, end)
                Path(fp).parent.mkdir(parents=True, exist_ok=True)

            keep_alive = n_pages == 1 and return_fig and not return_buffer
            result = _finish_plot(fig, axes, show=keep_alive, save=fp,
                                   return_fig=False, return_buffer=return_buffer, dpi=dpi)
            if fp is not None:
                print(f"Saved: {fp}")
            if return_buffer:
                buffers.append(result)
            if n_pages == 1 and return_fig and not return_buffer:
                return fig, axes

        if return_buffer:
            return buffers
        return None
