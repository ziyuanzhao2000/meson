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
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Patch

from mesoslide._slides import DEFAULT_TILE_KEY, SLIDE_ID
from mesoslide.preprocessing._extract_patches import (
    _resolve_slides,
    _resolve_slides_from_ref,
    _tile_size,
    extract_patch_images,
)
from mesoslide.preprocessing._extract_cluster_maps import extract_cluster_maps
from mesoslide.preprocessing._utils import channel_indices_from_markers
from ._image_grid import _draw_group_border, _draw_corner_label, _group_color_lookup
from ._cell_overlay import (
    alpha_composite,
    cells_in_patch,
    rasterize_cell_polygons,
    resolve_categorical_palette,
    translate_to_patch_local,
)
from ._utils import _finish_plot, FLUOROPHORE_COLORS, MARKER_COLOR_DEFAULTS

if TYPE_CHECKING:
    import anndata as ad
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


def _default_cell_overlay_background(wsi) -> Tuple[float, float, float]:
    """White for an H&E slide, black for CyCIF/mIF -- same 3-channel
    heuristic as `preprocessing._extract_patches._resolve_obsm_key_post_read`.
    Reads a single pixel at the coarsest pyramid level to check channel
    count without materializing real pixel data."""
    n_level = wsi.properties.n_level
    sample = wsi.read_region(0, 0, 1, 1, level=n_level - 1)
    n_channels = sample.shape[-1] if sample.ndim == 3 else 1
    return (1.0, 1.0, 1.0) if n_channels == 3 else (0.0, 0.0, 0.0)


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
    legend_palette: Optional[Dict] = None  # {category: rgba}, drawn once per page by render()
    legend_title: Optional[str] = None


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
        cache: bool = True,
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

        legend_palette = None
        if merge:
            legend_palette = {ch: to_rgba(tuple(merge_rgb[ch])) for ch in channels}

        self._blocks.append(_RowBlock(
            label_per_row=label_per_row,
            n_rows=len(label_per_row),
            frames=frames,
            legend_palette=legend_palette,
            legend_title="Merge" if merge else None,
        ))
        return self

    def add_cell_overlay_row(
        self,
        *,
        cells_key: str = "cells",
        slides=None,
        tile_key: str = DEFAULT_TILE_KEY,
        color_by: Optional[str] = None,
        palette: Optional[Dict] = None,
        legend_title: Optional[str] = None,
        fill_alpha: float = 0.35,
        edge_color: Optional[Union[str, Tuple]] = "white",
        edge_only: bool = False,
        linewidth: float = 1.0,
        supersample: int = 4,
        blend_with_previous: bool = True,
        background_color: Optional[Union[str, Tuple]] = None,
        label: str = "Cells",
    ) -> "GalleryPlan":
        """Overlay cell polygons (from `wsidata.shapes[cells_key]`) onto each patch.

        Cells are rasterized onto an RGBA overlay and alpha-composited onto
        an existing frame -- see `mesoslide.plotting._cell_overlay` -- rather
        than drawn as live vector artists (how `lazyslide.pl.WSIViewer` draws
        them), so the result is a plain image frame like every other
        `GalleryPlan` row and `render()` needs no changes to display it.

        Parameters
        ----------
        cells_key : str, default='cells'
            `wsidata.shapes` key holding cell polygons, e.g. written by
            :func:`mesoslide.tl.add_cell_polygons`. May live on a different
            `WSIData` than the one the patch table was tiled on (e.g. cells
            segmented on a separately-registered CyCIF slide) -- patch
            geometry always comes from `patches.obs['_slide_ref']`/its own
            tile spec, independently of which slide `slides` resolves to.
        slides : WSIData, list of WSIData, or {slide_id: WSIData}, optional
            The slide(s) holding `cells_key`. Defaults to
            `patches.obs['_slide_ref']`, i.e. the same slide(s) the patch
            table itself came from.
        tile_key : str, default='tiles'
            Tile shapes key on the *reference* slide (`patches.obs['_slide_ref']`)
            used to size each patch -- see :func:`mesoslide.pp.extract_patch_images`.
        color_by : str, optional
            Column on the cells shapes GeoDataFrame (e.g. `'phenotype'`) to
            color fills by. Omit for a single fixed color. Resolved once,
            globally, from every touched slide's full `cells_key[color_by]`
            column (not per-patch) -- so a given category gets the same
            color everywhere in the gallery, not just within one patch.
        palette : dict, optional
            `{category: color}`. Resolved automatically (a fixed qualitative
            palette for few categories, `distinctipy`-generated distinct
            colors otherwise) if omitted.
        legend_title : str, optional
            Heading for the legend `render(legend=True)` draws for this
            row's palette. Defaults to `color_by`.
        blend_with_previous : bool, default=True
            Composite onto the most recently added block's last row, in
            place (no new row/label added) -- matching how
            `add_cluster_map_rows(blend_with_previous=True)` overlays onto
            the H&E row. `False` appends a standalone new row instead.
        background_color : str or tuple, optional
            Background for a standalone row (`blend_with_previous=False`
            only -- ignored otherwise, since that branch always composites
            onto an existing frame). `None` (the default) auto-detects per
            cells-slide: white for an H&E slide, black for CyCIF/mIF (the
            same 3-channel heuristic used in
            `preprocessing._extract_patches`).

        Returns
        -------
        GalleryPlan
        """
        if blend_with_previous and not self._blocks:
            raise ValueError(
                "add_cell_overlay_row(blend_with_previous=True) requires at least "
                "one row already added (e.g. add_he_row) to overlay onto."
            )

        opened: list = []
        try:
            # Cells slide(s): explicit `slides`, or the same reference the
            # patch table itself carries.
            if slides is not None:
                cells_slide_map = _resolve_slides(slides)
            else:
                cells_slide_map, opened = _resolve_slides_from_ref(self.patches)
            cells_single = set(cells_slide_map) == {None}

            # Patch geometry always comes from the reference slide's own tile
            # spec -- the cells slide (e.g. a separately-registered CyCIF
            # segmentation slide) is not guaranteed to have been tiled itself,
            # mirroring extract_patch_images's own ref-slide fallback.
            ref_slide_map, ref_opened = _resolve_slides_from_ref(self.patches)
            opened.extend(ref_opened)
            ref_single = set(ref_slide_map) == {None}
            sizes = {sid: _tile_size(wsi, tile_key) for sid, wsi in ref_slide_map.items()}

            patch_df = self.patches.obs
            required = ["x", "y"]
            if not cells_single or not ref_single:
                required.append(SLIDE_ID)
            missing = [c for c in required if c not in patch_df.columns]
            if missing:
                raise ValueError(f"patches.obs missing required columns: {missing}")

            # Resolve the categorical palette once, globally, across every
            # touched slide's full cells layer -- not per patch's local
            # subset. resolve_categorical_palette assigns colors by
            # positional index into whatever categories it's given, so
            # calling it separately per patch (each seeing only its own
            # local subset of categories) would assign the same category a
            # different color in different patches whenever patches don't
            # all contain the exact same categories in the same order.
            resolved_palette = None
            if color_by is not None:
                all_values = []
                for sid, wsi in cells_slide_map.items():
                    if cells_key not in wsi.shapes:
                        raise KeyError(
                            f"wsidata.shapes has no '{cells_key}' for slide '{sid}'. "
                            "Run mesoslide.tl.add_cell_polygons first."
                        )
                    all_values.append(wsi.shapes[cells_key][color_by])
                resolved_palette = resolve_categorical_palette(
                    pd.concat(all_values, ignore_index=True), palette
                )

            bg_cache: Dict = {}

            frames = []
            for i in range(self.n_patches):
                patch = patch_df.iloc[i]
                cells_slide_id = None if cells_single else patch[SLIDE_ID]
                ref_slide_id = None if ref_single else patch[SLIDE_ID]
                wsi = cells_slide_map[cells_slide_id]
                if cells_key not in wsi.shapes:
                    raise KeyError(
                        f"wsidata.shapes has no '{cells_key}' for slide '{cells_slide_id}'. "
                        "Run mesoslide.tl.add_cell_polygons first."
                    )
                h, w = sizes[ref_slide_id]
                x, y = int(patch.x), int(patch.y)

                cells_gdf = wsi.shapes[cells_key]
                local = translate_to_patch_local(cells_in_patch(cells_gdf, x, y, w, h), x, y)

                if blend_with_previous:
                    base = _to_float01_rgb(self._blocks[-1].frames[i][-1])
                else:
                    if background_color is not None:
                        bg_rgb = to_rgba(background_color)[:3]
                    elif cells_slide_id in bg_cache:
                        bg_rgb = bg_cache[cells_slide_id]
                    else:
                        bg_rgb = _default_cell_overlay_background(wsi)
                        bg_cache[cells_slide_id] = bg_rgb
                    base = np.full((h, w, 3), bg_rgb, dtype=np.float32)

                overlay = rasterize_cell_polygons(
                    base.shape[:2], local,
                    color_by=color_by, palette=resolved_palette,
                    fill_alpha=fill_alpha, edge_color=edge_color, edge_only=edge_only,
                    linewidth=linewidth, supersample=supersample,
                )
                composited = alpha_composite(base, overlay)

                if blend_with_previous:
                    self._blocks[-1].frames[i][-1] = composited
                else:
                    frames.append([composited])
        finally:
            for wsi in opened:
                try:
                    wsi.close()
                except Exception:
                    pass

        if blend_with_previous:
            if resolved_palette is not None:
                self._blocks[-1].legend_palette = resolved_palette
                self._blocks[-1].legend_title = legend_title or color_by
        else:
            self._blocks.append(_RowBlock(
                label_per_row=[label], n_rows=1, frames=frames,
                legend_palette=resolved_palette,
                legend_title=(legend_title or color_by) if resolved_palette is not None else None,
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
        legend: bool = True,
        legend_width: float = 2.5,
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

        `legend` draws one legend per block that carries a `legend_palette`
        (set by `add_cell_overlay_row(color_by=...)` or
        `add_cycif_rows(merge=True)`), once per page. The figure's *width*
        is expanded by `legend_width` inches to make room -- the patch grid
        itself keeps exactly its `patch_display_size`-driven physical size
        rather than being squeezed to fit the legend into the original
        canvas.
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
        legend_blocks = [b for b in self._blocks if b.legend_palette] if legend else []

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

            grid_width = patch_display_size * n_cols
            draw_legend_here = bool(legend_blocks)
            fig_width = grid_width + legend_width if draw_legend_here else grid_width

            fig, axes = plt.subplots(
                n_rows_total, n_cols,
                figsize=(fig_width, patch_display_size * n_rows_total),
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
            # The grid occupies exactly `grid_width` inches regardless of
            # whether a legend is drawn: when it is, `fig_width` was already
            # expanded by `legend_width` above, so re-expressing the same
            # absolute margins as fractions of the (now larger) fig_width
            # keeps the patch grid's physical size unchanged rather than
            # shrinking it to make room.
            right_frac = grid_width * (1 - margin) / fig_width
            left_frac = grid_width * margin / fig_width
            fig.subplots_adjust(left=left_frac, right=right_frac, top=1 - margin, bottom=margin)

            for band in range(n_bands):
                for row_idx, label in enumerate(row_labels):
                    r = band * total_rows_per_patch + row_idx
                    pos = axes[r, 0].get_position()
                    y = (pos.y0 + pos.y1) / 2
                    fig.text(0, y, label, fontsize=12, rotation=90, va='center', ha='center')

            if draw_legend_here:
                y_positions = (
                    [0.5] if len(legend_blocks) == 1
                    else list(np.linspace(0.8, 0.2, len(legend_blocks)))
                )
                for block, y_pos in zip(legend_blocks, y_positions):
                    handles = [
                        Patch(facecolor=color, label=str(cat))
                        for cat, color in sorted(block.legend_palette.items(), key=lambda kv: str(kv[0]))
                    ]
                    ncols = max(1, -(-len(handles) // 20))  # ceil(len / 20)
                    fig.legend(
                        handles=handles, title=block.legend_title,
                        loc='center left', bbox_to_anchor=(right_frac, y_pos),
                        fontsize=9, ncols=ncols,
                    )

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
