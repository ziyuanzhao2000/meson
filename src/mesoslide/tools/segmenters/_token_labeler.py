"""Shared base for estimators that assign one uint8 label per ViT token.

`TokenClusterer` (KMeans centroids, ordered against a feature score) and
`TokenClassifier` (labelled reference tokens, nearest neighbour) both map a
batch of token embeddings ``(B, N_tokens, D)`` to a label grid
``(B, grid_h, grid_w)``. Everything downstream of that -- upsampling to
pixels, wrapping as a `ModelStage`, naming, the cache key used by
:func:`mesoslide.preprocessing.extract_cluster_maps` -- lives here, so both
plug into the same galleries and feature-extraction calls.
"""

from typing import Callable, Optional

import cv2
import numpy as np
import torch
from sklearn.base import BaseEstimator, TransformerMixin

from mesoslide.tools._model_stage import CallableStage

_INTERPOLATIONS = {"nearest": cv2.INTER_NEAREST_EXACT, "bilinear": cv2.INTER_LINEAR}


def _as_tokens(X) -> np.ndarray:
    """Token embeddings as a float64 (B, N_tokens, D) array."""
    if torch.is_tensor(X):
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"Expected token embeddings of shape (B, N_tokens, D), got {X.shape}")
    return X


def _square_grid(n_tokens: int) -> tuple:
    side = int(round(np.sqrt(n_tokens)))
    if side * side != n_tokens:
        raise ValueError(
            f"Cannot infer a square token grid from {n_tokens} tokens; set grid_size_."
        )
    return (side, side)


class _TokenLabeler(TransformerMixin, BaseEstimator):
    """Base class: subclasses implement `predict` and `n_labels_`.

    Subclasses must define the constructor parameters `interpolation` and
    `name`, and may set the fitted attributes `grid_size_`, `patch_size_`,
    `model_name_` and `feature_name_`.
    """

    @property
    def display_name(self) -> str:
        """`name` if set, else `feature_name_`, else ''."""
        return self.name or getattr(self, "feature_name_", None) or ""

    @property
    def n_labels_(self) -> int:
        """Labels lie in ``[0, n_labels_)``; used to scale colormaps."""
        raise NotImplementedError

    def predict(self, X) -> np.ndarray:
        """Label per token, shape (B, grid_h, grid_w), dtype uint8."""
        raise NotImplementedError

    def _validate_interpolation(self):
        if self.interpolation not in _INTERPOLATIONS:
            raise ValueError(
                f"interpolation must be one of {tuple(_INTERPOLATIONS)}, got {self.interpolation!r}"
            )

    def _set_grid_size(self, n_tokens: int):
        """Keep a grid_size_ already set from the model; otherwise infer a square grid."""
        grid = getattr(self, "grid_size_", None)
        if grid is not None:
            if grid[0] * grid[1] != n_tokens:
                raise ValueError(f"Expected {grid[0] * grid[1]} tokens ({grid[0]}x{grid[1]} grid), got {n_tokens}")
            return
        self.grid_size_ = _square_grid(n_tokens)

    def _grid_for(self, n_tokens: int) -> tuple:
        """`grid_size_` if set (checked against `n_tokens`), else a square grid."""
        grid = getattr(self, "grid_size_", None)
        if grid is None:
            return _square_grid(n_tokens)
        if grid[0] * grid[1] != n_tokens:
            raise ValueError(f"Expected {grid[0] * grid[1]} tokens ({grid[0]}x{grid[1]} grid), got {n_tokens}")
        return tuple(grid)

    def transform(self, X, output_size: Optional[tuple] = None) -> np.ndarray:
        """
        Token labels upsampled to pixel resolution.

        Parameters
        ----------
        X : array-like or torch.Tensor of shape (B, N_tokens, D)
        output_size : tuple, optional
            Target (height, width). Defaults to grid_size_ * patch_size_.

        Returns
        -------
        label_masks : ndarray of shape (B, H, W), dtype uint8
        """
        self._validate_interpolation()
        labels = self.predict(X)
        if output_size is None:
            if getattr(self, "patch_size_", None) is None:
                raise ValueError("output_size is required when patch_size_ is unknown.")
            (gh, gw), (ph, pw) = labels.shape[1:], self.patch_size_
            output_size = (gh * ph, gw * pw)
        H, W = output_size
        interp = _INTERPOLATIONS[self.interpolation]
        out = np.empty((len(labels), H, W), dtype=np.uint8)
        for i, grid in enumerate(labels):
            out[i] = cv2.resize(grid, (W, H), interpolation=interp)
        return out

    def as_stage(self, *, output_size: Optional[tuple] = None, name: Optional[str] = None,
                 cache: bool = True, overwrite: bool = False) -> CallableStage:
        """Wrap `transform` as a `ModelStage` for `run_model_stages`.

        Chain it downstream of an `ImageModelStage(dense=True)`. `output_size`
        is fixed for the whole stage since one batch shares one resolution.
        """
        stage_name = name or self.display_name or f"{type(self).__name__.lower()}_{id(self)}"
        return CallableStage(
            lambda token_embeddings: self.transform(token_embeddings, output_size),
            name=stage_name, input_kind="dense",
            output_kind="dense", cache=cache, overwrite=overwrite, device="cpu",
        )

    def reducer(self) -> Callable[[torch.Tensor], torch.Tensor]:
        """
        A `reducer` for :func:`mesoslide.tools.feature_extraction` with `dense=True`.

        Maps token embeddings ``(B, N_tokens, D)`` to labels ``(B, N_tokens)``
        (float32, on the input's device), so the label grid of every tile is
        cached flat in the tile table's obsm. Reshape a row with `grid_size_`
        (row-major), or pass the obsm key to :func:`mesoslide.tools.token_label_map`.
        """
        def reduce(patch_tokens: torch.Tensor) -> torch.Tensor:
            labels = self.predict(patch_tokens)  # (B, gh, gw) uint8
            flat = torch.from_numpy(labels.reshape(len(labels), -1)).to(torch.float32)
            return flat.to(patch_tokens.device) if torch.is_tensor(patch_tokens) else flat

        return reduce
