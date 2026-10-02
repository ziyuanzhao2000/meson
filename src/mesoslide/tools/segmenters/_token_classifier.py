"""Nearest-neighbour token classifier over labelled reference tokens."""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from sklearn.utils.validation import check_is_fitted

from ._token_labeler import _TokenLabeler

_METRICS = ("cosine", "euclidean")
_DTYPES = {"float32": torch.float32, "float64": torch.float64}


class TokenClassifier(_TokenLabeler):
    """
    Label each ViT token with the class of its nearest reference token (1-NN).

    The reference tokens (`prototypes_`) and their classes
    (`prototype_labels_`) come from :meth:`fit` or, for a classifier already
    trained with scikit-learn, from :meth:`from_sklearn`. Prediction is a
    chunked matrix product on `device`, so whole slides can be labelled on a
    GPU via ``feature_extraction(dense=True, reducer=clf.reducer())``.

    Shares `transform`, `as_stage`, `reducer` and `display_name` with
    :class:`TokenClusterer`, so it can be passed wherever a clusterer is
    (e.g. :func:`mesoslide.preprocessing.extract_cluster_maps`, gallery rows).

    Parameters
    ----------
    metric : {'cosine', 'euclidean'}, default='cosine'
    interpolation : {'nearest', 'bilinear'}, default='nearest'
        Upsampling method used by `transform`.
    name : str, optional
        Label for this classifier (cache key and gallery row label).
    device : str, optional
        Torch device for prediction. Defaults to "cuda" if available.
    dtype : {'float32', 'float64'}, default='float32'
        Precision of the distance computation. 'float64' reproduces
        scikit-learn's float64 result exactly, including near ties.
    chunk_size : int, default=65536
        Tokens per matrix product.

    Attributes
    ----------
    prototypes_ : ndarray of shape (n_prototypes, D)
    prototype_labels_ : ndarray of shape (n_prototypes,)
        Integer class of each prototype, in [0, 255].
    classes_ : ndarray
        Sorted unique classes.
    n_features_in_ : int
    grid_size_ : tuple of int or None
        Token grid (gh, gw); inferred as square when unset.
    patch_size_ : tuple of int or None
        ViT patch size in pixels; sets `transform`'s default output size.
    model_name_ : str or None
        `lazyslide_models.MODEL_REGISTRY` key of the vision model the
        prototypes were embedded with.
    """

    def __init__(
        self,
        metric: str = "cosine",
        *,
        interpolation: str = "nearest",
        name: Optional[str] = None,
        device: Optional[str] = None,
        dtype: str = "float32",
        chunk_size: int = 65536,
    ):
        self.metric = metric
        self.interpolation = interpolation
        self.name = name
        self.device = device
        self.dtype = dtype
        self.chunk_size = chunk_size

    def _validate_params(self):
        if self.metric not in _METRICS:
            raise ValueError(f"metric must be one of {_METRICS}, got {self.metric!r}")
        if self.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {tuple(_DTYPES)}, got {self.dtype!r}")
        self._validate_interpolation()

    @property
    def n_labels_(self) -> int:
        check_is_fitted(self, "prototype_labels_")
        return int(self.prototype_labels_.max()) + 1

    def fit(self, X, y) -> "TokenClassifier":
        """
        Store labelled reference tokens.

        Parameters
        ----------
        X : array-like of shape (n_tokens, D), or (B, N_tokens, D) with `y` of shape (B, N_tokens)
        y : array-like of int, labels in [0, 255]

        Returns
        -------
        self
        """
        self._validate_params()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])
            y = y.reshape(-1)
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"Expected X (n, D) and y (n,), got {X.shape} and {y.shape}")
        if not np.issubdtype(y.dtype, np.integer) or y.min() < 0 or y.max() > 255:
            raise ValueError("Labels must be integers in [0, 255] (maps are uint8).")
        self._tensor_cache = None
        self.prototypes_ = X
        self.prototype_labels_ = y.astype(np.int64)
        self.classes_ = np.unique(self.prototype_labels_)
        self.n_features_in_ = X.shape[1]
        return self

    @classmethod
    def from_sklearn(
        cls,
        knn,
        *,
        name: Optional[str] = None,
        grid_size: Optional[tuple] = None,
        patch_size: Optional[tuple] = None,
        model_name: Optional[str] = None,
        **kwargs,
    ) -> "TokenClassifier":
        """
        Build from a fitted `sklearn.neighbors.KNeighborsClassifier`.

        Only ``n_neighbors=1`` is supported. The metric is taken from `knn`
        ('cosine', or 'euclidean' / 'minkowski' with p=2).

        Parameters
        ----------
        knn : KNeighborsClassifier
        name, grid_size, patch_size, model_name
            Set `name`, `grid_size_`, `patch_size_` and `model_name_`.
        **kwargs
            Other constructor parameters (`device`, `dtype`, ...).
        """
        from sklearn.neighbors import KNeighborsClassifier

        if not isinstance(knn, KNeighborsClassifier):
            raise TypeError(f"Expected a KNeighborsClassifier, got {type(knn).__name__}")
        check_is_fitted(knn)
        if knn.n_neighbors != 1:
            raise NotImplementedError(
                f"Only n_neighbors=1 is supported, got n_neighbors={knn.n_neighbors}."
            )
        if knn.metric == "cosine":
            metric = "cosine"
        elif knn.metric == "euclidean" or (knn.metric == "minkowski" and knn.p == 2):
            metric = "euclidean"
        else:
            raise NotImplementedError(f"Unsupported metric {knn.metric!r} (p={knn.p}).")

        labels = np.asarray(knn.classes_)[np.asarray(knn._y)]
        clf = cls(metric, name=name, **kwargs).fit(np.asarray(knn._fit_X), labels)
        clf.grid_size_ = None if grid_size is None else tuple(grid_size)
        clf.patch_size_ = None if patch_size is None else tuple(patch_size)
        clf.model_name_ = model_name
        return clf

    def _torch_device(self) -> torch.device:
        if self.device is not None:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _prototype_tensors(self, device: torch.device, dtype: torch.dtype):
        """Prototypes (normalized for cosine) and labels on `device`, cached per (device, dtype)."""
        key = (str(device), dtype)
        cache = getattr(self, "_tensor_cache", None)
        if cache is None or cache[0] != key:
            ref = torch.as_tensor(self.prototypes_, dtype=dtype, device=device)
            if self.metric == "cosine":
                ref = ref / ref.norm(dim=1, keepdim=True).clamp_min(1e-30)
            labels = torch.as_tensor(self.prototype_labels_, device=device)
            self._tensor_cache = (key, ref, (ref ** 2).sum(1), labels)
        return self._tensor_cache[1:]

    def predict_flat(self, X) -> np.ndarray:
        """
        Class of the nearest prototype for each row of `X`.

        Parameters
        ----------
        X : array-like or torch.Tensor of shape (n, D)

        Returns
        -------
        labels : ndarray of shape (n,), dtype uint8
        """
        check_is_fitted(self, "prototypes_")
        self._validate_params()
        device, dtype = self._torch_device(), _DTYPES[self.dtype]
        ref, ref_sq, labels = self._prototype_tensors(device, dtype)
        X = torch.as_tensor(X) if not torch.is_tensor(X) else X
        if X.ndim != 2 or X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected tokens of shape (n, {self.n_features_in_}), got {tuple(X.shape)}")

        out = []
        with torch.inference_mode():
            for block in X.split(self.chunk_size):
                block = block.to(device=device, dtype=dtype)
                if self.metric == "cosine":
                    block = block / block.norm(dim=1, keepdim=True).clamp_min(1e-30)
                    nearest = (block @ ref.T).argmax(1)
                else:
                    # Squared distance up to the per-token ||x||^2 term, which doesn't change argmin.
                    nearest = (ref_sq[None, :] - 2.0 * block @ ref.T).argmin(1)
                out.append(labels[nearest])
        return torch.cat(out).to(torch.uint8).cpu().numpy()

    def predict(self, X) -> np.ndarray:
        """
        Class of the nearest prototype per token.

        Parameters
        ----------
        X : array-like or torch.Tensor of shape (B, N_tokens, D)

        Returns
        -------
        labels : ndarray of shape (B, grid_h, grid_w), dtype uint8
        """
        X = torch.as_tensor(X) if not torch.is_tensor(X) else X
        if X.ndim != 3:
            raise ValueError(f"Expected token embeddings of shape (B, N_tokens, D), got {tuple(X.shape)}")
        B, N, D = X.shape
        gh, gw = self._grid_for(N)
        return self.predict_flat(X.reshape(B * N, D)).reshape(B, gh, gw)

    def __getstate__(self):
        state = super().__getstate__()
        state.pop("_tensor_cache", None)
        return state

    def save(self, path: Union[str, Path]) -> None:
        """Write the fitted classifier to an `.npz` file (no pickled objects)."""
        check_is_fitted(self, "prototypes_")
        meta = {
            "metric": self.metric, "interpolation": self.interpolation, "name": self.name or "",
            "model_name": getattr(self, "model_name_", None) or "",
        }
        grid = getattr(self, "grid_size_", None)
        patch = getattr(self, "patch_size_", None)
        np.savez_compressed(
            path, prototypes=self.prototypes_, prototype_labels=self.prototype_labels_,
            grid_size=np.asarray(grid if grid is not None else [], dtype=np.int64),
            patch_size=np.asarray(patch if patch is not None else [], dtype=np.int64),
            **{k: np.asarray(v) for k, v in meta.items()},
        )

    @classmethod
    def load(cls, path: Union[str, Path], **kwargs) -> "TokenClassifier":
        """Read a classifier written by :meth:`save`. `kwargs` set `device`, `dtype`, ..."""
        with np.load(path, allow_pickle=False) as f:
            clf = cls(str(f["metric"]), interpolation=str(f["interpolation"]),
                      name=str(f["name"]) or None, **kwargs)
            clf.fit(f["prototypes"], f["prototype_labels"])
            clf.grid_size_ = tuple(int(v) for v in f["grid_size"]) or None
            clf.patch_size_ = tuple(int(v) for v in f["patch_size"]) or None
            clf.model_name_ = str(f["model_name"]) or None
        return clf
