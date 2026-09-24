from typing import Callable, Optional

import numpy as np
from scipy.sparse import issparse


def feature_column_index(feature_name: str) -> int:
    """Column index of a sparse-coding feature from its name, `f"{prefix}_{i}"` -> `i`.

    Inverse of the naming used when writing features into `table.X`
    (`mesoslide.tools._feature_extraction._write_sparse_features`), e.g.
    'UNI_SAE_41985' -> 41985.
    """
    suffix = feature_name.rsplit("_", 1)[-1]
    if not suffix.isdigit():
        raise ValueError(
            f"Cannot read a column index from feature name {feature_name!r}; expected "
            f"'<prefix>_<index>'. Pass a scorer callable instead."
        )
    return int(suffix)


def feature_scorer(model, feature_name: str, *, device: Optional[str] = None) -> Callable:
    """
    Scoring function for one feature of a fitted sparse-coding model.

    Parameters
    ----------
    model : SparseAutoencoder, LocalityConstrainedCoding, or compatible
        Fitted model with `transform(X, column_keep_indices=None, device=None, *,
        progress_bar=...)` returning an (n, M) matrix.
    feature_name : str
        Feature name as written into `table.X`, e.g. 'UNI_SAE_41985'; its
        trailing integer is the model's output column.
    device : str, optional
        Forwarded to `model.transform`.

    Returns
    -------
    callable
        Maps patch embeddings, shape (n, D), to that feature's scores, shape (n,).
    """
    column = feature_column_index(feature_name)
    n_features = getattr(model, "embed_dim_", None)
    if n_features is not None and column >= n_features:
        raise ValueError(f"{feature_name!r} maps to column {column}, but the model has {n_features} features.")

    def score(X) -> np.ndarray:
        codes = model.transform(np.asarray(X), column_keep_indices=[column], device=device,
                                progress_bar=False)
        codes = codes.toarray() if issparse(codes) else np.asarray(codes)
        return codes[:, 0]

    return score
