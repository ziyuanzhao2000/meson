from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.utils import check_random_state
import numpy as np
import scipy.sparse as sp
from tqdm.auto import tqdm


class MiniBatchDictionaryCoding(TransformerMixin, BaseEstimator):
    """
    L1 sparse coding with a learned dictionary, wrapping
    sklearn.decomposition.MiniBatchDictionaryLearning. Mirrors
    SparseAutoencoder / LocalityConstrainedCoding's fit/transform API.

    Solves min ||x - mean - c D||^2 + alpha ||c||_1 per patch. With
    positive_code=True, codes are nonnegative like SAE ReLU activations; with
    positive_dict=False, atoms are signed. Inputs are centered with the
    training mean (`center=True`) because, unlike an SAE, the model has no bias.

    Runs on CPU only; `device` arguments are accepted for interface parity and
    ignored.

    Parameters
    ----------
    n_components, alpha, fit_algorithm, transform_algorithm, transform_alpha,
    positive_code, positive_dict, batch_size, max_iter, n_jobs
        Forwarded to MiniBatchDictionaryLearning. transform_alpha=None
        defaults to alpha. Tune alpha to reach a target mean number of active
        features per patch (see `_training_log["mean_l0"]` after fit).
        fit_algorithm defaults to "cd" because sklearn rejects
        positive_code=True with fit_algorithm="lars".
    center : bool, default=True
        Subtract the training mean in fit and transform.
    transform_batch_size : int, default=65536
        Rows encoded per call in transform; each block is sparsified before
        the next, bounding memory for the dense code.
    dict_kwargs : dict, optional
        Extra keyword arguments for MiniBatchDictionaryLearning (e.g. `tol`,
        `max_no_improvement`, `transform_max_iter`).
    random_state : int, RandomState or None
    """

    def __init__(self,
                 n_components: int = 1024,
                 alpha: float = 1.0,
                 fit_algorithm: str = "cd",
                 transform_algorithm: str = "lasso_lars",
                 transform_alpha: "float | None" = None,
                 positive_code: bool = True,
                 positive_dict: bool = False,
                 batch_size: int = 2048,
                 max_iter: int = 50,
                 center: bool = True,
                 transform_batch_size: int = 65536,
                 n_jobs: "int | None" = -1,
                 dict_kwargs: "dict | None" = None,
                 random_state=None):
        self.n_components = n_components
        self.alpha = alpha
        self.fit_algorithm = fit_algorithm
        self.transform_algorithm = transform_algorithm
        self.transform_alpha = transform_alpha
        self.positive_code = positive_code
        self.positive_dict = positive_dict
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.center = center
        self.transform_batch_size = transform_batch_size
        self.n_jobs = n_jobs
        self.dict_kwargs = dict_kwargs
        self.random_state = random_state

    @property
    def components_(self) -> np.ndarray:
        """Dictionary atoms, shape (n_components, n_features)."""
        check_is_fitted(self)
        return self.model_.components_

    def fit(self, X, y=None, *,
            obsm_key=None, tile_key="tiles",
            device=None,
            fraction: float = 1.0,
            sample_size: "int | None" = None,
            verbose: "int | bool" = False):
        if obsm_key is not None:
            from mesoslide._slides import SlideSource
            source = SlideSource(X, tile_key=tile_key)
            X = np.vstack([table.obsm[obsm_key] for _, table in source])

        self.random_state_ = check_random_state(self.random_state)
        X = validate_data(self, X, accept_sparse=False, dtype=[np.float64, np.float32])
        assert len(X.shape) == 2  # expect X shape = B x d_emb
        if not 0 < fraction <= 1:
            raise ValueError(f"fraction must be in (0, 1], got {fraction}")
        if fraction < 1:
            n_samples = round(fraction * len(X))
            idx = self.random_state_.choice(len(X), size=n_samples, replace=False)
            X = X[idx]
        if sample_size is not None and sample_size < len(X):
            idx = self.random_state_.choice(len(X), size=sample_size, replace=False)
            X = X[idx]

        self.mean_ = X.mean(axis=0) if self.center else np.zeros(X.shape[1], dtype=X.dtype)
        X = X - self.mean_

        self.embed_dim_ = self.n_components
        self.model_ = MiniBatchDictionaryLearning(
            n_components=self.n_components,
            alpha=self.alpha,
            fit_algorithm=self.fit_algorithm,
            transform_algorithm=self.transform_algorithm,
            transform_alpha=self.transform_alpha,
            positive_code=self.positive_code,
            positive_dict=self.positive_dict,
            batch_size=self.batch_size,
            max_iter=self.max_iter,
            n_jobs=self.n_jobs,
            random_state=int(self.random_state_.randint(0, 2**31 - 1)),
            verbose=verbose,
            **(self.dict_kwargs or {}),
        )
        self.model_.fit(X)

        # Mean active features per patch on a training subsample, for tuning alpha.
        n_eval = min(len(X), 10000)
        eval_idx = self.random_state_.choice(len(X), size=n_eval, replace=False)
        mean_l0 = float((self.model_.transform(X[eval_idx]) != 0).sum(axis=1).mean())
        self._training_log = {
            "n_iter": self.model_.n_iter_,
            "n_steps": self.model_.n_steps_,
            "mean_l0": mean_l0,
        }
        if verbose:
            print(f"Mean active features per patch: {mean_l0:.2f}")
        return self

    def transform(self, X, column_keep_indices=None, device=None, *,
                  obsm_key=None, tile_key="tiles", sparse_key_added=None,
                  overwrite=False, save=True, progress_bar=False):
        if obsm_key is not None:
            from mesoslide.tools._feature_extraction import _write_sparse_features

            slides = X if isinstance(X, (list, tuple)) else [X]
            table_key = f"{tile_key}_table"
            sparse_key = sparse_key_added or f"{obsm_key}_dl"
            for slide in slides:
                table = slide.tables[table_key]
                if overwrite or not any(
                    v.startswith(f"{sparse_key}_") for v in table.var_names
                ):
                    matrix = self.transform(
                        table.obsm[obsm_key], column_keep_indices, device,
                        progress_bar=progress_bar,
                    )
                    table = _write_sparse_features(table, sparse_key, matrix)
                    slide.tables[table_key] = table
                    if save:
                        slide.write_element(table_key)
            return X

        check_is_fitted(self)
        X = validate_data(self, X, accept_sparse=False, reset=False,
                          dtype=[np.float64, np.float32])
        n_features = self.embed_dim_ if column_keep_indices is None else len(column_keep_indices)
        if len(X) == 0:
            return sp.csr_matrix((0, n_features), dtype=np.float32)

        blocks = []
        starts = range(0, len(X), self.transform_batch_size)
        for start in tqdm(starts, disable=not progress_bar):
            block = X[start:start + self.transform_batch_size] - self.mean_
            # Lasso codes depend on every atom, so encode fully, then slice.
            code = self.model_.transform(block)
            if column_keep_indices is not None:
                code = code[:, column_keep_indices]
            blocks.append(sp.csr_matrix(code.astype(np.float32)))
        return sp.vstack(blocks, format="csr")

    def inverse_transform(self, Xt) -> np.ndarray:
        """Reconstruct descriptors from codes (x_hat = codes @ D + mean)."""
        check_is_fitted(self)
        return np.asarray(Xt @ self.model_.components_) + self.mean_
