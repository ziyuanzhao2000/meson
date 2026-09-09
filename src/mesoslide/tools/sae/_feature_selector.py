import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class SAEFeatureSelector:
    """
    Selects informative SAE features based on activation frequency and score magnitude.

    Follows a fit/select pattern: compute_activation_stats() is the expensive step,
    after which thresholds can be adjusted freely without recomputation.

    Parameters
    ----------
    pct_threshold : float
        Minimum fraction of patches in which a feature must be active.
    max_score_threshold : float
        Minimum peak activation score a feature must reach across all patches.
    n_chunks : int
        Number of chunks to split the data into for memory-efficient computation.
    """

    def __init__(self, pct_threshold=0.01, max_score_threshold=0.5, n_chunks=40):
        self.pct_threshold = pct_threshold
        self.max_score_threshold = max_score_threshold
        self.n_chunks = n_chunks

        # set after fit
        self.pct_active_ = None      # fraction of patches where each feature fires
        self.max_score_ = None       # peak activation score per feature
        self._is_fitted = False

        # accumulators, so a cohort can be folded in one slide at a time
        self._n_active = None
        self._n_obs = 0
        self._max = None

    def compute_activation_stats(self, adata, feature_prefix, num_features):
        """
        Compute per-feature activation frequency and peak score.
        This is the expensive step -- only needs to be run once.

        Parameters
        ----------
        adata : AnnData
            Patch-level AnnData with sparse SAE embeddings in .X
        feature_prefix : str
            Prefix of feature columns, e.g. 'UNI_SAE'
        num_features : int
            Total number of SAE features

        See Also
        --------
        fit_slides : the same statistics streamed over a cohort.
        """
        self.start(num_features)
        self.accumulate(adata, feature_prefix, num_features)
        return self.finalize()

    def fit_slides(self, slides, feature_prefix, num_features, *,
                   tile_key="tiles", progress=True):
        """
        Compute activation statistics over a cohort, one slide at a time.

        Exactly equivalent to concatenating every slide and calling
        :meth:`compute_activation_stats`, but peak memory is one slide.
        Both statistics reduce associatively over rows: the peak score is a
        max, and the active fraction is a sum of nonzero counts over a sum of
        row counts.

        Parameters
        ----------
        slides : slides_table, AnnData, WSIData, or sequence/mapping of either
        feature_prefix : str
        num_features : int
        tile_key : str, default='tiles'
        progress : bool

        Returns
        -------
        self
        """
        from mesoslide._slides import SlideSource

        source = slides if isinstance(slides, SlideSource) else SlideSource(slides, tile_key=tile_key)
        self.start(num_features)
        it = tqdm(source, desc="Activation stats") if progress else source
        for _, table in it:
            self.accumulate(table, feature_prefix, num_features, progress=False)
        return self.finalize()

    # -- accumulation --------------------------------------------------------

    def start(self, num_features):
        """Reset the accumulators for a fresh pass over `num_features` features."""
        self._n_active = np.zeros(num_features, dtype=np.float64)
        self._n_obs = 0
        self._max = np.zeros(num_features, dtype=np.float64)
        self._is_fitted = False
        return self

    def accumulate(self, adata, feature_prefix, num_features, progress=True):
        """Fold one patch table into the running statistics."""
        feature_names = [f"{feature_prefix}_{i}" for i in range(num_features)]
        X_csc = adata[:, feature_names].X.tocsc()
        n = X_csc.shape[0]
        if n == 0:
            return self

        s = max(1, n // self.n_chunks)
        starts = range(0, n, s)
        it = tqdm(starts, desc="Computing activation stats") if progress else starts

        for start in it:
            Xc = X_csc[start:start + s]
            # Saving and reloading sparse data can add a leading axis; tolerate
            # both shapes rather than failing the whole pass.
            try:
                counts = np.array((Xc > 0).sum(axis=0))[0]
                mx = np.array(Xc.max(axis=0).toarray())[0]
            except IndexError:
                counts = np.array((Xc > 0).sum(axis=0)).ravel()
                mx = np.array(Xc.max(axis=0).toarray()).ravel()
            self._n_active += counts
            self._max = np.maximum(self._max, mx)

        self._n_obs += n
        return self

    def finalize(self):
        """Turn the accumulators into pct_active_ / max_score_."""
        if self._n_obs == 0:
            raise ValueError("No patches were accumulated.")
        # Fraction over the true total, rather than a mean of per-chunk means:
        # the latter is only correct when every chunk is the same size.
        self.pct_active_ = self._n_active / self._n_obs
        self.max_score_ = self._max
        self._is_fitted = True
        return self

    def plot_feature_selection(self):
        """
        Plot joint distribution of activation frequency vs peak score,
        with threshold reference lines and selection highlighted.
        """
        self._check_fitted()

        mask_nonzero = (self.pct_active_ > 0) & (self.max_score_ > 0)
        pct = self.pct_active_[mask_nonzero]
        mx = self.max_score_[mask_nonzero]
        selected = (pct > self.pct_threshold) & (mx > self.max_score_threshold)

        import pandas as pd
        data = pd.DataFrame({
            'pct_patches_active': pct,
            'max_feature_score': mx,
            'selected': selected
        })

        g = sns.JointGrid(data=data, 
                          x='pct_patches_active', 
                          y='max_feature_score')
        g.ax_joint.set_xscale('log')
        g.ax_joint.set_yscale('log')
        g.ax_joint.set_xlabel('Fraction of patches active')
        g.ax_joint.set_ylabel('Max feature score')
        
        g.plot_joint(
            sns.scatterplot, 
            hue=data['selected'],   # Pass hue here instead
            s=10, 
            linewidth=0, 
            legend=False, 
            # rasterized=True
        )

        # sns.histplot(data=data, x='pct_patches_active', hue='selected', ax=g.ax_marg_x, bins=50, legend=False, element="step")
        # sns.histplot(data=data, y='max_feature_score', hue='selected', ax=g.ax_marg_y, bins=50, legend=False, element="step")

        # g.ax_marg_x.set_yscale('log')
        # g.ax_marg_x.set_ylim(bottom=1)
        # g.ax_marg_y.set_xscale('log')
        # g.ax_marg_x.set_xlim(left=1)
        g.refline(x=self.pct_threshold, y=self.max_score_threshold)

        n_selected = selected.sum()
        g.ax_joint.set_title(f'{n_selected} features selected', pad=10)

        return g

    def get_selected_indices(self):
        """
        Return integer indices of features passing both thresholds.

        Returns
        -------
        np.ndarray of int
            Indices into the original feature array.
        """
        self._check_fitted()
        return np.where(
            (self.pct_active_ > self.pct_threshold) &
            (self.max_score_ > self.max_score_threshold)
        )[0]

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                "Call compute_activation_stats() before using this method."
            )