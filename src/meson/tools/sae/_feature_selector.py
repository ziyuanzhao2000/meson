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

    def compute_activation_stats(self, adata, feature_prefix, num_features):
        """
        Compute per-feature activation frequency and peak score.
        This is the expensive step — only needs to be run once.

        Parameters
        ----------
        adata : AnnData
            Patch-level AnnData with sparse SAE embeddings in .X
        feature_prefix : str
            Prefix of feature columns, e.g. 'UNI_SAE'
        num_features : int
            Total number of SAE features
        """
        feature_names = [f"{feature_prefix}_{i}" for i in range(num_features)]
        X_csc = adata[:, feature_names].X.tocsc()

        n = X_csc.shape[0]
        s = n // self.n_chunks
        pct_chunks, max_chunks = [], []

        for i in tqdm(range(self.n_chunks), desc="Computing activation stats"):
            start = i * s
            end = (i + 1) * s if i < self.n_chunks - 1 else n
            Xc = X_csc[start:end]
            pct_chunks.append(np.array((Xc > 0).mean(axis=0))[0])
            max_chunks.append(np.array(Xc.max(axis=0).toarray())[0])

        self.pct_active_ = np.stack(pct_chunks).mean(axis=0)
        self.max_score_ = np.stack(max_chunks).max(axis=0)
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

        g = sns.JointGrid(data=data, x='pct_patches_active', y='max_feature_score', hue='selected')
        g.ax_joint.set_xscale('log')
        g.ax_joint.set_yscale('log')
        g.ax_joint.set_xlabel('Fraction of patches active')
        g.ax_joint.set_ylabel('Max feature score')
        g.plot_joint(sns.scatterplot, s=10, linewidth=0, legend=False)
        g.plot_marginals(sns.histplot, bins=50)
        g.ax_marg_x.set_yscale('log')
        g.ax_marg_y.set_xscale('log')
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