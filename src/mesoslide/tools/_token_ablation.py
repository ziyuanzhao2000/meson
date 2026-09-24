"""Token ablation: how much a patch-level score depends on each group of tokens.

For each patch and each token group (e.g. a `TokenClusterer` cluster or a
tissue-class label), the vision model is re-run with that group's patch tokens
removed, and, as a size-matched control, with the same number of randomly
chosen tokens removed. The resulting patch embeddings are scored by any
callable (e.g. `mesoslide.tools.sparse_coding.feature_scorer`). A group's
`excess_drop` is how much more the score falls when that group is removed
than when random tokens are removed.
"""

from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from scipy.stats import wilcoxon
from tqdm.auto import tqdm

from ._model_stage import ImageModelStage


def token_ablation(
    images,
    token_labels,
    model,
    scorer: Callable[[np.ndarray], np.ndarray],
    *,
    full_scores=None,
    device: Optional[str] = None,
    batch_size: int = 64,
    seed: int = 0,
    progress_bar: bool = True,
) -> pd.DataFrame:
    """
    Score drop from removing each token group of each patch, with random-token controls.

    Parameters
    ----------
    images : array-like of shape (n_patches, C, H, W)
        Patch pixels (uint8), e.g. from :func:`mesoslide.preprocessing.extract_patch_images`.
    token_labels : array-like of int, shape (n_patches, n_tokens) or (n_patches, gh, gw)
        Group label of every patch token, row-major over the model's token grid.
    model : str or lazyslide_models.ImageModel
        ViT-style vision model with a timm VisionTransformer backbone.
    scorer : callable
        Maps pooled patch embeddings, shape (n, D), to one score per patch,
        shape (n,). See :func:`mesoslide.tools.sparse_coding.feature_scorer`.
    full_scores : array-like of shape (n_patches,), optional
        Unablated patch scores (e.g. `patches.obs['_feature_score']`), added as a
        `full_score` column for reporting; not needed for `excess_drop`.
    device : str, optional
        Torch device for the vision model. Defaults to "cuda" if available.
    batch_size : int, default=64
    seed : int, default=0
        Seed for the random-token controls.
    progress_bar : bool, default=True

    Returns
    -------
    pd.DataFrame
        One row per (patch, label present in the patch) with columns `patch`,
        `label`, `n_tokens` (removed), `token_fraction`, `score_label_removed`,
        `score_random_removed` and `excess_drop` (= `score_random_removed` -
        `score_label_removed`), plus `full_score` when given. Labels covering
        every token of a patch are skipped (nothing would remain).
    """
    if isinstance(images, list):
        raise ValueError("images must share one shape; got a list of differently sized patches.")
    labels = np.asarray(token_labels).reshape(len(images), -1)
    n_tokens = labels.shape[1]
    rng = np.random.default_rng(seed)

    # Two jobs per (patch, label): remove the label's tokens, remove as many random tokens.
    rows, jobs = [], []
    for p, patch_labels in enumerate(labels):
        for label in np.unique(patch_labels):
            removed = patch_labels == label
            n_removed = int(removed.sum())
            if n_removed == n_tokens:
                continue
            random_removed = np.zeros(n_tokens, dtype=bool)
            random_removed[rng.choice(n_tokens, n_removed, replace=False)] = True
            rows.append(dict(patch=p, label=int(label), n_tokens=n_removed,
                             token_fraction=n_removed / n_tokens))
            jobs += [(p, np.flatnonzero(~removed)), (p, np.flatnonzero(~random_removed))]

    stage = ImageModelStage(model, dense=False, device=device)
    pooled = [None] * len(jobs)
    # Jobs with the same number of kept tokens can share a batch.
    by_size = {}
    for j, (_, keep) in enumerate(jobs):
        by_size.setdefault(len(keep), []).append(j)
    batches = [ids[i:i + batch_size] for ids in by_size.values() for i in range(0, len(ids), batch_size)]
    for ids in tqdm(batches, desc="Token ablation", disable=not progress_bar):
        image_batch = torch.as_tensor(np.stack([np.asarray(images[jobs[j][0]]) for j in ids]))
        keep_idx = np.stack([jobs[j][1] for j in ids])
        out = stage.encode_kept_tokens(image_batch, keep_idx).float().cpu().numpy()
        for j, embedding in zip(ids, out):
            pooled[j] = embedding

    scores = np.asarray(scorer(np.stack(pooled)), dtype=np.float64).reshape(-1)
    if len(scores) != len(jobs):
        raise ValueError(f"scorer returned {len(scores)} scores for {len(jobs)} embeddings")

    result = pd.DataFrame(rows)
    result["score_label_removed"] = scores[0::2]
    result["score_random_removed"] = scores[1::2]
    result["excess_drop"] = result["score_random_removed"] - result["score_label_removed"]
    if full_scores is not None:
        result["full_score"] = np.asarray(full_scores, dtype=np.float64)[result["patch"].to_numpy()]
    return result


def summarize_token_ablation(results: pd.DataFrame) -> pd.DataFrame:
    """
    Per label: patches containing it, mean token fraction, mean scores, mean
    `excess_drop`, and a Wilcoxon signed-rank p-value for `excess_drop` != 0.

    Parameters
    ----------
    results : pd.DataFrame
        Output of :func:`token_ablation`.

    Returns
    -------
    pd.DataFrame indexed by `label`, sorted by mean `excess_drop` (descending).
    """
    columns = ["token_fraction", "score_label_removed", "score_random_removed", "excess_drop"]
    if "full_score" in results:
        columns.insert(1, "full_score")
    grouped = results.groupby("label")
    summary = grouped[columns].mean()
    summary.insert(0, "n_patches", grouped.size())
    summary["wilcoxon_p"] = grouped["excess_drop"].apply(
        lambda d: wilcoxon(d).pvalue if len(d) > 1 and np.any(d != 0) else np.nan
    )
    return summary.sort_values("excess_drop", ascending=False)
