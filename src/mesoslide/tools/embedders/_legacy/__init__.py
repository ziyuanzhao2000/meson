"""Deprecated model-manager embedder classes.

Superseded by `lazyslide_models.MODEL_REGISTRY` for vision foundation models,
consumed via `mesoslide.tools._embed_patch.embed_patch`. Kept here only for
existing callers (`mesoslide.tools._legacy._embed_patch`,
`mesoslide.tools.segmenters.TokenClusterizer`, `mesoslide.scripts.*`) that
still depend on the old `sdata`-table-based pipeline.
"""

import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        'test': ['TestEmbedder'],
        'UNI': ['UNIEmbedder'],
        'UNI2': ['UNI2Embedder'],
        'Virchow2': ['Virchow2Embedder'],
        'SAE': ['SparseAutoencoder'],
        'FrequencyRankedKMeans': ['FrequencyRankedKMeans'],
    }
)
