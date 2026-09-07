"""Vision foundation-model embedding.

Model instantiation is now delegated to `lazyslide_models.MODEL_REGISTRY`
(see `mesoslide.tools._embed_patch.embed_patch`) rather than maintained here.
The classes previously defined in this package (`UNIEmbedder`,
`UNI2Embedder`, `Virchow2Embedder`, `TestEmbedder`, `SparseAutoencoder`,
`FrequencyRankedKMeans`) have moved to `mesoslide.tools.embedders._legacy`.
"""
