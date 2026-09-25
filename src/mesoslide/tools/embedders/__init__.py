"""Vision foundation-model embedding.

Model instantiation is now delegated to `lazyslide_models.MODEL_REGISTRY`
(see `mesoslide.tools._feature_extraction.feature_extraction`) rather than
maintained here.
The classes previously defined in this package (`UNIEmbedder`,
`UNI2Embedder`, `Virchow2Embedder`, `TestEmbedder`, `SparseAutoencoder`,
`FrequencyRankedKMeans`) have moved to `mesoslide.tools.embedders._legacy`.
The current sparse-coding models (`SparseAutoencoder`, `LocalityConstrainedCoding`,
`MiniBatchDictionaryCoding`) live in `mesoslide.tools.sparse_coding`.
"""
