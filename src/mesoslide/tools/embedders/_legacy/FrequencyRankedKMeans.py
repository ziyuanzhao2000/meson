"""Re-export kept so that models pickled under this path still load.

`FrequencyRankedKMeans` now lives in `mesoslide.tools.sparse_coding`.
"""
from mesoslide.tools.sparse_coding._frequency_ranked_kmeans import FrequencyRankedKMeans

__all__ = ["FrequencyRankedKMeans"]
