"""Backward-compat shim -- moved to mesoslide.tools.sae._autoencoder.

Kept importable at this path so `pickle`/`joblib.load` can still resolve
SparseAutoencoder instances saved under this module's old qualified name.
"""
from mesoslide.tools.sae._autoencoder import (
    SimpleAutoencoder, train_simple_sae, SparseAutoencoder,
)
