from spatialdata import SpatialData
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

def _get_embedder(embedder_name: str):
    """Convert embedder name to embedder instance"""
    embedder_name = embedder_name.lower()
    if embedder_name == "test":
        from .embedders.test import TestEmbedder
        return TestEmbedder()
    elif embedder_name == "uni":
        from .embedders.UNI import UNIEmbedder
        return UNIEmbedder()
    else:
        raise ValueError(f"Unknown embedder: {embedder_name}")
    
def embed_patch(sdata: SpatialData,
                embedder,
                *,
                image_name: str | None = None,
                point_name: str | None = 'grid_point',
                patch_name: str | None = 'patch',
                batch_size: int = 32):
    if image_name is not None:
        patch_name = f'{image_name}_{point_name}_{patch_name}'
    if isinstance(embedder, str):
        embedder = _get_embedder(embedder)
    patch_array = sdata.tables[patch_name].obsm['patch']
    patch_dataset = TensorDataset(torch.tensor(patch_array.compute()))
    embedding_array = np.zeros((len(patch_array), embedder.embed_dim))
    
    with torch.no_grad():
        for idx, batch in enumerate(DataLoader(patch_dataset, 
                                batch_size=batch_size,
                                shuffle=False)):
            batch_embedding = embedder(batch[0]).cpu().numpy()
            embedding_array[idx*batch_size:idx*batch_size+len(batch_embedding), :] = batch_embedding
    
    sdata.tables[patch_name].obsm[f'{embedder.name}_embedding'] = embedding_array
    return sdata
