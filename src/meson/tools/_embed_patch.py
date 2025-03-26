from spatialdata import SpatialData
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from .._readwrite import get_base_level
from .._utils import get_optimal_chunk_size
from tqdm import tqdm

def _get_embedder(embedder_name: str):
    """Convert embedder name to embedder instance"""
    embedder_name = embedder_name.lower()
    if embedder_name == "test":
        from .embedders import TestEmbedder
        return TestEmbedder()
    elif embedder_name in ["uni", "UNI"]:
        from .embedders import UNIEmbedder
        return UNIEmbedder()
    elif embedder_name in ["uni2", "UNI2", "uni2-h", "UNI2-h"]:
        from .embedders import UNI2Embedder
        return UNI2Embedder()
    else:
        raise ValueError(f"Unknown embedder: {embedder_name}")

class PatchDataset(Dataset):
    def __init__(self, sdata: SpatialData, 
                        image_name: str | None = None,
                        point_name: str | None = 'grid_point',
                        patch_name: str | None = 'patch'):
        if image_name is not None:
            patch_name = f'{image_name}_{point_name}_{patch_name}'
        image = get_base_level(sdata[image_name])
        self.image = image.chunk(chunks=get_optimal_chunk_size(image))
        print(self.image.chunksizes)
        self.patch_df = sdata[patch_name].obs

    def __getitem__(self, index):
        row = self.patch_df.iloc[index]
        patch = self.image[
            :,  # All channels
            row['ymin']:row['ymax'],
            row['xmin']:row['xmax']
        ]
        return patch.compute().data
    
    def __len__(self):
        return len(self.patch_df)
    

def embed_patch(sdata: SpatialData,
                embedder,
                *,
                image_name: str | None = None,
                point_name: str | None = 'grid_point',
                patch_name: str | None = 'patch',
                batch_size: int = 32, 
                num_workers: int = 1,
                device: str = 'cpu'):
    if isinstance(embedder, str):
        embedder = _get_embedder(embedder)
    embedder.model.to(device)
    # patch_array = sdata.tables[patch_name].obsm['patch']
    # patch_dataset = TensorDataset(torch.tensor(patch_array.compute()))
    patch_dataset = PatchDataset(sdata, image_name, point_name, patch_name)
    embedding_array = np.zeros((len(patch_dataset), embedder.embed_dim))
    patch_dataloader = DataLoader(patch_dataset, 
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=False)
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(patch_dataloader)):
            batch_embedding = embedder(batch.to(device)).cpu().numpy()
            embedding_array[idx*batch_size:idx*batch_size+len(batch_embedding), :] = batch_embedding
    if image_name is not None:
        patch_name = f'{image_name}_{point_name}_{patch_name}'
    sdata.tables[patch_name].obsm[f'{embedder.name}_embedding'] = embedding_array
    return sdata
