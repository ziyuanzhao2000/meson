from spatialdata import SpatialData
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, Dataset
from meson._readwrite import get_base_level
from meson._utils import get_optimal_chunk_size, overwrite_element
from tqdm import tqdm
import joblib
from scipy.sparse import csc_array, hstack
import pandas as pd

def _get_embedder(sdata, embedder_name: str, token=None):
    """Convert embedder name to embedder instance"""
    if embedder_name == "test":
        from meson.tools.embedders import TestEmbedder
        return TestEmbedder(), "Vision"
    elif embedder_name in ["uni", "UNI"]:
        from meson.tools.embedders import UNIEmbedder
        return UNIEmbedder(token=token), "Vision"
    elif embedder_name in ["uni2", "UNI2", "uni2-h", "UNI2-h"]:
        from meson.tools.embedders import UNI2Embedder
        return UNI2Embedder(token=token), "Vision"
    else:
        for model_info in sdata.attrs['models_metadata']:
            model_name = model_info['name']
            if embedder_name == model_name:
                return sdata.attrs['models'][model_name], model_info['model_type']
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
                token: str | None = None,
                obsm_key: str | None = None,
                device: str = 'cpu',
                save: bool = True,
                overwrite: bool = False):
    if image_name is not None:
        patch_name_full = f'{image_name}_{point_name}_{patch_name}'
    else:
        patch_name_full = patch_name
    patch_table = sdata.tables[patch_name_full]
    if isinstance(embedder, str):
        embedder_name = embedder
        embedder, model_type = _get_embedder(sdata, embedder, token=token)
    else:
        embedder_name = embedder.name
    
    if model_type == "Vision":
        embedding_name = f'{embedder.name}_embedding'
        if overwrite is False and embedding_name in patch_table.obsm:
            return sdata
        if hasattr(embedder, 'model'):
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
        patch_table.obsm[embedding_name] = embedding_array
    elif model_type == "SAE":
        sae_embedding = embedder.transform(patch_table.obsm[obsm_key])
        new_vars = [f'{embedder_name}_{i}' for i in range(sae_embedding.shape[1])]
        existing_vars = list(patch_table.var.index)
        # slice columns of the table to remove these columns that are redundant with embedding
        diff_vars = [item for item in existing_vars if item not in new_vars]
        if len(diff_vars) == 0 and (not overwrite):
            return sdata
        if len(diff_vars) < len(existing_vars) or overwrite:
            patch_table = patch_table[:, diff_vars].copy() # may be slow with large table?
        print(patch_table.X.shape)
        print(patch_table.var.index, pd.Index(diff_vars + new_vars))
        patch_table.var.index = pd.Index(diff_vars + new_vars)
        patch_table.X = hstack((csc_array(patch_table.X), sae_embedding.tocsc())).tocsr()
    
    if save:
        overwrite_element(sdata, patch_name_full)
    return sdata
