from typing import TYPE_CHECKING

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from meson._readwrite import get_base_level, overwrite_element
from meson._utils import get_optimal_chunk_size
from tqdm import tqdm
from scipy.sparse import csc_array, hstack
import pandas as pd
from PIL import Image


if TYPE_CHECKING:
    from spatialdata import SpatialData

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
    elif embedder_name in ["kmeans", "KMeans", "FrequencyRankedKMeans"]:
        from meson.tools.embedders import FrequencyRankedKMeans
        for model_info in sdata.attrs['models_metadata']:
            model_name = model_info['name']
            if embedder_name == model_name:
                return sdata.attrs['models'][model_name], "KMeans"
        raise ValueError(f"KMeans model '{embedder_name}' not found in sdata.attrs['models']")
    else:
        for model_info in sdata.attrs['models_metadata']:
            model_name = model_info['name']
            if embedder_name == model_name:
                return sdata.attrs['models'][model_name], model_info['model_type']
        raise ValueError(f"Unknown embedder: {embedder_name}")

class PatchDataset(Dataset):
    def __init__(self, sdata: "SpatialData", 
                        image_name: str | None = None,
                        point_name: str | None = 'grid_point',
                        patch_name: str | None = 'patch', 
                        precache: bool = False,
                        color_transform = None):
        self.precache = precache
        if image_name is not None:
            patch_name = f'{image_name}_{point_name}_{patch_name}'
        image = get_base_level(sdata[image_name])
        if self.precache:
            self.image = image.compute().data
        else:
            self.image = image.chunk(chunks=get_optimal_chunk_size(image))
        print("Initialized patch dataset for image with shape", self.image.shape)
        self.patch_df = sdata[patch_name].obs

        self.color_transform = color_transform

    def __getitem__(self, index):
        row = self.patch_df.iloc[index]
        patch = self.image[
            :,  # All channels
            row['ymin']:row['ymax'],
            row['xmin']:row['xmax']
        ]
        if not self.precache:
            patch = patch.compute().data
    
        if not self.color_transform is None:
            patch = Image.fromarray(patch.transpose(1,2,0))
            return np.array(self.color_transform(patch)).transpose(2, 0, 1)
        else:
            return patch
    def __len__(self):
        return len(self.patch_df)
    

def embed_patch(sdata: "SpatialData",
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
                overwrite: bool = False,
                precache: bool = False, 
                color_transform = None,
                column_keep_indices: list[int] | None = None):
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
        patch_dataset = PatchDataset(sdata, image_name, point_name, patch_name, 
                                    precache=precache,
                                    color_transform=color_transform)
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
        if precache:
            del patch_dataset.image
            del patch_dataset
            del patch_dataloader
    elif model_type == "SAE":
        sae_embedding = embedder.transform(patch_table.obsm[obsm_key], device=device, 
                                           column_keep_indices=column_keep_indices)
        if column_keep_indices is not None:
            new_vars = [f'{embedder_name}_{i}' for i in column_keep_indices]
        else:
            new_vars = [f'{embedder_name}_{i}' for i in range(sae_embedding.shape[1])]
        existing_vars = list(patch_table.var.index)
        # slice columns of the table to remove these columns that are redundant with embedding
        diff_vars = [item for item in existing_vars if item not in new_vars]
        if len(diff_vars) == 0 and (not overwrite):
            return sdata
        if len(diff_vars) < len(existing_vars) or overwrite:
            patch_table = patch_table[:, diff_vars].copy() # may be slow with large table?
        patch_table.var.index = pd.Index(diff_vars + new_vars)
        patch_table.X = hstack((csc_array(patch_table.X), sae_embedding.tocsc())).tocsr()
        sdata[patch_name_full] = patch_table
    elif model_type == "KMeans":
        base_embeddings = patch_table.obsm[obsm_key]
        kmeans_labels = embedder.transform(base_embeddings)
        label_col_name = f'{embedder_name}_label'
        patch_table.obs[label_col_name] = kmeans_labels
        for i in range(embedder.n_clusters):
            patch_table.obs[f'{embedder_name}_label_{i}'] = (kmeans_labels == i).astype(int)
        
    if save:
        overwrite_element(sdata, patch_name_full)
    return sdata
