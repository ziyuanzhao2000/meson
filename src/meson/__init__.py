import lazy_loader as lazy
import importlib

__getattr__, __dir__, _ = lazy.attach(__name__, 
    submodules = ['preprocessing', 'tools', 'plotting'],
    submod_attrs = {
        '_readwrite': ['SpatialData', 'export_patch', 'read_zarr', 'overwrite_element'],
        '_wsi': ['add_wsi', 'read_wsi'],
        '_settings': ['settings'],
        '_interpolation': ['interpolate_edt', 'interpolate_multiclass', 'interpolate_linear'],
        '_utils': ['csv2mask', 
                   'select_random_patches', 'select_patches_for_binary_feature',    
                   'select_top_patches', 'select_negative_patches',
                   'get_patch_scores', 'copy_feature_score_to_obs']
    }
)

pp = importlib.import_module(f"{__name__}.preprocessing")
tl = importlib.import_module(f"{__name__}.tools")
