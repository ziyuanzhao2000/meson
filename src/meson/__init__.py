import lazy_loader as lazy
import importlib

__getattr__, __dir__, _ = lazy.attach(__name__, 
    submodules = ['preprocessing', 'tools', 'plotting'],
    submod_attrs = {
        '_readwrite': ['SpatialData', 'export_patch', 'read_zarr', 'overwrite_element'],
        '_wsi': ['add_wsi', 'read_wsi'],
        '_settings': ['settings'],
        '_utils': ['csv2mask']
    }
)

pp = importlib.import_module(f"{__name__}.preprocessing")
tl = importlib.import_module(f"{__name__}.tools")

