import lazy_loader as lazy
import importlib

__getattr__, __dir__, _ = lazy.attach(__name__, 
    submodules = ['preprocessing', 'tools'],
    submod_attrs = {
        '_readwrite': ['SpatialData', 'export_patch, read_zarr'],
        '_wsi': ['add_wsi', 'read_wsi'],
        '_settings': ['settings']
    }
)

pp = importlib.import_module(f"{__name__}.preprocessing")
tl = importlib.import_module(f"{__name__}.tools")

