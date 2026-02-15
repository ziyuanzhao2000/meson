import lazy_loader as lazy
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
    '_make_bbox': ['make_bbox'],
    '_make_grid': ['make_grid'],
    '_make_patch': ['make_patch'],
    '_segment_tissue': ['segment_tissue'],
    '_registration': ['ForwardTransform'],
    }
)