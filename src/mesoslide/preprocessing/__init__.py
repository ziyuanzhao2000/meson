import lazy_loader as lazy

# make_bbox / make_grid / make_patch / segment_tissue were deprecated when their work
# moved to ezslide and lazyslide (zs.pp.find_tissues, zs.pp.tile_tissues). Their modules
# now live in _legacy/ and are no longer exported.
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
    '_extract_patches': ['extract_patches'],
    '_extract_saliency_maps': ['extract_saliency_maps'],
    '_registration': ['ForwardTransform'],
    }
)
