import lazy_loader as lazy
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
    'UNet': ['GenericSegmenter'],
    'TokenClusterizer': ['TokenClusterizer'],
    '_adaptive_sampling': ['adaptive_sample_wsi', 'adaptive_refine_step']
    }
)
