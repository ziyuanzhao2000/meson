# /n/scratch/users/z/ziz531/meson/src/meson/plotting/__init__.py

import lazy_loader as lazy
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        '_utils': ['make_transparent_to_color_colormaps', 'get_transparent_colormap'],
        '_feature_reports': ['plot_feature_spatial_distribution']
    }
)