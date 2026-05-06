import lazy_loader as lazy
__getattr__, __dir__, __all__ = lazy.attach(
    __name__,
    submod_attrs={
        '_utils': [
            'make_transparent_to_color_colormaps',
            'get_transparent_colormap',
            'resize_image_to_fit',
            'FLUOROPHORE_COLORS',
            'FLUOROPHORE_CMAPS',
            'MARKER_COLOR_DEFAULTS',
            'get_marker_colormap',
        ],
        '_feature_reports': ['plot_feature_spatial_distribution', 'create_feature_pdf'],
        '_patch_gallery': [
            'plot_patch_gallery',
            'plot_patch_gallery_with_saliency',
            'plot_feature_gallery',
            'extract_patch_images'
        ],
        '_image_grid': ['_plot_image_grid'],
        '_clustered_heatmap': ['plot_clustered_heatmap'],
        '_rasterization': ['interpolate_multiclass', 'extract_samples'],
        '_feature_map': ['plot_feature_map'],
        '_assemble_pyramid': ['assemble_pyramid'],
    }
)