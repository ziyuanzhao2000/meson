from matplotlib.colors import LinearSegmentedColormap, Normalize
import matplotlib.pyplot as plt
import spatialdata

def plot_feature_map(
    sdata,
    image_name,
    feature_name,
    bbox_postfix='_grid_point_bbox',
    patch_postfix='_grid_point_patch',
    cmap=None,
    fill_alpha=0.7,
    figsize=(10, 10),
    colorbar=False,
    title=None,
    norm=None,
    method='datashader',
    datashader_reduction='max',
    return_ax=False
):
    """
    Plot a feature map overlay on a spatial image.
    
    Parameters
    ----------
    sdata : spatialdata.SpatialData
        SpatialData object containing the image and annotations
    image_name : str
        Name of the image element to plot
    feature_name : str
        Name of the feature column in patch observations to visualize
    bbox_postfix : str, optional
        Postfix for the bounding box element name (default: '_grid_point_bbox')
    patch_postfix : str, optional
        Postfix for the patch table element name (default: '_grid_point_patch')
    cmap : matplotlib colormap, optional
        Colormap to use for the feature. If None, uses transparent to green colormap
    fill_alpha : float, optional
        Alpha value for the overlay (default: 0.7)
    figsize : tuple, optional
        Figure size (default: (10, 10))
    colorbar : bool, optional
        Whether to show colorbar (default: False)
    title : str, optional
        Plot title. If None, uses default title format
    norm : matplotlib Normalize, optional
        Normalization for image rendering. If None, uses Normalize(vmin=0, vmax=255)
    method : str, optional
        Rendering method (default: 'datashader')
    datashader_reduction : str, optional
        Datashader reduction method (default: 'max')
    return_ax : bool, optional
        Whether to return the axes object (default: False)
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object
    ax : matplotlib.axes.Axes
        Axes object (only if return_ax=True)
    """
    # Create mini SpatialData with only necessary elements
    sdata_mini = spatialdata.SpatialData()
    sdata_mini[image_name] = sdata[image_name]
    sdata_mini[f'{image_name}{bbox_postfix}'] = sdata[f'{image_name}{bbox_postfix}']
    sdata_mini[f'{image_name}{patch_postfix}'] = sdata[f'{image_name}{patch_postfix}']
    
    # Set default colormap if not provided
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list(
            'transparent_to_green', 
            [(1, 1, 1, 0), (0, 1, 0, 0.5)], 
            N=256
        )
    
    # Set default normalization if not provided
    if norm is None:
        norm = Normalize(vmin=0, vmax=255)
    
    # Set default title if not provided
    if title is None:
        title = f'Image {image_name}'
    
    # Create figure and plot
    fig, ax = plt.subplots(figsize=figsize)
    sdata_mini.pl.render_images(element=image_name, norm=norm)\
        .pl.render_shapes(
            element=f'{image_name}{bbox_postfix}',
            color=feature_name,
            cmap=cmap,
            fill_alpha=fill_alpha,
            method=method,
            datashader_reduction=datashader_reduction
        )\
        .pl.show(image_name, title=title, colorbar=colorbar, ax=ax)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    
    if return_ax:
        return fig, ax
    return fig