from mesoslide._interpolation import interpolate_multiclass  # noqa: F401  (re-exported by mesoslide.plotting)


def extract_samples(df, column, patch_size = 448):
    samples = {}
    for xmin in df.xmin.unique():
        for ymin in df.ymin.unique():
            xcenter = int(xmin + patch_size // 2)
            ycenter = int(ymin + patch_size // 2)
            samples[(xcenter, ycenter)] = -1 # initialize all samples as background
    for row in df.itertuples():
        xmin, xmax, ymin, ymax = row.xmin, row.xmax, row.ymin, row.ymax
        xcenter = int(xmin + patch_size // 2)
        ycenter = int(ymin + patch_size // 2)
        samples[(xcenter, ycenter)] = getattr(row, column)
        
    return samples