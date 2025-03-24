import openslide

from ._settings import settings

img_exts = {
    "tif",
    "tiff",
    "ome.tif",
    "ome.tiff",
    "zarr"
}

def read(

    ):
    filekey = str(filename)
    filename = settings.writedir / (filekey + "." + settings.file_format_data)
    if not filename.exists():
        msg = (
            f"Reading with filekey {filekey!r} failed, "
            f"the inferred filename {filename!r} does not exist. "
            "If you intended to provide a filename, either use a filename "
            f"ending on one of the available extensions {img_exts} "
            "or pass the parameter `ext`."
        )
        raise ValueError(msg)

def read_HnE():
    pass

def read_mIF():
    pass