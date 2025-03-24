### Installation
First clone the repo and also [UNI](https://github.com/mahmoodlab/UNI)'s repo Then execute the following scripts
```bash
conda create -n meson python=3.10 -y
conda activate meson
```

Go to the UNI repo and install it,
```bash
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

Finally, go back to this repo, install openslide from conda-forge (this is because building the dependencies is a pain in the ass)
and tifffile from pip (following their official installation guide)
```bash
conda install openslide
python -m pip install -U tifffile[all]
```

then install the package with `pip install -e .`

The Jupyter notebook `example.ipynb` (WIP) shows how to load WSI, get the tissue mask, 
apply UNI foundation model to tiled patches, cluster the patch level embeddings, 
and then visualize the results. For now you can check out `scripts/analysis_TB_3D_HnE.py` for an example.

