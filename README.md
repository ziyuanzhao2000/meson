# Compositional and interpretable representation of histology using AI foundation models and sparse autoencoders

Table of Contents
------------------
* General Information
  * Associated Publication
  * Recommended Citation
  * Useful Links
* Applying the FM-SAE workflow
  * Installation
  * WSI Preprocessing and extract FM patch embeddings
  * Extract Sparse Autoencoder (SAE) embeddings
  * Plotting feature maps and representative patches
  * Training a new SAE from FM patch embeddings
* Additional notes / comments

------------------ 
General Information
------------------
### Compositional and interpretable representation of histology using AI foundation models and sparse autoencoders <Publication or Dataset Title>   
**Authors:** Ziyuan Zhao*, Zoltan Maliga*, Emmanuel C. Ogbonna, Soheil R. Talemi, Shannon Coy, Andréanne Gagné, Kapongo Lumamba, Isaac H. Solomon, Angela Shih, Sandro Santagata, Adrie J.C. Steyn, Threnesan Naidoo†, Peter K. Sorger† 
  
*Co-first Authors: Z.Z., Z.M.

†Co-Senior Authors: T.N., P.K.S.  
​  
**Please cite this data as the following:**      
Zhao, Z. et al. (2026). Compositional and interpretable representation of histology using AI foundation models and sparse autoencoders. {journal/biorxv}    

**Relevant links:** <remove links that are not relevant>  
> * Publication DOI: [doi.org/MY-PAPER-DOI](https://doi.org/MY-PAPER-DOI-URL) 
> * Associated GitHub Repository: [MY-REPO](https://github.com/labsyspharm/2025-Vallius-Shi-Novikov-melanoma-PCAII)  
> * To view an archived record of this repository: [My-ZENODO-DOI](https://zenodo.org/doi/MY-ZENODO-DOI-URL) 
> * To view the image data online, visit: [My-ATLAS-PAGE](https://tissue-atlas.org/MY-ATLAS-PAGE-URL)
​
**Licenses/restrictions placed on the data:** CC-BY [creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/)


------------------ 
Applying the FM-SAE workflow
------------------

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

------------------ 
Additional notes / comments
------------------