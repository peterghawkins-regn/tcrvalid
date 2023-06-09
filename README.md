TCR-VALID: TCR Variational Autoencoder Landscape for Interpretable Dimensions
===============================================================================

This is a package for building capacity controlled VAEs for TCR sequence data.

Inside there are tools to train representation models of TCR sequences and investigate their smoothness, disentanglement and ability to cluster and classify TCRs.


Installation
-------------

We advise initially creating a conda environment via
 - conda env create -n tcrvalid python=3.8 pip

And subsequently we can install TCRVALID via pip pointing at the directory where you cloned the repo:
 - pip install /path/to/tcrvalid/

In case there are any issues in versioning we additionally provide:
- full_databricks_requirements.txt: used to pip install reqs to reinstate databricks runtime env used during our model training
 - minimal\_requirements.txt: installs the key packages
 
 
Models and data
----------------

All required models are co-packaged with TCR-VALID, and can be loaded by name with a loading utility function.

The following will load a single named model for TRB CDR2-CDR3 sequences:
```
from tcrvalid.load_models import *
model_name = '1_2'
loaded_trb_models = load_named_models(model_name,chain='TRB',as_keras=from_keras)[model_name]
```

where for TRB available model names are:

'0_0', '1_1', '1_2', '1_5', '1_10', '1_20', '1_2_full_40'

The first digit represent the value of $\beta$ used in the KL-loss term of the model, and the second the capacity of the model. 

The model with 'full' in the name is the final TRB model trained on the larges TRB training dataset, which was collected at the 40th epoch at the minima of the validation loss. All other models were trained on the "smallTRB" dataset of approx 4 million sequences.

For TRA we provide the TCR-VALID model with capacity=2: '1_2'.

Multiple models can be collected into a python dictionary for looping through them via:
```
from tcrvalid.load_models import *
#  model_names is a list of model names to collect
model_names = ['1_2','1_5']
loaded_trb_models = load_named_models(model_names,chain='TRB',as_keras=from_keras)
```
 
#### *data*

We also co-package esssential datasets for reproducing our findings. Namely:
 - antigen-labeled TCR dataset
 - smallTRB test dataset and TRA test dataset
     - due to the data volume we do not provide the full datasets
 - Full datasets can be collated using the trb_repertoires.csv and tra_repertoires.csv
     - these give the repertoire_ids from iRecetor and VDJServer when compiling the datasets
 - TR\{A/B\}\_reference.csv:
     - two files of CDR1 and CDR2 sequences associated with each V gene.
     - collected using IMGT reference - IMGT &reg; the international ImMunoGeneTics information system &reg;, http\:\/\/www\.imgt\.org. \[Nucleic Acids Res., 43, D413\-D422 (2015)\].Terms of use [here](https://www.imgt.org/about/termsofuse.php).

Some of the data is stored in "results_data"

 - this is data used in some notebooks for ease of generating certain plots, but is not co-packaged with TCR-VALID is it not required for general use.

Examples
---------

#### *notebooks:*

In the /notebook directory you will find several examples and plotting methods:

 - cluster_tcrs_example.ipynb
     - example of how to cluster
 - hinton_feature_iportance_example.ipynb
     - example of calculating feature importances of representations to generative factors
 - clustering_benchmark_plot.ipynb
     - comparing different TCR clustering methods   
 - disentangling_score_plot.ipynb
     - Using pre-calculated feature importances to calculate disentangling scores
 - cluster_analysis_tcr_generation.ipynb
     - Analysis of two large flu clusters in labeled antigen data
     - generating a PWM logo image of a TCR generated de novo from a mean representation of a group of TCRs
 - example_single_latent_traversal.ipynb
     - linearly move in latent space between two TCRs
     - generate TCRs at steps on the way and score model smoothness 

#### *scripts:*

In the script directory you will find:
 
 - classification.py
     - script to perform classification of TCR-Ag bindin using TCR-VALID representations
     - includes OOD detection performance evaluation
 - single_interpolation.py
     - see notebook "example_single_latent_traversal" to see more details and figures
     - Interpolate in the TCR_VALID latent space between two TCRs and generate TCRs along all points in trajectory
     - Compare generated TCRs along trajectory fixed CDR3 and reference CDR2s to score manifold continuity
 - meta_interpolation.py
     - Perform many latent space traversals of the kind in "single_interpolation"
     - track performance over all MonteCarlo traversals for many models

## License

Copyright 2023 Regeneron Pharmaceuticals Inc.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at https://www.apache.org/licenses/LICENSE-2.0.

Please see the "LICENSE" file.

#### Third-party software and data

Note that the hinton.py module in /notebooks/ contains a function from matplotlib which has its own license, which is available in the notebooks directory as "matplotlib_v.1.2.0_LICENSE".

tcrvalid\/data\/TR\{A/B\}\_reference.csv:
 - two files of CDR1 and CDR2 sequences associated with each V gene.
 - collected using IMGT reference - IMGT &reg; the international ImMunoGeneTics information system &reg;, http\:\/\/www\.imgt\.org. \[Nucleic Acids Res., 43, D413\-D422 (2015)\].Terms of use [here](https://www.imgt.org/about/termsofuse.php).
 
tcrvalid\/data\/tts_full_TRA_te and tcrvalid\/data\/tts_TRBte are a subset of data collated from iReceptor \[Corrie et al. Immuonological reviews 284 (2018) \]and VDJServer \[Christley et al. Frontiers in immunology 9 (2018)\].



