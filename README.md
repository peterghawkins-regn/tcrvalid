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
 
Installation should not take more than a few minutes via pip.
For an idea of runtime of TCRVALID - check out the results in results\_data/timings
 
Models and data
----------------

All required models are co-packaged with TCR-VALID, and can be loaded by name with a loading utility function.

The following will load a single named model for TRB CDR2-CDR3 sequences:
```
from tcrvalid.load_models import *
model_name = '1_2'
loaded_trb_models = load_named_models(model_name,chain='TRB',as_keras=True)[model_name]
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
loaded_trb_models = load_named_models(model_names,chain='TRB',as_keras=True)
```

### Example embedding

```
from tcrvalid.load_models import *
from tcrvalid.physio_embedding import SeqArrayDictConverter
# get model for TRB
model_name = '1_2_full_40'
model = load_named_models(model_name,chain='TRB',as_keras=True)[model_name]

# get mapping object to convert protein sequence 
# to physicochemical representation
mapping = SeqArrayDictConverter()

# convert seq to physicochemical - list could have many sequences
# sequences are "CDR2-CDR3" formatted
x = mapping.seqs_to_array(['FYNNEI-ASSETLAGANTQY'],maxlen=28)

# get TCR-VALID representation
z,_,_ = model.predict(x)
print(z)
# expect:
# [[ 0.2436572  -1.5467906  -0.63847804  0.83660173  0.10755217 -0.28501382
#    0.9832421  -0.19073558  0.38733137 -0.5093988   0.5247447  -0.660075
#    0.04878296  0.5692204  -1.3631787   1.3796847 ]]
```

Embeddings of TCRs such as this can be used for clustering, classification, generation. More examples of such cases can be found in notebooks/. Package imports should take ~5s, model loading ~1.5s, and embedding calculation <1s. Times based on a 4-core CPU machine.
 
#### *data*

We also co-package esssential datasets for reproducing our findings. Namely:
 - antigen-labeled TCR dataset (TCR-VALID internal and GLIPH2 reference)
 - Spike splits of irrelevant CD4 TCRs for 1-5x spike in benchmarking
 - smallTRB test dataset and TRA test dataset
     - due to the data volume we do not provide the full datasets
 - Full datasets can be collated using the trb_repertoires.csv and tra_repertoires.csv
     - these give the repertoire_ids from iRecetor and VDJServer when compiling the datasets
 - TR\{A/B\}\_reference.csv:
     - two files of CDR1 and CDR2 sequences associated with each V gene.
     - collected using IMGT reference - IMGT &reg; the international ImMunoGeneTics information system &reg;, http\:\/\/www\.imgt\.org. \[Nucleic Acids Res., 43, D413\-D422 (2015)\].Terms of use [here](https://www.imgt.org/about/termsofuse.php).

Some of the data is stored in "results_data"

 - this is data used in some notebooks for ease of generating certain plots, but is not co-packaged with TCR-VALID is it not required for general use.
 
#### *comparitor_tooling*
We provide the tooling and wrappers to perform our TCR-antigen clustering benchmarking with and without irrelevant TCR spike ins. Briefly:
 - TCR-Antigen reference sets: TCR-Valid internal & GLIPH2 reference sets
 - Wrappers provided for following TCR featurization/clustering for benchmarking: tcr-dist, iSMART, clusTCR, deepTCR, ESM, tcrBERT.
   - comparitor\_tooling/distance\_based\_tools is a subpackage that can be used to run tcr-dist or iSMART - see README therein.
   - comparitor\_tooling/Spikein\_scores/Wrapper.ipynb will loop and score all methods and spike-in sets.
 - Individual per reference and tool clustering scoring for spike in ranges 0-5x can be found in Spikein_scores folder.
  - Clustering scoring meta dataframe for all benchmarking can be found in master_score.csv and is generated by Mater_scoring_generator.ipynb
 - Timing benchmarking tooling and scoring can be found in timing folder, or in the case of clusTCR in clustcr/scripts/timing.py
 - Scripts to score transformer methods are in comparitor\_methods/transformer/\{ESM,TCRBERT\}
 - DeepTCR representations for clustering are generated via DeepTCR-VAE in comapritor\_tooling/deeptcr_vae - see README therein
 - DeepTCR classification with OOD detection is shown in comparitor\_tooling/deeptcr_classifier/id_ood_detection.py
 

Examples
---------

#### *notebooks:*

In the /notebook directory you will find several examples and plotting methods:

 - cluster_tcrs_example.ipynb
     - example of how to cluster
 - hinton_feature_iportance_example.ipynb
     - example of calculating feature importances of representations to generative factors
 - clustering_benchmark_plot.ipynb
     - comparing different TCR clustering methods accounting for TRA & TRB chains and CDR usage  
 - disentangling_score_plot.ipynb
     - Using pre-calculated feature importances to calculate disentangling scores
 - cluster_analysis_tcr_generation.ipynb
     - Analysis of two large flu clusters in labeled antigen data
     - generating a PWM logo image of a TCR generated de novo from a mean representation of a group of TCRs
 - example_single_latent_traversal.ipynb
     - linearly move in latent space between two TCRs
     - generate TCRs at steps on the way and score model smoothness 
 - OneHotvsPC.ipynb
     - Featurize TCR-antigen reference data sets with One hot or Physicochemical featurization.
     - PCA reduce to 16D representations and cluster using DBscan to benchmark.  
 - Plot_classification_results.ipynb
     - Generate plots for Figure 6, banchmarking TCR-VALID's latent representations applied to classification and out of distribution benchmarking.
 - Plot_spike-in_clustering.ipynb
     - Import the clustering benchmarking for TCR-Antigen reference sets with and without spike in of irrelevant TCRs. Generate plots for supplemental figures 2-7
 - Plot_timings.ipynb
     - Import clustering timing benchmarking and generate the plots for timing in figure 4 and supplemental figure 8 
     
 - in /comparitor\_tooling/Spikein\_scores/Wrapper.ipynb:
     - Wrapper notebook to compute all the clustering benchmarking with and without spike ins (0-5x folds of irrelevant TCRs) for TCR-antigen reference sets (TCR-VALID internal and GLIPH2 references) using tcr-dist, iSMART and TCR-VALID.

#### *scripts:*

In the script directory you will find:
 
 - id_ood_classification.py
     - script to perform classification of TCR-Ag binding using TCR-VALID representations
     - includes OOD detection performance evaluation, with HLA\*02/non-HLA\*02 split
 - classification.py
     - script to perform classification of TCR-Ag binding using TCR-VALID representations
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

The wrappers around, and changes to, tcr-dist and iSMART in tcrvalid\/comparitor_tooling\/distance_based_tools, at scripts\/modified_third_party and tcrclustering\/modified_third_party, have their own licenses and are present in those programs.



