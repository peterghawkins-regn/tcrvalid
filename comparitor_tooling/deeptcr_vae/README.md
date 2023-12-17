DeepTCR - VAE for clustering and classification
-----------------

To re-run all:
 1. Run unlabelled_data_to_csv.ipynb first in the tcrvalid environment
 2. create a deepttcr conda env using the yml file
 3. Then run extract_features.sh (uncommenting cases of interest) to calculate features for each case (pre-saved here in repo)
 
To see how clustering works on the resulting features:
 - run the apply_clustering_on_deeptcr.ipynb

For classification:
For comparison of classification we used deepTCR in its classifier mode - see ../deeptcr_classification.
Script here creates dataset for ID and OOD peptides that are then used in the scripts in ../deeptcr_classification.


 
