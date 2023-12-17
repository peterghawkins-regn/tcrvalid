#!/bin/bash

# make the necessary conda environments
#cd envs
#conda env create -f tcrdist_env.yml
#conda env create -f ismart_env.yml
#cd ..

# create a place to put external tools
mkdir -p lib
cd lib

# get a version of tcr-dist (specific for consistency)
# update the repo with specific changed files 
# (from tcrclustering/modified_third_party/tcrdist)
if [ ! -d "tcr-dist" ]; then
    source ${CONDA_PREFIX}/etc/profile.d/conda.sh
	conda activate tcrdist
	git clone https://github.com/phbradley/tcr-dist.git
	cd tcr-dist
    git checkout 1f5509a
	python setup.py
	cd ..
fi

cd ..

# update tcr-dist module to modified version (license inside)
# this allows su to collect the tcrs in each cluster (not just figures)
cp tcrclustering/modified_third_party/tcrdist/make_tall_trees.py lib/tcr-dist/


