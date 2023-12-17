#!/bin/bash

# if not already done - make the necessary conda environments
conda env create -f clustcr_env.yml

# create a place to put external tools
mkdir -p lib
cd lib

# get a version of tcr-dist (specific for consistency)
# update the repo with specific changed files 
# (from tcrclustering/modified_third_party/tcrdist)
if [ ! -d "clusTCR" ]; then
	git clone https://github.com/svalkiers/clusTCR.git
	cd clusTCR
    git checkout db12543
    cd ..
fi