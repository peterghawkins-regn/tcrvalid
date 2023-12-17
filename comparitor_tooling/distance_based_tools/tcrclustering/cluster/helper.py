# Copyright 2023 Regeneron Pharmaceuticals Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from tcrclustering.cluster.ismart import IsmartClusterer,IsmartOptions
from tcrclustering.cluster.tcrdist import TCRDistClusterer,TCRDistOptions
from tcrclustering.cluster.tcrdist_parallel import TCRDistParallelClusterer,TCRDistParallelOptions
from tcrclustering import defaults
# from tcrclustering.cluster.metric_processing import (
#     MetricProcessor,
#     RescaleProcessor,
#     GapFillProcessor,
#     MetricCutProcessor 
# )
import pandas as pd
import os
tcrdistrun_path = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    'scripts',
    'modified_third_party',
    'tcrdist',
    'tcrdistrun.py'
)
TMP_PATH = os.path.join(os.path.expanduser('~'),'tmp','tcrclustering')
if not os.path.exists(TMP_PATH):
    os.makedirs(TMP_PATH)
TCR_PATH = os.path.join(os.path.expanduser('~'),'tmp','tcrclustering_output')
if not os.path.exists(TCR_PATH):
    os.makedirs(TCR_PATH)

def get_clusterer_by_name(name, options):
    if name=='ismart':
        return IsmartClusterer(options)
    elif name=='tcrdist':
        return TCRDistClusterer(options)
    elif name=='tcrdist_parallel':
        return TCRDistParallelClusterer(options)
    else:
        raise NotImplementedError("name : "+name+" is not a known clusterer type")
        
        

    
def ismart_simple_run(
        df,
        threshold=7.5,
    ):
    """ Run iSMART over data 
    
    Assumes you already removed duplicates appropriately, and don't care about distance matrices, just clusters
    No cluster are removed based on size.
    
    parameters
    -----------
    df: pandas.DataFrame
        TCR data to be clustered
        
    threshold: float
        iSMART distance threshold
        
    
    returns
    -------- 
    df_w_clusters: pandas.DataFrame
        clustering dataframe output
        
    """
    
    # with an options refactoring this could be very easily cleaned up so could swap
    # out the method more easily
    opts = IsmartOptions(
        defaults.DEFAULT_CONDA_PREFIX,
        defaults.DEFAULT_ISMART_PATH,
        defaults.DEFAULT_ISMART_env,
        'standard_env',
        TCR_PATH,
        tmp_dir=TMP_PATH,
        threshold=threshold,
        keep_dists=True # needed for UMAP
    )
    clusterer = get_clusterer_by_name('ismart', opts)
    
    df_w_clusters = clusterer.run_simple_clustering(
        df
    )
    
    return df_w_clusters

def tcrdist_simple_run(
        df,
        chains='AB',
        threshold=None,
        mode='AS_SVG'
    ):
    """ Run TCRdist over data and anchor points
    
    TCRdist original algorithm is run. 
    
    parameters
    -----------
    df: pandas.DataFrame
        TCR data to be clustered
        
    chains: str
    B for B only, A for A only, AB for paired.
        
    threshold: float
        None sets the distance radius for clustering to the default. Other values change the radius
        insode the tcrdist algo. e.g 70.
        
    returns
    -------- 
    df_w_clusters: pandas.DataFrame
        clustered dataframe
        
    """
    
    # with an options refactoring this could be very easily cleaned up so could swap
    # out the method more easily
    opts = TCRDistOptions(
        defaults.DEFAULT_CONDA_PREFIX,
        tcrdistrun_path,
        'tcrdist',
        'standard_env',
        TCR_PATH, 
        tmp_dir=TMP_PATH,
        chains=chains,
        threshold=threshold,
        subject_col=None,
        from_raw_nt_seqs=False,
        cluster_mode=mode
    )
    clusterer = get_clusterer_by_name('tcrdist', opts)
    
    df_w_clusters = clusterer.run_simple_clustering(
        df
    )
    
    return df_w_clusters
    
def tcrdist_parallel_simple_run(
        df,
        nproc=8,
        min_samples=3,
        eps=0.4,
        chains='AB'
    ):
    """ Run TCRdistParallel 
    
    parameters
    -----------
    df: pandas.DataFrame
        TCR data to be clustered
    
    nproc: int, optional, default=8
        number of processors to use
    
    min_samples: int, optional, default=3
        DSBSCAN parameter : Number of samples within distance eps considered close neighbor
        
    eps: float, optional, default=0.4
        DSBSCAN parameter : distance around a TCR considered close neighbor
        
    chains: string, optional,default='AB'
        'AB' means use both chains. 'B' for beta-only. 'A' for alpha-only.
        
        
    returns
    -------- 
    df_w_clusters: pandas.DataFrame
        clustered df
        
    """
    
    opts = TCRDistParallelOptions(
         defaults.DEFAULT_CONDA_PREFIX,
         'tcrdist',
         'standard_env',
         '.',
         nproc=nproc,
         chains=chains,
    )

    opts.min_samples = min_samples
    opts.eps = eps
    clusterer = get_clusterer_by_name('tcrdist_parallel', opts)
    
    df_w_clusters = clusterer.run_simple_clustering(
        df
    )
    
    return df_w_clusters
