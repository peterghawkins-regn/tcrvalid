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

# in addition to the tcrvalid base env you will need to pip install the following:
#%pip install levenshtein
#%pip install tcrdist3
#
# You also need to install the distance_based_tools in the dir above this
# see the README for details
#
# run in the tcrvalid conda env

import Levenshtein
from sklearn.metrics import pairwise_distances
import pwseqdist as pw
from tcrdist.rep_funcs import _pw,_pws
import numba
import numpy as np
import pandas as pd
import time

from tcrvalid.physio_embedding import SeqArrayDictConverter
from tcrvalid.load_models import *
from tcrvalid.plot_utils import set_simple_rc_params
from tcrvalid.physio_embedding import SeqArrayDictConverter
from tcrvalid.defaults import *

import sys
import os
sys.path.append('../distance_based_tools')
pd.options.mode.chained_assignment = None
from tcrclustering.cluster.helper import (
    tcrdist_parallel_simple_run,
    tcrdist_simple_run,
    ismart_simple_run
)
import tcrclustering
from sklearn.cluster import DBSCAN

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context='talk',palette='bright')
set_simple_rc_params()

# --------------------------------------------------------------------------
# ---- Constants and helper functions
# --------------------------------------------------------------------------

tcrdist_metrics = { "cdr3_a_aa" : pw.metrics.nb_vector_tcrdist,
            "pmhc_a_aa" : pw.metrics.nb_vector_tcrdist,
            "cdr2_a_aa" : pw.metrics.nb_vector_tcrdist,
            "cdr1_a_aa" : pw.metrics.nb_vector_tcrdist,
            "cdr3_b_aa" : pw.metrics.nb_vector_tcrdist,
            "pmhc_b_aa" : pw.metrics.nb_vector_tcrdist,
            "cdr2_b_aa" : pw.metrics.nb_vector_tcrdist,
            "cdr1_b_aa" : pw.metrics.nb_vector_tcrdist}

tcrdist_weights = { 
            "cdr3_a_aa" : 3,
            "pmhc_a_aa" : 1,
            "cdr2_a_aa" : 1,
            "cdr1_a_aa" : 1,
            "cdr3_b_aa" : 3,
            "pmhc_b_aa" : 1,
            "cdr2_b_aa" : 1,
            "cdr1_b_aa" : 1}

tcrdist_kargs = {   "cdr3_a_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':3, 'ctrim':2, 'fixed_gappos':False},
            "pmhc_a_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
            "cdr2_a_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
            "cdr1_a_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
            "cdr3_b_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':3, 'ctrim':2, 'fixed_gappos':False},
            "pmhc_b_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
            "cdr2_b_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True},
            "cdr1_b_aa" : {'use_numba': True, 'distance_matrix': pw.matrices.tcr_nb_distance_matrix, 'dist_weight': 1, 'gap_penalty':4, 'ntrim':0, 'ctrim':0, 'fixed_gappos':True}}

def ut(dmat):
    N=dmat.shape[0]
    return  dmat[np.triu_indices(N,1)]

def lev_dmat(seqs):
    N = len(seqs)
    ds_mat_lv = np.zeros((N,N))
    for i in range(N):
        for j in range(i,N):
            ds_mat_lv[i,j] = Levenshtein.distance(seqs[i],seqs[j])
    # ds_lv = ds_mat_lv[np.triu_indices(N,1)]
    return ds_mat_lv

def tcrvalid_dmat(seq_reps):
    N=len(seq_reps)
    if N>30:
        ds_tcrvalid = pairwise_distances(seq_reps)#[np.triu_indices(N,1)]
    else:
        ds_tcrvalid = numba_dist(seq_reps)
    return ds_tcrvalid

@numba.jit(nopython=True, fastmath=True, parallel=False, boundscheck=False, nogil=True)
def numba_dist(a):
    dist = np.zeros((a.shape[0],a.shape[0]))
    for i in range(a.shape[0]):
        for j in range(i+1,a.shape[0]):
            for c in range(16):
                dist[i,j] += (a[i, c] - a[j, c])**2
    return np.sqrt(dist)

def tcrdist_dmat(seqs):
    tmp_df = pd.DataFrame({'cdr2-cdr3':seqs})
    tmp_df['cdr2_b_aa'] = tmp_df['cdr2-cdr3'].map(lambda x: x.split('-')[0])
    tmp_df['cdr3_b_aa'] = tmp_df['cdr2-cdr3'].map(lambda x: x.split('-')[1])
    df_tcrd = tmp_df[['cdr2_b_aa','cdr3_b_aa']]

    mymetrics  = {k:tcrdist_metrics[k] for k in df_tcrd.columns}
    myweights  = {k:tcrdist_weights[k] for k in df_tcrd.columns}
    mykargs  = {k:tcrdist_kargs[k] for k in df_tcrd.columns}

    ds_tcrd = _pws(
        df = df_tcrd, 
        metrics = mymetrics, 
        weights= myweights, 
        kargs=mykargs, 
        cpu = 1, 
        store = False
    )
    return ds_tcrd['tcrdist']

def tcrdist_dbscan(seqs, eps=35.,min_samples=2,metric='euclidean'):
    dd = tcrdist_dmat(seqs)
    dbs = DBSCAN(eps=eps,min_samples=min_samples,metric=metric)
    dbs.fit(dd) 
    return dbs

def tcrvalid_dbscan(zs, eps=3.0,min_samples=2,metric='euclidean'):
    dbs = DBSCAN(eps=eps,min_samples=min_samples,metric=metric)
    dbs.fit(zs) 
    return dbs

def n_repeats(n,cutoff=10000):
    if n<cutoff:
        return 5
    else:
        return 1
    
def timed_response(fn,input_var,n,cutoff=10000):
    all_times = []
    count=0
    while count<n_repeats(n,cutoff=cutoff):
        t0 = time.time()
        dd_= fn(input_var)
        t1 = time.time()
        all_times.append(t1-t0)
        count+=1
    # print(len(all_times))
    return np.mean(all_times)



# --------------------------------------------------------------------------
# ---- Load data
# --------------------------------------------------------------------------


trb_test_pq = data_path_small_trb['te'] 

loaded_trb_models = load_named_models('1_2_full_40',chain='TRB',as_keras=True)
trb_model = loaded_trb_models['1_2_full_40']

mapping = SeqArrayDictConverter()

te_seq_trb_df = pd.read_parquet(trb_test_pq,columns=['new_meta_vcall','cdr2_cdr3']).head(int(1e5))
te_seq_trb_df['cdr3'] = te_seq_trb_df['cdr2_cdr3'].map(lambda x: 'C'+x.split('-')[1]+'F')

# --------------------------------------------------------------------------
# ---- Time embedding using TCRVALID
# --------------------------------------------------------------------------


ns = np.array([10,100,1000,3000,10000,30000,100000])

times_embed_cpu = np.zeros((len(ns),))
for i,n in enumerate([10]+list(ns)):
    print(i,n)
    in_choice = te_seq_trb_df.cdr2_cdr3.sample(n)
    t0 = time.time()
    z_ = trb_model.predict(mapping.seqs_to_array(in_choice,maxlen=28))
    t1 = time.time()
    if i>0:
        times_embed_cpu[i-1] = t1-t0
    
big_z,_,_ = trb_model.predict(mapping.seqs_to_array(te_seq_trb_df.cdr2_cdr3.values,maxlen=28))

# --------------------------------------------------------------------------
# ---- Time getting distance matrix
# --------------------------------------------------------------------------

times_valid = np.zeros((len(ns),))
times_valid_numba = np.zeros((len(ns),))
times_dist = np.zeros((len(ns),))
times_lev = np.zeros((len(ns),))

for i,n in enumerate([10]+list(ns)):
    # Note added additional n=10 to start of loop
    # this is becuase first time numba functions are run they take longer to compile
    
    # get the data for this size (n):
    idx = np.random.randint(len(big_z), size=n)
    zs_ = big_z[idx,:]
    seqs_ = te_seq_trb_df['cdr2_cdr3'].iloc[idx].values

    print(i,i-1,n)
    
    # TCRVALID timing
    t_ = timed_response(tcrvalid_dmat,zs_,n)
    if i>0:
        times_valid[i-1] = t_

    if n<40000:
        # tcrdist3 (heavily optimized vs tcrdist) on same sequences
        t_ = timed_response(tcrdist_dmat,seqs_,n,cutoff=400)
        if i>0:
            times_dist[i-1] = t_
            
        t_ = timed_response(lev_dmat,seqs_,n, cutoff=400)
        if i>0:
            times_lev[i-1] = t_
            
            
# --------------------------------------------------------------------------
# ---- output
# --------------------------------------------------------------------------

df_timings_pairs = pd.DataFrame({
    'n':ns,
    'pair':ns*(ns-1)*0.5,
    'time_tcrvalid_pre-embedded':times_valid,
    'time_tcrvalid_only-embedding':times_embed_cpu,
    'time_tcrvalid_incl-embed':times_valid+times_embed_cpu,
    'time_tcrdist3':times_dist,
    'time_levenshtein':times_lev
})
df_timings_pairs.to_csv('../../results_data/timings/pairwise_data.csv')


# --------------------------------------------------------------------------
# ---- Clustering timing
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# ---- set up iSMART without storing distance matrix for fair test
# -- note iSMART not included in distance matrix comparison as
# -- only outputs a fraction of the distances between pairs
# --------------------------------------------------------------------------

TMP_PATH = os.path.join(os.path.expanduser('~'),'tmp','tcrclustering')
if not os.path.exists(TMP_PATH):
    os.makedirs(TMP_PATH)
TCR_PATH = os.path.join(os.path.expanduser('~'),'tmp','tcrclustering_output')
if not os.path.exists(TCR_PATH):
    os.makedirs(TCR_PATH)
    
def do_ismart_nokeepdists(df,threshold):
    opts = tcrclustering.cluster.ismart.IsmartOptions(
        tcrclustering.defaults.DEFAULT_CONDA_PREFIX,
        tcrclustering.defaults.DEFAULT_ISMART_PATH,
        tcrclustering.defaults.DEFAULT_ISMART_env,
        'standard_env',
        TCR_PATH,
        tmp_dir=TMP_PATH,
        threshold=threshold,
        keep_dists=False # needed for UMAP
    )
    clusterer = tcrclustering.cluster.ismart.IsmartClusterer(opts)
    
    df_w_clusters = clusterer.run_simple_clustering(
        df
    )
    return df_w_clusters


# --------------------------------------------------------------------------
# ---- do timing of clustering
# --------------------------------------------------------------------------


times_valid_cl = np.zeros((len(ns),))
times_ismart_cl = np.zeros((len(ns),))
times_dist_cl = np.zeros((len(ns),))


for i,n in enumerate([10]+list(ns)):
    # Note added additional n=10 to start of loop
    # this is becuase first time numba functions are run they take longer to compile
    
    print(i,i-1,n)
    # get the data for this size (n):
    idx = np.random.randint(len(big_z), size=n)
    zs_ = big_z[idx,:]
    seqs_ = te_seq_trb_df['cdr2_cdr3'].iloc[idx].values
    
    # ------------------------------------
    
    t_ = timed_response(tcrvalid_dbscan, zs_, n)
    if i>0:
        times_valid_cl[i-1] = t_ #t1-t0
    
    # ------------------------------------
    
    if n<40000:
        # ------------------------------------
        print('ismart  : ', n)
        df = pd.DataFrame(
            {
                'sequence_id':['tcr_{}'.format(label) for label in np.arange(len(idx))],
                'cdr3_TRB':te_seq_trb_df['cdr3'].iloc[idx].values,
                'v_gene_TRB':te_seq_trb_df['new_meta_vcall'].iloc[idx].values,
            }
        )
        lambda_fn = lambda x: do_ismart_nokeepdists(x,threshold=7.5)
        t_ = timed_response(lambda_fn, df, n, cutoff=400)
        if i>0:
            times_ismart_cl[i-1] = t_ #t1-t0
    
    if n<40000:
        # ------------------------------------
        print('tcrdist3')
        
        t_ = timed_response(tcrdist_dbscan, seqs_, n, cutoff=400)
        
        if i>0:
            times_dist_cl[i-1] = t_
        # ------------------------------------
        

df_timings_cl = pd.DataFrame({
    'n':ns,
    'pair':ns*(ns-1)*0.5,
    'time_tcrvalid_pre-embedded':times_valid_cl,
    'time_tcrvalid_only-embedding':times_embed_cpu,
    'time_tcrvalid_incl-embed':times_valid_cl+times_embed_cpu,
    'time_ismart':times_ismart_cl,
    'time_tcrdist3':times_dist_cl,
})
df_timings_cl.to_csv('../../results_data/timings/cluster_data.csv')