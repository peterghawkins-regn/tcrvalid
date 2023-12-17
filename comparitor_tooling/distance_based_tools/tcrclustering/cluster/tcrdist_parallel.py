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
from tcrclustering.cluster.base import BaseTCRClusterer, Options
from tcrclustering.cluster.utils import load_json_to_dict, AA_TO_NUC
from tcrclustering.modified_third_party.tcrdist import tcrdist_distances
from tcrclustering.modified_third_party.tcrdist.tcrdist_distances import (
    default_distance_params,
    rep_dists,
    weighted_cdr3_distance
)
import os
import pandas as pd
import subprocess
import shutil
import pickle
import numpy as np
from functools import partial
from tcrclustering.parse.reference import ReferenceAAMapper
from multiprocessing.pool import Pool
import multiprocessing as mp
from sklearn.cluster import DBSCAN

class TCRDistParallelOptions(Options):
    
    def __init__(self,
                 conda_prefix,
                 env,
                 standard_env,
                 output_path,
                 tmp_dir = './tmp',
                 nproc = -1,
                 organism='human',
                 chains='AB',
                 eps=80,
                 min_samples=3,
                 gap_penalty=-6,
                 gap_number=0,
                 subject_col='SubjectID',
                 from_raw_nt_seqs=False,
                ):
        super(TCRDistParallelOptions,self).__init__()
        self.conda_prefix = conda_prefix
        self.env = env
        self.standard_env = standard_env
        self.output_path = output_path
        self.tmp_dir = tmp_dir
        self.eps = eps
        self.min_samples = min_samples
        self.gap_penalty = gap_penalty
        self.gap_number = gap_number
        self.subject_col = subject_col
        self.organism = organism
        self.chains=chains
        self.from_raw_nt_seqs = from_raw_nt_seqs
        if nproc==-1:
            self.nproc=os.cpu_count()-1
        else:
            self.nproc=nproc

        self.name = 'tcrdist_parallel'
        
    @classmethod
    def from_json(cls,json_path):
        json_dict = load_json_to_dict(json_path)
        args = [
             'conda_prefix',
             'env',
             'standard_env',
             'output_path',
        ]
        inargs = [json_dict[a] for a in args]
        kwargs = [
            'tmp_dir',
            'eps',
            'min_samples'
            'gap_penalty',
            'gap_number',
            'use_v'
        ]
        inkwargs = {kw:json_dict[kw] for kw in kwargs}
        inkwargs.update({'nproc':os.cpu_count()-1})
        out_opts = cls(*inargs,**inkwargs)
        out_opts.add_note = json_dict['note']
        return out_opts
    
def seq_distance_row_lazy(
    i,
    v_a=None,
    v_b=None,
    cdr3_a=None,
    cdr3_b=None,
    n=None,
    chains='AB',
    rep_dists=rep_dists,
    default_distance_params=default_distance_params
):
    def single_case(inc):
        dist=0
        if 'A' in chains:
            dist += int(
                    rep_dists[v_a[inc[0]]][v_a[inc[1]]] +\
                    weighted_cdr3_distance( cdr3_a[inc[0]], cdr3_a[inc[1]], default_distance_params )
            )
        if 'B' in chains:
            dist += int(
                    rep_dists[v_b[inc[0]]][v_b[inc[1]]] +\
                    weighted_cdr3_distance( cdr3_b[inc[0]], cdr3_b[inc[1]], default_distance_params )
            )
        return int(default_distance_params.scale_factor * dist)
    ds = np.zeros((n,),dtype=np.uint16)
    for j in range(n):
        ds[j] = single_case((i,j))
    return ds


class TCRDistParallelClusterer(BaseTCRClusterer):
    
    def __init__(self,options):
        super(TCRDistParallelClusterer,self).__init__(options)
        self.distance_approach=True
        
    def _prepare_df(self,df):
        reference_mappers = {
            'TRB' : ReferenceAAMapper(chain='TRB'),
            'TRA' : ReferenceAAMapper(chain='TRA')
        }
        for lc_name,locus in zip(['a','b'],['TRA','TRB']):
                df['v_gene_'+locus] = df['v_gene_'+locus].map(lambda x : reference_mappers[locus].collect_v_name(x))
                df['j_gene_'+locus] = df['j_gene_'+locus].map(lambda x : reference_mappers[locus].collect_j_name(x))
                df = df[df['cdr3_'+locus].map(lambda x: len(x)>5)]
                df = df[~df['v_gene_'+locus].isna()]
        return df
    
    def _get_distance_matrix(self,df,chains='AB'):
        # new_df = df[df['spike']==0]
        v_a = df.v_gene_TRA.values
        v_b = df.v_gene_TRB.values
        cdr3_a = df.cdr3_TRA.values
        cdr3_b = df.cdr3_TRB.values
        n=len(df)
        with Pool(self.options.nproc) as pool:
            arr2d = np.array( 
                pool.map( 
                    partial(
                        seq_distance_row_lazy,  
                        v_a=v_a,
                        v_b=v_b,
                        cdr3_a=cdr3_a,
                        cdr3_b=cdr3_b,
                        n=n,
                        chains=chains
                    ),
                    np.arange(n)
                ) 
            ) 
        return arr2d
        
   
    def _run_dbscan(self,df):
        clustering = DBSCAN(
            eps=self.options.eps, 
            min_samples=self.options.min_samples,
            metric='precomputed').fit(self.distance_matrix)
        df['cluster'] = clustering.labels_
        df['cluster'] = df['cluster'].replace({-1:np.nan})
        return df
        
    def _run_clustering(self,
                    df,
                    barcode_column='barcode',
                    clono_column='clono_id',
                    save_to_output=True
                   ):
        print("internal len start ", len(df))
        df = self._prepare_df(df)
        print("internal len postprep ", len(df))
        
        self.internal_ids = df.sequence_id
        self.internal_ids_name = 'sequence_id'
        self.distance_matrix = self._get_distance_matrix(df,chains=self.options.chains)
        
        df = self._run_dbscan(df)
        self.df_run = df
        print("internal len post dbscan ", len(df))
        print("inside internal : ",list(df.columns))
        return None
    
    def _assign_cluster_info(self,df_out,df,clono_column):
        clono_cluster_map = {
            clono:cl for clono,cl in zip(df_out[clono_column],df_out['cluster'])
        }
        df['cluster'] = df[clono_column].map(clono_cluster_map)
        cl_size_map = {v:c for v,c in df_out['cluster'].value_counts().iteritems()}
        df['cluster_size_orig'] = df['cluster'].map(cl_size_map)
        return df
    
    def apply_clustering(self,
                         df,
                         clono_column
                        ):
        df_cl = self._assign_cluster_info(self.df_run,df,clono_column)
        return df_cl
