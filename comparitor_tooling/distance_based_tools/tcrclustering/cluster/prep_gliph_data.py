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
import pandas as pd
import numpy as np
from tcrvalid.data_subsetting import make_subset,sources_,chains_,features_
from tcrvalid.metrics import get_cluster_purity_df,clustering_scoring
from tcrvalid.defaults import *

def get_rename_dict(chain_type):
    """ rename df columns to match wrapper package expectations
    """
    if chain_type == 'TRB':
        rename_cols = {
            'junction_aa': 'cdr3_TRB',
            'meta_v_call': 'v_gene_TRB',
            'meta_j_call': 'j_gene_TRB',
        }
    elif chain_type=='TRA':
         rename_cols = {
            'junction_aa': 'cdr3_TRA',
            'meta_v_call': 'v_gene_TRA',
            'meta_j_call': 'j_gene_TRA',
        }
    elif chain_type=='TR':
        rename_cols = {
            'junction_aa_TRA': 'cdr3_TRA',
            'meta_v_call_TRA': 'v_gene_TRA',
            'meta_j_call_TRA': 'j_gene_TRA',
            'junction_aa_TRB': 'cdr3_TRB',
            'meta_v_call_TRB': 'v_gene_TRB',
            'meta_j_call_TRB': 'j_gene_TRB',
            #'peptide_TRA': 'peptide' 
        }
    else: 
        raise ValueError('unknown chain type')
    return rename_cols

def get_data(chain_type,noflu):
    """ pull in the labeled dataset and make subset
    
    Currently assumes CDR23 dropping in this implementation
    
    If using only one chain - introduce dummy variables into other 
    so that consistent package input. Not used f not requested.
    
    """
    #df = pd.read_csv('/data/home/allen.leary/repos/tcrvalid/tcrvalid/data/GLIPH_Antigen_ref/Antigen_ref.csv')
    df = pd.read_csv(labelled_data_path)
    source_type='both'
    tmp_df = make_subset(
        df,
        sources=sources_[source_type],
        chains=chains_[chain_type],
        feature=features_['CDR23'],
        noflu=noflu
    )
    rename_cols = get_rename_dict(chain_type)
    tmp_df = tmp_df.rename(columns=rename_cols)
    if chain_type=='TRB':
        tmp_df['v_gene_TRA'] = ['TRAV1-1']*len(tmp_df)
        tmp_df['j_gene_TRA'] = ['TRAJ1']*len(tmp_df)
        tmp_df['cdr3_TRA'] = ['CAAAAAAAF']*len(tmp_df)
    elif chain_type=='TRA':
        tmp_df['v_gene_TRB'] = ['TRBV19']*len(tmp_df)
        tmp_df['j_gene_TRB'] = ['TRBJ2-1']*len(tmp_df)
        tmp_df['cdr3_TRB'] = ['CAAAAAAAF']*len(tmp_df)
    #vc = tmp_df.value_counts('peptide')
    #tmp_df = tmp_df[tmp_df['peptide'].isin(vc[vc>min_size].index)]
    return tmp_df

def prep_gliph_data(chain_type,data_type):
    """ Pull in and harmonize TCR sequences from GLIPH paper for downstream clustering benchmarking
        GLIPH publication does not use TRA so same dummies are used.
        Data type is either ref antigen labelled data or the path the spike in fold required and passed in the spike in cluster scoring with option to label spike in with either NaN or unique hash.
    """
    if data_type=='ref':
        tmp_df = pd.read_csv('/data/home/allen.leary/repos/tcrvalid/tcrvalid/data/GLIPH_Antigen_ref/Antigen_ref.csv')
        tmp_df['new_meta_vcall'] = tmp_df['v_call'].str.split('*').str[0]
        tmp_df['j_gene_TRB'] = tmp_df['j_call'].str.split('*').str[0]
        tmp_df.reset_index(inplace=True)
        #tmp_df['index'] = tmp_df['index'].apply(str)+tmp_df['junction_aa']
    else:
        tmp_df = pd.read_csv(data_type)
        tmp_df=tmp_df.rename(columns={'TRBV':'new_meta_vcall','CDR3':'junction_aa'})
        tmp_df['peptide']= np.nan
    
    df_TRBV_ref=pd.read_csv('/data/home/allen.leary/repos/tcrvalid/comparitor_tooling/distance_based_tools/data/reference/cdr12_labelled/TRBV_gene_ref.csv')        
    tmp_df=tmp_df.merge(df_TRBV_ref,on='new_meta_vcall',how='left',indicator=True)
    tmp_df["clono_id"] = tmp_df["cdr2_no_gaps"] + "-" + tmp_df.junction_aa.str[1:-1]
    tmp_df=tmp_df.rename(columns={'new_meta_vcall':'v_gene_TRB','CDR3':'cdr3_TRB','TRBJ':'j_gene_TRB','index':'sequence_id'})        
    rename_cols = get_rename_dict(chain_type)
    tmp_df = tmp_df.rename(columns=rename_cols)
    tmp_df=tmp_df.drop_duplicates()
    mask = tmp_df['clono_id'].str.len()<= 28
    tmp_df=tmp_df.loc[mask]
    tmp_df=tmp_df[~tmp_df.clono_id.str.contains('\*')].reset_index(drop=True)
    

    if chain_type=='TRB':
        tmp_df['v_gene_TRA'] = ['TRAV1-1']*len(tmp_df)
        tmp_df['j_gene_TRA'] = ['TRAJ1']*len(tmp_df)
        tmp_df['cdr3_TRA'] = ['CAAAAAAAF']*len(tmp_df)
    elif chain_type=='TRA':
        tmp_df['v_gene_TRB'] = ['TRBV19']*len(tmp_df)
        tmp_df['j_gene_TRB'] = ['TRBJ2-1']*len(tmp_df)
        tmp_df['cdr3_TRB'] = ['CAAAAAAAF']*len(tmp_df)

    return tmp_df
    