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
import os

__all__ = [
    'make_subset',
    'sources_',
    'chains_',
    'features_'
]

def make_subset(df,sources=None,chains=None,noflu=False,feature='CDR23',min_size=3,max_len=28):
    """

    parameters:
    -----------

    df: pd.DataFrame
    The dataframe of reference TCRs, each chain with own row, and cell_id to join

    source: list[string], optional, default is all sources
    If source name not provided - get all sources.
    Otherwise use ['REGN_ref_public','REGN_ref_10x'] to get REGN dataset

    chains: list[string], optional, defualt: TRB only
    Provide either ['TRB'], ['TRA'], ['TRA','TRB']

    feature: string
    CDR3 or CDR23 (both CDR2 and CDR3)

    min_size: int
    The minimum number of unqiue TCRs to a single peptide allowed. All
    ppetides have fewer than 3 unique TCRs (uniqueness defined by feature 
    and chains) will have their cogante TCRs removed.

    max_len: int
    The max length of amino acids in TRA or TRB

    returns
    --------

    df: pd.DataFrame
    pandas Dataframe with requested sources, chain(s) (joined if both),
    with duplictaes on the chosen feature removed, and all peptides with fewer 
    than min_size number of uniqe TCRs removed.


    """

    if sources is not None:
        df = df[df['source'].isin(sources)].copy()

    if noflu:
        df=df[df['peptide']!='GILGFVFTL']

    if chains is None:
        chains = ['TRB']
    
    df = df[df['locus'].isin(chains)]

    if feature=='CDR23':
        df['pre_feature'] = df['cdr2_no_gaps'] + '-' + df.junction_aa.map(lambda x: x[1:-1])
    elif feature=='CDR3':
        df['pre_feature'] = df.junction_aa.map(lambda x: x[1:-1])
    elif feature=='CDR123':
        df['pre_feature'] = df['cdr1_no_gaps'] + '-' + df['cdr2_no_gaps'] + '-' + df.junction_aa.map(lambda x: x[1:-1])
    else:
        raise ValueError()

    df = df[df['pre_feature'].map(lambda x: len(x)<=max_len)]

    if len(chains)==2:
        df = df[df['locus']=='TRA'].merge(df[df['locus']=='TRB'],on='cell_id',suffixes=['_TRA','_TRB'])
        df['feature'] = df['pre_feature_TRA'] + '-' + df['pre_feature_TRB']
        df['peptide'] = df['peptide_TRA']
    else:
        df['feature'] = df['pre_feature']

    df = df.drop_duplicates(subset=['feature'])

    df['clono_id'] = df['feature']

    if len(chains)!=2:
        df['sequence_id'] = df['cell_id']+'_'+df['locus']
    else:
        df['sequence_id'] = df['cell_id']

    vc = df.value_counts('peptide')
    df = df[df['peptide'].isin(vc[vc>min_size].index)]

    return df

sources_ = {
  'regnonly':['REGN_ref_public','REGN_ref_10x'],
  'both':None
}

chains_ = {
  'TRB':['TRB'],
  'TRA':['TRA'],
  'TR':['TRA','TRB']
}

features_ = {
  'CDR3':'CDR3',
  'CDR23':'CDR23',
  'CDR123':'CDR123'
}