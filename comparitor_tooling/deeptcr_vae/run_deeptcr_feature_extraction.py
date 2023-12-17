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

import argparse

import sys
sys.path.append('/data/home/peter.hawkins/repos/mp_bb/tcrvalid/')

import pandas as pd
import numpy as np
import json
import tempfile
import os
import sklearn

from DeepTCR.DeepTCR import DeepTCR_U

from tcrvalid.metrics import get_cluster_purity_df,clustering_scoring
from tcrvalid.defaults import *

from deeptcr_data_formatting import df_to_deeptcr_dir,safemake

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        type=str,
        default='TRB',
        help='chain type: TRB,TRA, or TR',
        dest='chain_type'
    )
    parser.add_argument(
        '-s',
        type=int,
        default=3,
        dest='min_size',
        help='path to data',
    )
    parser.add_argument(
        '--c',
        default=False,
        action='store_true',
        dest='classi',
        help='for classi task - sets peptides with at least 100 post-facto',
    )
    parser.add_argument(
        '--idood',
        default=False,
        action='store_true',
        dest='idood',
        help='for pure classi task - saves OOD and ID sets sperately, no VAE feats',
    )
    return parser

OOD = [
    'IVTDFSVIK',
    'RAKFKQLL',
    'IPSINVHHY'
]
    
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

# note this is as in tcrvalid witha minor change due to difference in
# pandas version and API b/w deepTCR ebvironemtn and the tcrvalid environment
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
    
    # Minor modification due to value_counts() definition in different pandas version
    #vc = df.value_counts('peptide')
    #
    vc = df['peptide'].value_counts()
    
    df = df[df['peptide'].isin(vc[vc>min_size].index)]

    return df

def get_data(chain_type,min_size=3,noflu=False, feature_name='CDR23', classi=False):
    """ pull in the labeled dataset and make subset
    
    Currently assumes CDR23 dropping in this implementation
    
    If using only one chain - introduce dummy variables into other 
    so that consistent package input. Not used f not requested.
    
    """
    df = pd.read_csv(labelled_data_path)
    source_type='both'
    tmp_df = make_subset(
        df,
        sources=sources_[source_type],
        chains=chains_[chain_type],
        feature=features_[feature_name],
        min_size=min_size,
        noflu=noflu
    )
    if classi:
        vc = tmp_df['peptide'].value_counts()
        peptides = vc[vc>100].index
        return tmp_df[tmp_df['peptide'].isin(peptides)]
    else:
        return tmp_df
        
def get_dirname(chain_type,min_size,classi,base_name = 'deeptcr_datafeats_'):
    base_name += chain_type
    base_name += '_'
    base_name += str(min_size)
    if classi:
        base_name += '_classi'
    return base_name
        
def get_deeptcr_reps(chain_type='TRB',min_size=3, classi=False, idood=False):
    """ read data, convert format, Train VAE, extract features """
    df = get_data(chain_type, min_size=min_size, classi=classi)
    
    if not idood:

        dirname = os.path.join(
            'deeptcr_features',
            get_dirname(chain_type,min_size,classi)
        )
        safemake(dirname)

        df_to_deeptcr_dir(df, chain_type, dirname)
        
    else:
        dirname_id = os.path.join(
            'deeptcr_features',
            get_dirname(chain_type,min_size,classi)+'_ID'
        )
        safemake(dirname_id)

        df_to_deeptcr_dir(df[~df['peptide'].isin(OOD)], chain_type, dirname_id)
        
        dirname_ood = os.path.join(
            'deeptcr_features',
            get_dirname(chain_type,min_size,classi)+'_OOD'
        )
        safemake(dirname_ood)

        df_to_deeptcr_dir(df[df['peptide'].isin(OOD)], chain_type, dirname_ood)
        print('ID/OOD option run - return without VAE features')
        return None
        
    tmp_dirname = get_dirname(chain_type,min_size,classi,base_name='tmpvae_')
    DTCRU = DeepTCR_U(tmp_dirname)

    #Load Data from directories
    if chain_type=='TRB':
        DTCRU.Get_Data(
            directory=dirname,
            Load_Prev_Data=False,
            aggregate_by_aa=True,
            count_column=0,
            aa_column_beta=1,
            v_beta_column=2,
            j_beta_column=None
        )
    elif chain_type=='TRA':
        DTCRU.Get_Data(
            directory=dirname,
            Load_Prev_Data=False,
            aggregate_by_aa=True,
            count_column=0,
            aa_column_alpha=1,
            v_alpha_column=2,
            j_beta_column=None
        )
    elif chain_type=='TR':
        DTCRU.Get_Data(
            directory=dirname,
            Load_Prev_Data=False,
            aggregate_by_aa=True,
            count_column=0,
            aa_column_alpha=1,
            aa_column_beta=2,
            v_alpha_column=3,
            v_beta_column=4,
            j_beta_column=None
        )
    else:
        raise ValueError('unknown chain type : {}'.format(chain_type))

    DTCRU.Train_VAE(Load_Prev_Data=False)

    if chain_type=='TRB':
        features = DTCRU.Sequence_Inference(
            beta_sequences=DTCRU.beta_sequences,
            v_beta=DTCRU.v_beta
        )
    elif chain_type=='TRA':
        features = DTCRU.Sequence_Inference(
            alpha_sequences=DTCRU.alpha_sequences,
            v_alpha=DTCRU.v_alpha
        )
    elif chain_type=='TR':
        features = DTCRU.Sequence_Inference(
            beta_sequences=DTCRU.beta_sequences,
            v_beta=DTCRU.v_beta,
            alpha_sequences=DTCRU.alpha_sequences,
            v_alpha=DTCRU.v_alpha
        )
    else:
        raise ValueError('unknown chain type : {}'.format(chain_type))


    idx_to_class = {i:v for i,v in enumerate(pd.Series(DTCRU.class_id).unique())}
    class_to_idx = {v:i for i,v in enumerate(pd.Series(DTCRU.class_id).unique())}
    int_labels = [class_to_idx[c] for c in DTCRU.class_id]

    np.save( os.path.join(dirname,'features.npy'), features)
    np.save( os.path.join(dirname,'labels.npy'), np.array(int_labels))
    with open(os.path.join(dirname,'labelnames.json'), 'w') as fp:
        json.dump(idx_to_class, fp)
    df.to_csv(os.path.join(dirname,'input_df_predeeptcrreformat.csv'))
    
    if classi:
        te_seq_trb_df = pd.read_csv('trb_unlabelled.csv')
        te_seq_tra_df = pd.read_csv('tra_unlabelled.csv')
        features_u = DTCRU.Sequence_Inference(
            beta_sequences=np.array(te_seq_trb_df['junction_aa']).astype('<U22'),
            v_beta=np.array(te_seq_trb_df['new_meta_vcall']).astype('<U12'),
            alpha_sequences=np.array(te_seq_tra_df['junction_aa']).astype('<U22'),
            v_alpha=np.array(te_seq_tra_df['new_meta_vcall']).astype('<U12')
        )
        np.save( os.path.join(dirname,'features_u.npy'), features_u)
    return None

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
    print(args.chain_type, args.min_size, args.classi)
    get_deeptcr_reps(
        chain_type=args.chain_type,
        min_size=args.min_size,
        classi=args.classi,
        idood=args.idood
    )