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

def safemake(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def df_to_deeptcrformat(df,locus='TR'):
    if locus == 'TRB':
        df = df[[
            'junction_aa',
            'meta_v_call',
        ]]
        df = df.rename(columns={
            'junction_aa':'beta', 
            'meta_v_call':'v_beta',
        })
        df['counts'] = 1
        df = df[['counts','beta','v_beta']]
    elif locus=='TRA':
        df = df[[
            'junction_aa',
            'meta_v_call',
        ]]
        df = df.rename(columns={
            'junction_aa':'alpha', 
            'meta_v_call':'v_alpha',
        })
        df['counts'] = 1
        df = df[['counts','alpha','v_alpha']]
    else:
        df = df[[
            'junction_aa_TRA',
            'meta_v_call_TRA',
            'junction_aa_TRB',
            'meta_v_call_TRB',
        ]]
        df = df.rename(columns={
            'junction_aa_TRA':'alpha', 
            'meta_v_call_TRA':'v_alpha',
            'junction_aa_TRB':'beta', 
            'meta_v_call_TRB':'v_beta',
        })
        df['counts'] = 1
        df = df[[
            'counts','alpha','beta','v_alpha','v_beta'
        ]]
    return df

def df_to_deeptcr_dir(df, locus, dirname):
    safemake(dirname)
    for pep in df.peptide.unique():
        directory = os.path.join(dirname,pep)
        safemake(directory)
        outpath = os.path.join(
            directory,
            pep+'.tsv'
        )
        # tsv seperator
        df_to_deeptcrformat(
            df[df['peptide']==pep],
            locus=locus
        ).to_csv(
            outpath,
            index=False,
            sep='\t',
        )