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

""" Run DeepTCR classification and ID/OOD detection calculations

Note
-----
DeepTCR uses large amounts of RAM on inference, in our case this script tried to access
>100Gb of RAM

"""
import pandas as pd
import numpy as np

import tempfile
import os
import sklearn

import json
import argparse

from DeepTCR.DeepTCR import DeepTCR_SS


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-o',
        type=str,
        default='tcrvalid_classi_outputs',
        help='output directory',
        dest='outdir'
    )
    parser.add_argument(
        '-r',
        type=int,
        default=1,
        dest='repeat',
        help='label the data with a repeat number',
    )
    return parser

def calculate_confidences(y):
    n = y.shape[1]
    return (n/(n-1.))*(np.max(y,axis=1) - (1./n))


def get_data_locs():
    pre = '../deeptcr_vae/deeptcr_features'
    base = 'deeptcr_datafeats'
    chain = 'TR'
    minsize = 3

    name_ID = os.path.join(
        pre,
        base+'_'+chain+'_'+str(minsize)+'_classi_ID'
    )
    name_OOD = os.path.join(
        pre,
        base+'_'+chain+'_'+str(minsize)+'_classi_OOD'
    )
    return name_ID,name_OOD

def train_models(name_ID,n_folds=10):
    DTCR_SS = DeepTCR_SS('classi_A02')
    DTCR_SS.Get_Data(
        directory=name_ID,
        Load_Prev_Data=False,
        aggregate_by_aa=True,
        count_column=0,
        aa_column_alpha=1,
        aa_column_beta=2,
        v_alpha_column=3,
        v_beta_column=4,
    )

    DTCR_SS.Monte_Carlo_CrossVal(
        folds=n_folds,
        test_size=0.25,
        weight_by_class=True,
        suppress_output=True
    )

    return DTCR_SS

def get_split_idxs(DTCR_SS, n_folds=10):
    split_idxs = np.round(
        np.linspace(0,DTCR_SS.y_test.shape[0]+1,n_folds+1)
    ).astype(int) # folds+1 splits, round to int
    return split_idxs

def get_rocs(DTCR_SS, split_idxs):
    results = []
    for i in range(len(split_idxs)-1):
        idx0 = split_idxs[i]
        idx1 = split_idxs[i+1]
        r = sklearn.metrics.roc_auc_score(
            DTCR_SS.y_test[idx0:idx1],
            DTCR_SS.y_pred[idx0:idx1],
            average='weighted',
            multi_class='ovr'
        )
        results.append( r )
    return results

def ood_confidences_table(DTCR_SS,split_idxs,name_OOD,n_folds=10):
    
    peps = [
        'IPSINVHHY',
        'IVTDFSVIK',
        'RAKFKQLL'
    ]

    df_ood = pd.concat(
        [pd.read_csv(os.path.join(name_OOD,'{}/{}.tsv'.format(pep,pep)),sep='\t') for pep in peps]
    )

    results_dfs = []
    for i in range(n_folds):
        print(i)
        y_ood = DTCR_SS.Sequence_Inference(
            alpha_sequences=np.array(df_ood['alpha']).astype('<U22'),
            beta_sequences=np.array(df_ood['beta']).astype('<U22'),
            v_alpha=np.array(df_ood['v_alpha']).astype('<U12'),
            v_beta=np.array(df_ood['v_beta']).astype('<U12'),
            models=['model_{:d}'.format(i)]
        )
        confidences_id = calculate_confidences(DTCR_SS.y_pred[split_idxs[i]:split_idxs[i+1]])
        confidences_ood = calculate_confidences(y_ood)

        confs = np.hstack(
            (confidences_id, 
             confidences_ood)
        )
        types = ['test-ID']*DTCR_SS.y_pred[split_idxs[i]:split_idxs[i+1]].shape[0]
        types.extend(['test-OOD']*y_ood.shape[0])

        tmp_df = pd.DataFrame(
            {
                'data':types,
                'confidence':confs
            }
        )
        tmp_df['fold'] = i

        results_dfs.append(tmp_df)
    all_result_df = pd.concat(results_dfs)
    return all_result_df

def main(outdir, repeat=1):
    name_ID,name_OOD = get_data_locs()
    DTCR_SS = train_models(name_ID)
    split_idxs = get_split_idxs(DTCR_SS)
    rocs = get_rocs(DTCR_SS, split_idxs)
    all_result_df = ood_confidences_table(DTCR_SS,split_idxs,name_OOD)
    df_rocid = pd.DataFrame({
        'fold': np.arange(len(rocs)),
        'roc' : rocs
    })
    all_result_df.to_csv(os.path.join(outdir, 'ood_confidence_deeptcr_repeat_{:d}.csv'.format(repeat)),index=False)
    df_rocid.to_csv(os.path.join(outdir, 'classroc_deeptcr_repeat_{:d}.csv'.format(repeat)),index=False)
    return None

if __name__=='__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args.outdir, repeat=args.repeat)
    
    
    