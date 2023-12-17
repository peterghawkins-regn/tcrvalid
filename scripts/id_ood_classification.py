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
""" Run TCR-VALID classification and ID/OOD detection calculations

run as:
python id_ood_classification.py -o ../results_data/classification_data -r 1
from the dir of this file

"""
import numpy as np
import mlflow
import pandas as pd
import os
import sys

from tcrvalid.load_models import *
from tcrvalid.plot_utils import set_simple_rc_params
from tcrvalid.physio_embedding import SeqArrayDictConverter
from tcrvalid.data_subsetting import *
from tcrvalid.defaults import *
from tcrvalid.classification import *

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pickle

import argparse

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
    

def main(outdir, repeat=1):
    # ----------------- set up chosen models and data to use --------------------------------
    from_keras = True #expect True in github code
    load_mini_trb = True #expect True in github code, for paper results: False (very similar)
    pull_using_spark = False #expect False in github. if true, use spark to read in data, easier to use pandas for small dataset

    model_names = {
        'TRA': ['1_2'],
        'TRB': ['1_2_full_40']
    }

    if pull_using_spark:
        if not load_mini_trb:
            trb_test_pq = data_path_full_trb_spark['te']
        else:
            trb_test_pq = data_path_small_trb_spark['te']
        tra_test_pq = data_path_full_tra_spark['te']
    else:
        if not load_mini_trb:
            trb_test_pq = data_path_full_trb['te']
        else:
            trb_test_pq = data_path_small_trb['te']    
        tra_test_pq = data_path_full_tra['te']

    print(tra_test_pq, trb_test_pq)    

    mapping = SeqArrayDictConverter()

    loaded_trb_models = load_named_models(model_names['TRB'],chain='TRB',as_keras=from_keras)
    loaded_tra_models = load_named_models(model_names['TRA'],chain='TRA',as_keras=from_keras)

    if pull_using_spark:
        sdf_trb=spark.read.format("parquet").load(trb_test_pq)
        te_seq_trb_df = sdf_trb.select('cdr2_no_gaps','cdr2_cdr3').limit(100000).toPandas()

        sdf_tra=spark.read.format("parquet").load(tra_test_pq)
        te_seq_tra_df = sdf_tra.select('cdr2_no_gaps','cdr2_cdr3').limit(100000).toPandas()
    else:
        te_seq_trb_df = pd.read_parquet(trb_test_pq, columns=['cdr2_no_gaps','cdr2_cdr3']).head(100000)
        te_seq_tra_df = pd.read_parquet(tra_test_pq, columns=['cdr2_no_gaps','cdr2_cdr3']).head(100000)

    df = pd.read_csv(labelled_data_path)
    sources = sources_['both']
    chains = chains_['TR']
    feature = features_['CDR23']
    subset_df = make_subset(df,sources=sources,chains=chains,feature=feature,min_size=3, max_len=28)

    # use only antigens with at least 100 unique TCRs
    vc = subset_df.value_counts('peptide')
    peptides = vc[vc>100].index
    df_labelled = subset_df[subset_df['peptide'].isin(peptides)]

    OOD = [
        'IVTDFSVIK',
        'RAKFKQLL',
        'IPSINVHHY'
    ]

    tmp_ = df_labelled[~df_labelled['peptide'].isin(OOD)]
    df_ood = df_labelled[df_labelled['peptide'].isin(OOD)]

    df_labelled = tmp_

    peptides_id = list(df_labelled.peptide.unique())
    pep_dict = {p:i for i,p in enumerate(peptides_id)}
    df_labelled['label'] = df_labelled['peptide'].map(pep_dict)

    peptides_ood = list(df_ood.peptide.unique())
    pep_dict_ood = {p:i for i,p in enumerate(peptides_ood)}
    df_ood['label'] = df_ood['peptide'].map(pep_dict_ood)

    tra_model = loaded_tra_models['1_2']
    trb_model = loaded_trb_models['1_2_full_40']

    # convert raw data to TCRVALID features

    x_l, y_l = get_labelled_data(
      df_labelled,
      mapping,
      trb_model=trb_model,
      tra_model=tra_model
    )

    x_u = get_unlabelled_data(
      te_seq_trb_df, 
      te_seq_tra_df,
      mapping,
      trb_model=trb_model,
      tra_model=tra_model
    )

    x_l_ood, y_l_ood = get_labelled_data(
      df_ood,
      mapping,
      trb_model=trb_model,
      tra_model=tra_model
    )

    # run MC-xval for different balances b/w classification
    # and OOD detection objectives
    results_dfs = []
    performances = []
    for b in [0.,1.,10.,100.,500.]:
        # 10-fold MC xval
        for i in range(10):
            performance, model, data_l = run_case(
                x_l,x_u,y_l,
                balance=b,
                dim_in=32,
                all_outputs=True,
                test_size=0.125, 
                vali_size=0.125, 
                vali_absolute=True
            )
            pred_ood = model.predict(x_l_ood)
            pred_id = model.predict(data_l['test'][0])

            confs = np.hstack((calculate_confidences(pred_id), calculate_confidences(pred_ood)))
            types = ['test-ID']*pred_id.shape[0]
            types.extend(['test-OOD']*pred_ood.shape[0])

            tmp_df = pd.DataFrame(
                {
                    'data':types,
                    'confidence':confs
                }
            )
            tmp_df['balance'] = b
            tmp_df['fold'] = i

            results_dfs.append(tmp_df)

            performance['balance'] = b
            performance['fold'] = i

            performances.append(performance)

    all_result_df = pd.concat(results_dfs)

    df_rocid = pd.DataFrame(
        [(p['balance'],p['fold'],p['AUROC_test_avg']) for p in performances],
        columns = ['balance','fold','roc']
    )
    df_ood_unlabelled = pd.DataFrame(
        [(p['balance'],p['fold'],p['AUROC_ood']) for p in performances],
        columns = ['balance','fold','roc']
    )

    all_result_df.to_csv(os.path.join(outdir, 'ood_confidence_tcrvalid_repeat_{:d}.csv'.format(repeat)),index=False)
    df_rocid.to_csv(os.path.join(outdir, 'classroc_tcrvalid_repeat_{:d}.csv'.format(repeat)),index=False)
    df_ood_unlabelled.to_csv(os.path.join(outdir, 'unlabeled_ood_roc_tcrvalid_repeat_{:d}.csv'.format(repeat)),index=False)
    perf_fname = os.path.join(outdir, 'perf_tcrvalid_repeat_{:d}.pickle'.format(repeat))
    with open(perf_fname, 'wb') as handle:
        pickle.dump(performances, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return None

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args.outdir, repeat=args.repeat)
    
    