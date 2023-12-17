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
import torch
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import argparse
import os

import sys
sys.path.append('../../..')

from tcrvalid.data_subsetting import *
from tcrvalid.cluster_loop import dbscan_loop
from tcrvalid.metrics import clustering_scoring

print("CUDA on : ", torch.cuda.is_available())
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


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
        '-f',
        type=str,
        default='CDR3',
        dest='feature',
        help='feature: CDR123,CDR23,CDR3',
    )
    parser.add_argument(
        '-cache',
        type=str,
        default=None,
        dest='torchhub_cache',
        help='optional location to search for torchhub cache'
    )
    parser.add_argument(
        '-o',
        type=str,
        default='outputs',
        dest='outpath',
        help='where to save outputs'
    )
    return parser

def get_reps(data, model, batch_converter):
    representations = np.zeros((len(data),1280))
    for i in range(len(data)):
        final_input = (str(i), data[i])
        batch_labels, batch_strs, batch_tokens = batch_converter([final_input])
        bt = batch_tokens.to(device)
        with torch.no_grad():
            result = model(bt, repr_layers=[33])
            rep_np = result["representations"][33].detach().to("cpu").numpy()
            representations[i,:] = np.squeeze(rep_np).mean(axis=0)
    return representations


def trim_indices(pw_distances):
    new_pw_distances = []
    for i in range(len(pw_distances)-1):
        arr = list(pw_distances[i][i+1:])
        for j in range(len(arr)):
            new_pw_distances.append(arr[j])
    return new_pw_distances

def get_epsilons(df, low_bound, up_bound):
    distances = pairwise_distances(df, metric="euclidean")
    trimmed = trim_indices(distances)
    median = pd.Series(trimmed).median()
    upper = median * up_bound
    lower = median* low_bound
    range_vals = upper - lower
    increment = range_vals/15
    return np.arange(lower, upper, increment)

        
class MeanFeaturesTransformer:
    def __init__(self, column, model, batch_converter, metric, lower, upper, num_eps):
        self.labeled_reps = get_reps(column, model, batch_converter)
        self.metric = metric
        self.lower = lower
        self.upper = upper
        self.num_eps = num_eps
        self.epsilon_values = self.get_epsilons(self.labeled_reps)
    
    def get_epsilons(self, reps):
        distances = pairwise_distances(reps, metric="euclidean")
        trimmed = trim_indices(distances)
        median = pd.Series(trimmed).median()
        upper = median * self.upper
        lower = median* self.lower
        range_vals = upper - lower
        increment = range_vals/self.num_eps
        return np.arange(lower, upper, increment)
    
    
def main(chains, feature, outpath, torchhub_dir=None):
    
    df = pd.read_csv('../../../tcrvalid/data/antigen_reference_tcrs.csv')
    
    if torchhub_dir:
        torch.hub.set_dir(torchhub_dir)

    model, alphabet = torch.hub.load("facebookresearch/esm:main", "esm1b_t33_650M_UR50S")
    batch_converter = alphabet.get_batch_converter()
    tok2idx = alphabet.tok_to_idx
    idx2tok = {v:k for k,v in tok2idx.items()}
    model = model.to(device)
    
    subset_df = make_subset(
        df,
        sources=sources_['both'],
        chains=chains_[chains],
        feature=features_[feature]
    )
    
    seq_list = list(subset_df.feature.values)
    
    file_name = "scores_{}_{}".format(
        chains,
        feature
    )

    if chains in ['TRA','TRB']:
        scan_helper = MeanFeaturesTransformer(
            seq_list, 
            model,
            batch_converter, 
            "euclidean",
            0.3, 
            0.8, 
            15
        )

        all_scores_df = dbscan_loop(
            subset_df, 
            scan_helper.labeled_reps, 
            scan_helper.epsilon_values,
            metric='euclidean',
            eps_format=':.2f'
        )
    else:
        alpha_list = list(subset_df.pre_feature_TRA.values)
        beta_list = list(subset_df.pre_feature_TRB.values)
        alpha_feats = get_reps(alpha_list,model,batch_converter)
        beta_feats = get_reps(beta_list,model,batch_converter)
        feats = np.concatenate((alpha_feats, beta_feats),axis=1)
        epsilons = get_epsilons(feats, .3, .8)
        all_scores_df = dbscan_loop(
            subset_df, 
            feats, 
            epsilons,
            metric='euclidean',
            eps_format=':.2f'
        )
        
        
    if outpath:
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        all_scores_df.to_csv(
            os.path.join(outpath, file_name)
        )
    
                
if __name__=='__main__':
    
    parser = get_parser()
    args = parser.parse_args()
    
    main(
        args.chain_type, 
        args.feature, 
        args.outpath, 
        torchhub_dir=args.torchhub_cache
    )
    
    
    
    
    