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
import sys
sys.path.append('../../..')

from tcrvalid.data_subsetting import *
from tcrvalid.cluster_loop import dbscan_loop
from tcrvalid.metrics import clustering_scoring

import argparse
import os
import json
import subprocess
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c',
        type=str,
        default='TRB',
        help='chain type: TRB,TRA',
        dest='chain_type'
    )
    parser.add_argument(
        '-f',
        type=str,
        default='CDR3',
        dest='feature',
        help='feature: CDR3',
    )
    parser.add_argument(
        '-o',
        type=str,
        default='outputs',
        dest='outpath',
        help='where to save outputs'
    )
    return parser


def get_tcr_reps(path, reps_path, layer):
    python_script = "get_embeddings.py"
    subprocess.run(["python", python_script, path, reps_path, "-l", str(layer)])
    tcr_reps = pd.read_csv(reps_path)
    tcr_df = tcr_reps.iloc[:,1:].values
    return tcr_df

def get_cluster_output(path):
    with open(path) as file:
        line_num = 0
        start_df = pd.DataFrame()
        for i in file:
            label = line_num
            line_num +=1
            seq_list = i.strip().split(",")
            df = pd.DataFrame(seq_list)
            labels = np.repeat(line_num,len(df))
            df["cluster"] = labels
            start_df = start_df.append(df, ignore_index=True)
        start_df = start_df.rename(columns={0: "Sequence"})
        file.close()
        return start_df
    

class TcrTransformer:
    def __init__(self, seq_path, tcr_reps_path, lower=0.3, upper=0.8, num_eps=10, layer=8):
        #pd.DataFrame(column).to_csv("sequences.txt")
        self.lower = lower
        self.upper = upper
        self.num_eps = num_eps
        self.labeled_reps = get_tcr_reps(seq_path, tcr_reps_path, layer)
        self.epsilon_values = self.get_epsilons(self.labeled_reps)
        self.distances = self.get_distances(self.labeled_reps)
        
    def get_epsilons(self, df):
        distances = pairwise_distances(df, metric="euclidean")
        trimmed = trim_indices(distances)
        median = pd.Series(trimmed).median()
        upper = median *self.upper
        lower = median*self.lower
        range_vals = upper - lower
        increment = range_vals/self.num_eps
        return np.arange(lower, upper, increment)
    
    def get_distances(self,df):
        dist = pairwise_distances(df, metric="euclidean")
        trimmed = trim_indices(dist)
        return trimmed
    
# def list_to_csv(seq):
#     df = pd.DataFrame(seq).to_csv("sequences.txt")
#     print(df)
    
def trim_indices(pw_distances):
    new_pw_distances = []
    for i in range(len(pw_distances)-1):
        arr = list(pw_distances[i][i+1:])
        for j in range(len(arr)):
            new_pw_distances.append(arr[j])
    return new_pw_distances
    
def main(chains, feature, outpath):

    df = pd.read_csv('../../../tcrvalid/data/antigen_reference_tcrs.csv')

    subset_df = make_subset(
        df,
        sources=sources_['both'],
        chains=chains_[chains],
        feature=features_[feature]
    )
    
    test = pd.DataFrame(subset_df.feature.values)
    test.to_csv("new_sequences.txt", index=False, header=False)
    scan_helper = TcrTransformer(
        "new_sequences.txt", 
        "new_tcr_reps.txt",
    )
    all_scores_df = dbscan_loop(
        subset_df, 
        scan_helper.labeled_reps, 
        scan_helper.epsilon_values,
        metric='euclidean',
        eps_format=':.2f'
    )
    
    file_name = "scores_{}_{}".format(
        chains,
        feature
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
        args.outpath
    )
    
    
    
