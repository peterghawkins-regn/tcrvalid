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
sys.path.append('../lib/clusTCR')
sys.path.append('../../..')

from clustcr import Clustering
import os
import pandas as pd
import numpy as np
import time
from tcrvalid.defaults import *

trb_test_pq = data_path_small_trb['te']
te_seq_trb_df = pd.read_parquet(trb_test_pq,columns=['new_meta_vcall','cdr2_cdr3']) #.head(int(1e5))
te_seq_trb_df['cdr3'] = te_seq_trb_df['cdr2_cdr3'].map(lambda x: 'C'+x.split('-')[1]+'F')

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

ns = np.array([10,100,1000,3000,10000,30000,100000])
times_cltcr = np.zeros((len(ns),))
for i,n in enumerate(ns):
    tmp_ins = te_seq_trb_df['cdr3'].sample(n=n).values
    cltcr = Clustering(n_cpus=8)
    t = timed_response(cltcr.fit,tmp_ins,n)
    times_cltcr[i] = t
    
df_timings = pd.DataFrame({
    'n':ns,
    'pair':ns*(ns-1)*0.5,
    'time_clustcr':times_cltcr
})
df_timings.to_csv('../../../results_data/timings/clustcr_cluster_timing.csv')
    
