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
""" Perform many latent traversals in latent space and track model smoothness """
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns

import sys

from tcrvalid.load_models import *
from tcrvalid.plot_utils import set_simple_rc_params
from tcrvalid.physio_embedding import SeqArrayDictConverter,manhattan
from tcrvalid.interpolation import *
from tcrvalid.defaults import *

set_simple_rc_params()

pull_using_spark=False
from_keras=True

# ---------- Load a subset of the data ------------------------------------------
n_use = 100000
cols = ['new_meta_vcall','cdr2_cdr3']
if pull_using_spark:
    sdf=spark.read.format("parquet").load(data_path_small_trb_spark['te'])
    te_seq_df = sdf.select(*cols).limit(n_use).toPandas()
else:
    te_seq_df = pd.read_parquet(data_path_small_trb['te'], columns=cols).head(n_use)

te_seq_df['cdr3'] = te_seq_df['cdr2_cdr3'].map(lambda x: x.split('-')[1])

# ----------  Set up reference library and distance trackers --------------------
ref_cdr2s,ref_cdr2_seqs = get_ref_cdr2s(trbv_ref_path)
ref_cdr2_pairs = get_ref_cdr2_pairs(trbv_ref_path,mapping=None)

cdr2_d_tracker = CDR2DistanceTracker(ref_cdr2s,6,manhattan)
cdr3_d_tracker = CDR3DistanceTracker(None,manhattan)

tracker_tuples = [
  (cdr2_d_tracker,mean_statistic,'CDR2'),
  (cdr3_d_tracker,mean_statistic,'CDR3'),
  (cdr3_d_tracker,mean_startendnorm_statistic,'CDR3_norm')
]

mapping = SeqArrayDictConverter()

# --------------------------------------------------------------------------------
# meta-interpolation object helps to perform the many latent traversals and track 
# the requested properties on each traversal
mi = MetaInterpolation(mapping,tracker_tuples)

# --------------------------------------------------------------------------------
# the models we want to perform analysis on
model_names =  [
    '10_1',
    '1_2',
    '1_5',
    '1_10',
    '0_0',
]
loaded_models = load_named_models(model_names,chain='TRB',as_keras=from_keras)
loaded_decoders = load_named_models(model_names,chain='TRB',as_keras=from_keras,encoders=False)

# --------------------------------------------------------------------------------
# create a list of tcr pairs to perform latent traversals between
# for each traversal pair, the distances requested will be recorded
# Here use 30 CDR3s and 10 random pairs of CDR2 per CDR3.
results=[]
tcr_pair_list = list(tcr_pair_gen(te_seq_df,ref_cdr2_pairs,30,10))

print('going to interpolate {} pairs of tcrs'.format(len(tcr_pair_list)))

# --------------------------------------------------------------------------------
# for each model - track distance on all TCR-pair traversals and store in a dataframe
for k in loaded_models.keys():
    print(k)
    #10 points on each interpolation
    r_df = mi(
        tcr_pair_list,
        loaded_models[k],
        loaded_decoders[k],
        10 
    )
    r_df['model'] = k
    results.append(r_df)
    
# --------------------------------------------------------------------------------
# box-plot of the smoothness metrics over the many trials for each model
boxcols = {
    'boxprops':{'facecolor':'none', 'edgecolor':'k'},
    'medianprops':{'color':'k'},
    'whiskerprops':{'color':'k'},
    'capprops':{'color':'k'}
}

pl_df = pd.concat(results)

model_names_d = {
  '0_0': 'AE',
  '10_1': r'$\beta=10$'+'\n'+r'$C=1$',
  '1_2': r'$\beta=1$'+'\n'+r'$C=2$',
  '1_5': r'$\beta=1$'+'\n'+r'$C=5$',
  '1_10': r'$\beta=1$'+'\n'+r'$C=10$',
}

pl_df.model = pl_df.model.map(model_names_d)

order = ['$\\beta=10$\n$C=1$', '$\\beta=1$\n$C=2$', '$\\beta=1$\n$C=5$','$\\beta=1$\n$C=10$', 'AE']

f,axes = plt.subplots(2,1,figsize=(3.5,3.5))

sns.boxplot(
  data = pl_df,
  x='model',
  y='CDR2',
  ax=axes[0],
  order=order,
  **boxcols
)
axes[0].set_ylabel(r'$C_{CDR2}$')

sns.boxplot(
  data = pl_df,
  x='model',
  y='CDR3_norm',
  ax=axes[1],
  order=order,
  **boxcols
)
axes[1].set_ylabel(r'$D_{CDR3,norm}$')

axes[0].set_xticklabels([])
axes[0].set_xlabel('')

plt.tight_layout()

#plt.savefig('metainterpolation_TRB.pdf')


