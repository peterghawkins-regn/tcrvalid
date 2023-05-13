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
from sklearn.cluster import DBSCAN
import pandas as pd
from .metrics import clustering_scoring

def dbscan_loop(df,representations,eps_values,metric,min_samples=3,eps_format=':.2f'):
    """ Perform clustering on TCRs in df using related representation array

    parameters
    -----------

    df: pd.DataFrame
    The dataframe of TCRs

    representations: np.array
    Should be of size [n_tcrs,n_features], where n_tcrs is the number of TCRs in the input 
    dataframe, and n_features is the size of the representation 

    eps_values: np.array
    Values of the distance paramter within dbscan to use for fitting and scoring

    metric: string, optional, default = 'euclidean'
    The distance metruc to use inside dbscan. 'euclidean' and 'manhattan' are key values.

    eps_format: string, optional , default = '.2f'
    String formatting code. default is 2dp, if using small eps, try ':.3f' etc.
    If using ints, you can use ':d'

    returns
    --------

    scores_df : pd.DataFrame
    Dataframe of the scoring of the clustering at the various epsilon values provided.
    Each row is an epsilon value, and has columns for key clustering metrics.

    """
    scores=dict()
    # range over which eps to be varied depends on the model
    for eps in eps_values:
        dbs = DBSCAN(eps=eps,min_samples=min_samples,metric=metric)
        dbs.fit(representations) 

        eps_name = str('eps_{'+eps_format+'}').format(eps)

        df[eps_name] = dbs.labels_ # put cluster labels back into df that goes with representations
        scores[eps_name], _ = clustering_scoring(
          df,
          df,
          epitope_col = 'peptide',
          cluster_col = eps_name
        )

    scores_df = pd.DataFrame(scores).T
    if 'd' in eps_format:
        scores_df['epsilon'] = scores_df.index.map(lambda x: int(x.split('_')[1]))
    else:
        scores_df['epsilon'] = scores_df.index.map(lambda x: float(x.split('_')[1]))

    return scores_df
