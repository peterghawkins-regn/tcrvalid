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

def get_cluster_purity_df(
        df,
        epitope_col='id',
        cluster_col='cluster',
        include_spike=False
    ):
    """ Get statsistics for clustering performance
    
    Assumes that df only contains unique TCRs - drop duplicate clonotypes prior to using this function.
    Will give you statisictcs for each cluster in the data - how pure that cluster is and so on, described further below.
    
    metrics included:
        - cluster_purity : purity of  the cluster - 100% if all TCRs in cluster have same epitope
        - cluster_mode : most common epitope represented in cluster
        - cluster_mode_count : number of TCRs in cluster with the modal epitope
        - cluster_pmhc_count : number of different unique epitopes (pMHCs) in the cluster
        - cluster_size : number of TCRs in the cluster
        - cluster_size_sp0 : (if include_spike=True) number of TCRs from the unspiked data in the cluster
        - cluster_size_sp1 : (if include_spike=True) number of TCRs from the spike-in data in the cluster
    
    parameters:
    ------------
    
    df: Pandas Dataframe
        dataframe of TCRs -  a row per TCR with cluster and epitope labels.
        must have a 'sequence_id' column - unique for each row
        
    epitope_col: str
        name of the column containing epitopes
        
    cluster_col: str
        name of the column containing the cluster labels
        
    include_spike: bool
        If True include metric outputs for how many spiked and unspiked TCRs in each cluster.
        only useful if have spiked and unspiked TCRs in the data, and a column 'spike' with entries 0
        for unspiked and 1 for spiked data must be in the df.

    returns
    ---------
    
    cluster_purity_df: Pandas DataFrame
        dataframe indexed by column with various metrics as columns
    
    """
    
    joint_df = df
    
    def purity(x,epitope_col='id'):
        mode = x[epitope_col].mode()[0]
        return 100*len(x[x[epitope_col]==mode])/len(x) 

    cluster_purity_df = joint_df.groupby(cluster_col)\
            .apply(lambda x : purity(x,epitope_col=epitope_col) )\
            .rename('cluster_purity')\
            .to_frame()

    cluster_mode_df = joint_df.groupby(cluster_col)\
            .apply(lambda x : x[epitope_col].mode()[0] )\
            .rename('cluster_mode')\
            .to_frame()

    cluster_modecount_df = joint_df.groupby(cluster_col)\
            .apply(lambda x : len(x[x[epitope_col]==x[epitope_col].mode()[0]]) )\
            .rename('cluster_mode_count')\
            .to_frame()

    cluster_pepcount_df = joint_df.groupby(cluster_col)\
            .apply(lambda x : len(x[epitope_col].unique()) )\
            .rename('cluster_pmhc_count')\
            .to_frame()

    cluster_size_df = joint_df.groupby(cluster_col)\
            .apply(lambda x : len(x['sequence_id'].unique()) )\
            .rename('cluster_size')\
            .to_frame()
    
    cluster_purity_df = cluster_purity_df.join(
        [
            cluster_mode_df,
            cluster_modecount_df,
            cluster_pepcount_df,
            cluster_size_df
        ]
    )
    
    if include_spike:
        cluster_size0_df = joint_df.groupby(cluster_col)\
                .apply(lambda x : len(x[x['spike']==0]['sequence_id'].unique()) )\
                .rename('cluster_size_sp0')\
                .to_frame()

        cluster_size1_df = joint_df.groupby(cluster_col)\
                .apply(lambda x : len(x[x['spike']==1]['sequence_id'].unique()) )\
                .rename('cluster_size_sp1')\
                .to_frame()

        cluster_purity_df = cluster_purity_df.join(
            [
                cluster_size0_df,
                cluster_size1_df,
            ]
        )
        
    return cluster_purity_df

def clustering_scoring(
        df,
        df_cl, 
        cluster_purity_df=None, 
        epitope_col='id',
        cluster_col='cluster',
    ):
    """ for a given clustering of TCRs, calculate how good the clustering is
    
    parameters
    -----------
    
    df: pandas.Dataframe
        datafrema of all TCRs upon which clustering was performed. can include repeated clones.
        
    df_cl: pandas.DatFrame
        dataframe with unique clonotypes and cluster/epitope labeling where available.
    
    cluster_purity_df: (pandas.Dataframe,None) optional, default=None
        dataframe of metrics of each cluster of the TCRs in df_cl. If None will be calculated using 
        get_cluster_purity_df()
        
    epitope_col: str
        name of column storing binding epitope of each TCR (TCRs not required to have this info)
        
    cluster_col: str
        name of the column containing cluster labeling to use.
        Expect unclustered TCRs to have -1, and clusters to have integer labels >=0 
    
    
    """
    pcnt_clustered = 100.0*\
        len(df_cl[df_cl[cluster_col]>-0.1])\
        /len(df_cl)
    
    pcnt_clustered_in_largest = 100.0*df_cl[cluster_col][df_cl[cluster_col]>-0.1].value_counts().iloc[0]/len(df_cl)
    num_clusters = len(df_cl[df_cl[cluster_col]>-0.1][cluster_col].unique())
    
    #map cluster labeling into all TCR data via merging on clonotype
    # this is only necessary if cluster col is not already in df
    if cluster_col not in df:
        df_use = df.merge(df_cl[['clono_id',cluster_col]],left_on='clono_id',right_on='clono_id',how='left')
    else:
        df_use = df
    
    # get the cluster_purity_df using unique TCRs which do have epitope mapping
    if cluster_purity_df is None:
        cluster_purity_df = get_cluster_purity_df(
            df_use[~df_use[epitope_col].isna()]\
                .drop_duplicates('clono_id'),
            epitope_col=epitope_col,
            cluster_col=cluster_col
        )
        
    cluster_purity_df = cluster_purity_df\
        .sort_values('cluster_size',ascending=False)
    
    if -1 in cluster_purity_df.index:
        cluster_purity_df = cluster_purity_df.drop([-1]) # remove cluster which means "unclustered"
    
    mean_purity = cluster_purity_df.cluster_purity.mean()
    
    n_epitopes_captured = len(cluster_purity_df.cluster_mode.unique())
    
    # % well clustered vs all unique TCRs
    well_clustered_total = 100.0*\
        cluster_purity_df[cluster_purity_df['cluster_purity']>90.0]\
        .cluster_size.sum()\
        /len(df_use[~df_use[epitope_col].isna()].drop_duplicates('clono_id'))
    
    # % well clustered among those unique TCRs which were clustered
    well_clustered = 100.0*\
        cluster_purity_df[cluster_purity_df['cluster_purity']>90.0]\
        .cluster_size.sum()\
        /cluster_purity_df.cluster_size.sum()
    
    # % of all the epitope labelled clones which are clustered
    df_use = df_use[~df_use[epitope_col].isna()]\
                .drop_duplicates('clono_id')
    
    pcnt_labeled_clustered = 100.0*\
        len(df_use[df_use[cluster_col]>-0.1])\
        /len(df_use)
    
    value_dict = {
        'n_clusters':num_clusters,
        'n_epitopes_captured': n_epitopes_captured,
        'percent_clustered':pcnt_clustered,
        'percent_clustered_in_largest':pcnt_clustered_in_largest,
        'percent_labeled_clustered':pcnt_labeled_clustered,
        'mean_purity':mean_purity,
        'well_clustered_total':well_clustered_total,
        'well_clustered':well_clustered,
    }
    
    return value_dict,cluster_purity_df