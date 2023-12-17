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
from tcrclustering.cluster.base import BaseTCRClusterer, Options
from tcrclustering.cluster.utils import load_json_to_dict, AA_TO_NUC
import os
import pandas as pd
import subprocess
import shutil
import pickle
import numpy as np
from tcrclustering.parse.reference import ReferenceAAMapper

class TCRDistOptions(Options):
    
    def __init__(self,
                 conda_prefix,
                 tcrdist_run_script_path,
                 env,
                 standard_env,
                 output_path,
                 tmp_dir = './tmp',
                 organism='human',
                 chains='AB',
                 threshold=None,
                 gap_penalty=-6,
                 gap_number=0,
                 subject_col='SubjectID',
                 from_raw_nt_seqs=False,
                 cluster_mode = 'AS_SVG'
                ):
        """
        
        
        """
        super(TCRDistOptions,self).__init__()
        self.conda_prefix = conda_prefix
        self.script_path = tcrdist_run_script_path
        self.env = env
        self.standard_env = standard_env
        self.output_path = output_path
        self.tmp_dir = tmp_dir
        self.gap_penalty = gap_penalty
        self.gap_number = gap_number
        self.subject_col = subject_col
        self.organism = organism
        self.chains = chains
        if threshold is None:
            self.threshold = 0
        else:
            self.threshold = threshold
        self.from_raw_nt_seqs = from_raw_nt_seqs
        
        self.tmp_filename='tmp_tcrdist_data.tsv'
        self.name = 'tcrdist'
        self.cluster_mode = cluster_mode
        
    @classmethod
    def from_json(cls,json_path):
        json_dict = load_json_to_dict(json_path)
        args = [
             'conda_prefix',
             'path',
             'env',
             'standard_env',
             'output_path',
        ]
        inargs = [json_dict[a] for a in args]
        kwargs = [
            'tmp_dir',
            'threshold',
            'gap_penalty',
            'gap_number',
            'use_v'
        ]
        inkwargs = {kw:json_dict[kw] for kw in kwargs}
        out_opts = cls(*inargs,**inkwargs)
        out_opts.add_note = json_dict['note']
        return out_opts


class TCRDistClusterer(BaseTCRClusterer):
    
    def __init__(self,
                 options
                ):
        super(TCRDistClusterer,self).__init__(options)
        self.check_create_tmpdir()
        self.distance_approach=True
        self.dist_ending = self.options.chains
        if self.options.chains=='AB':
            self.cluster_ending = self.options.chains+'_A'
        else:
            self.cluster_ending = self.options.chains+'_'+self.options.chains
        
    def check_create_tmpdir(self):
        if not os.path.exists(self.options.tmp_dir):
            os.makedirs(self.options.tmp_dir)
        return None
    
    def _get_columns_keep(self):
        columns_keep=['sequence_id',
                      'seq_nt_TRA',
                      'seq_nt_TRB',
                     ]
        if self.options.subject_col is not None:
            columns_keep.append(self.options.subject_col)
        return columns_keep
    
    def _construct_tmp_df(self,df):
        if self.options.from_raw_nt_seqs:
            columns_keep=self._get_columns_keep()
            df = df[columns_keep]
            df = df.dropna()
            df['epitope'] = 'epitope'
            if self.options.subject_col is not None:
                df['subject'] = df[self.options.subject_col]
            else:
                df['subject'] = 'subject'

            df = df.rename(columns={
                'sequence_id':'id',
                'seq_nt_TRA':'a_nucseq',
                'seq_nt_TRB':'b_nucseq'
            })
            df = df[[
                'id',
                'epitope',
                'subject',
                'a_nucseq',
                'b_nucseq'
            ]]
        else:
            df['members'] = df.sequence_id
            df['epitope'] = 'epitope'
            if self.options.subject_col is not None:
                df['subject'] = df[self.options.subject_col]
            else:
                df['subject'] = 'subject'
                
            reference_mappers = {
                'TRB' : ReferenceAAMapper(chain='TRB'),
                'TRA' : ReferenceAAMapper(chain='TRA')
            }
            for lc_name,locus in zip(['a','b'],['TRA','TRB']):
                df['v'+lc_name+'_gene'] = df['v_gene_'+locus].map(lambda x : reference_mappers[locus].collect_v_name(x))
                df['j'+lc_name+'_gene'] = df['j_gene_'+locus].map(lambda x : reference_mappers[locus].collect_j_name(x))
                df['v'+lc_name+'_genes'] = df['v'+lc_name+'_gene']
                df['j'+lc_name+'_genes'] = df['j'+lc_name+'_gene']
                df = df.dropna(subset=['v'+lc_name+'_gene']) #,'j'+lc_name+'_gene']) #j isn't 'lookup'`d so shouldn't matter
                df['j'+lc_name+'_gene'] = df['j'+lc_name+'_gene'].fillna('unknown')
                df['v'+lc_name+'_countreps'] = df['v'+lc_name+'_gene'].map(lambda x : x.split('*')[0])
                df['j'+lc_name+'_countreps'] = df['j'+lc_name+'_gene'].map(lambda x : x.split('*')[0],na_action='ignore')
               
                
            print('TCRdist processing to clone file complete, len=',len(df))
            df['a_protseq_prob'] = 1
            df['b_protseq_prob'] = 1
            
            df = df.rename(columns={
                'sequence_id':'clone_id',
                'cdr3_TRB' : 'cdr3b',
                'cdr3_TRA' : 'cdr3a',       
            })
            df['clone_size']=1
            
            df = df[df['cdr3a'].map(lambda x: len(x)>5)]
            df = df[df['cdr3b'].map(lambda x: len(x)>5)]
            
            df = df[[
                'clone_id',
                'epitope',
                'subject',
                'va_gene',
                'ja_gene',
                'vb_gene',
                'jb_gene',
                'va_genes',
                'ja_genes',
                'vb_genes',
                'jb_genes',
                'va_countreps',
                'ja_countreps',
                'vb_countreps',
                'jb_countreps',
                'cdr3a',
                'cdr3b',
                'a_protseq_prob',
                'b_protseq_prob',
                'members',
                'clone_size'
            ]]
        return df
    
    def _df_to_tmp(self,df):
        df = self._construct_tmp_df(df)
        if self.options.from_raw_nt_seqs:
            out_path = os.path.join(self.options.tmp_dir,self.options.tmp_filename)
        else:
            out_path = os.path.join(
                self.options.tmp_dir,
                self.options.tmp_filename.split('.')[0]+'_parsed_seqs_probs_mq20_clones.tsv'
            )
        df.to_csv(
            out_path,
            sep='\t',
            header=True,
            index=False
        )
        return out_path
    
    def build_cmd(
                        self,
                        data_path,
                        output_path
                        ):
        cmd = 'python '
        cmd += self.options.script_path
        cmd += ' --organism '
        cmd += self.options.organism
        cmd += ' --dist_chains '
        cmd += self.options.chains
        cmd += ' --radius_factor '
        cmd += '{:.1f}'.format(self.options.threshold)
        if self.options.from_raw_nt_seqs:
            cmd += ' --pair_seqs_file '
            cmd += data_path
        else:
            cmd += ' --clones_file '
            cmd += data_path
        cmd += ' --webdir '
        cmd += output_path
        cmd += ' --make_fake_quals '
        cmd += ' --force '
        cmd += ' --no_probabilities '
        cmd += ' --intrasubject_nbrdists ' 
        cmd += ' --constant_seed ' 
        #cmd += ' --dry_run'
        
        cmd = cmd.strip()
        return cmd
    
    def _tmp_to_output(self):
        files = [
            '_parsed_seqs_probs_mq20_clones.tsv',
            '_parsed_seqs_probs_mq20_clones_{}.dist'.format(self.dist_ending),
            '_parsed_seqs_probs_mq20_clones_grouped_clusters_{}.pickle'.format(self.cluster_ending),
            '_parsed_seqs_probs_mq20_clones_mini_clusters_{}.pickle'.format(self.cluster_ending)
        ]
        files = [self.options.tmp_filename.split('.')[0]+f for f in files]
        for f in files:
            path_src = os.path.join(self.options.tmp_dir,f)
            path_dest = self.options.output_path
            # prefer move, but there can be weird overwrite semantics, cp seems safer
            shutil.copy(path_src,path_dest)
            
    def _clear_out_files(self):
        files = [
            '_parsed_seqs_probs_mq20_clones.tsv',
            '_parsed_seqs_probs_mq20_clones_{}.dist'.format(self.dist_ending),
            '_parsed_seqs_probs_mq20_clones_grouped_clusters_{}.pickle'.format(self.cluster_ending),
            '_parsed_seqs_probs_mq20_clones_mini_clusters_{}.pickle'.format(self.cluster_ending)
        ]
        files = [self.options.tmp_filename.split('.')[0]+f for f in files]
        tmp_files = [os.path.join(self.options.tmp_dir,f) for f in files]
        out_files = [os.path.join(self.options.output_path,f) for f in files]
        for pa in tmp_files:
            if os.path.exists(pa):
                os.remove(pa)
        for pa in out_files:
            if os.path.exists(pa):
                os.remove(pa)
    
        
    def _pull_in_clone_table(self,path_name):
        clone_table = pd.read_table(
                            os.path.join(
                                self.options.output_path,
                                path_name+'clones.tsv'
                                ),
                           index_col=0,
                           sep='\t',
                           )

        clone_table = clone_table.rename(columns={'members':'sequence_id'})
        # note: clone table could have duplicates...
        return clone_table
            
    def _pull_in_cluster_df(self,path_name):
        if self.options.cluster_mode=='AS_SVG':
            ab_cl_fpath = os.path.join(
                self.options.output_path,
                path_name+'clones_grouped_clusters_{}.pickle'.format(
                    self.cluster_ending
                )
            )
            with open(ab_cl_fpath,'rb') as fp:
                cl_data = pickle.load(fp)
        else:
            ab_cl_fpath = os.path.join(
                self.options.output_path,
                path_name+'clones_mini_clusters_{}.pickle'.format(
                    self.cluster_ending
                )
            )
            with open(ab_cl_fpath,'rb') as fp:
                cl_data = pickle.load(fp)

        clusters_ = []
        clones_ = []
        for idx,mems in enumerate(cl_data):
            clusters_.extend( [idx]*len(mems) )
            clones_.extend( mems )

        clone_cluster_df = pd.DataFrame(data={'clone_id':clones_,'cluster':clusters_})#.groupby('cluster').count().sort()

        clone_counts = clone_cluster_df.groupby('cluster')\
            .count()\
            .rename(columns={'clone_id':'count'})\
            .sort_values('count',ascending=False)
        cluster_size_map = {clone:count for clone, count in zip(clone_counts.index,clone_counts['count'])}
        
        def choose_biggest_cluster(x):
            sizes = [cluster_size_map[y] for y in x]
            idx_best = np.argmax(sizes)
            return x[idx_best]

        clone_to_cluster_df = clone_cluster_df.groupby('clone_id').apply(lambda x: list(x['cluster'])).rename('all_clusters').to_frame()
        clone_to_cluster_df['cluster'] = clone_to_cluster_df['all_clusters'].map(lambda x: x[0] if len(x)==1 else choose_biggest_cluster(x))

        clone_to_cluster_df = clone_to_cluster_df.reset_index()
        return clone_to_cluster_df
    
    def _pull_in_results(self,path_name):
        cluster_df = self._pull_in_cluster_df(path_name)
        clone_df = self._pull_in_clone_table(path_name).reset_index()
        outdf = clone_df.merge(
            cluster_df,
            left_on='clone_id',
            right_on='clone_id',
            how='outer'
        )
        return outdf
    
    def _run_clustering(self,
                        df,
                        barcode_column='barcode',
                        clono_column='clono_id',
                        save_to_output=True
                       ):
        
        save_to_output = True
        
        self._clear_out_files()
        
        data_path = self._df_to_tmp(df)
        outpath = self.options.output_path
       
        if not os.path.exists(self.options.output_path):
            os.makedirs(self.options.output_path)
            
        if not outpath.endswith('/'):
            outpath = outpath+'/'
            
        cmd = self.build_runtime_cmd(data_path,outpath) 
        self._run_cmd(cmd)
        self._tmp_to_output()
        return None
    
    def _assign_cluster_info(self,df_out,df,clono_column):
        # only can be used on refenerence if input df already had clones removed
        clono_cluster_map = {
            clono:cl for clono,cl in zip(df_out['sequence_id'],df_out['cluster'])
        }
        df['cluster'] = (df['sequence_id']).map(clono_cluster_map)
        cl_size_map = {v:c for v,c in df_out['cluster'].value_counts().iteritems()}
        df['cluster_size_orig'] = df['cluster'].map(cl_size_map)
        
        return df
    
    def apply_clustering(self,
                         df,
                         clono_column
                        ):
        expected_name = self.options.tmp_filename.split('.')[0]+'_parsed_seqs_probs_mq20_'
        df_out = self._pull_in_results(expected_name)
        df_cl = self._assign_cluster_info(df_out,df,clono_column)
        return df_cl
    