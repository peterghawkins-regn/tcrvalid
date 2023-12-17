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
from tcrclustering.cluster.utils import load_json_to_dict
from tcrclustering.parse.reference import ReferenceAAMapper
import os
import pandas as pd
import subprocess
from itertools import product
from scipy.sparse import coo_matrix


class IsmartOptions(Options):
    
    def __init__(self,
                 conda_prefix,
                 path,
                 env,
                 standard_env,
                 output_path,
                 tmp_dir = './tmp',
                 threshold=7.5,
                 gap_penalty=-6,
                 gap_number=1,
                 keep_dists=False,
                 use_v=True,
                ):
        super(IsmartOptions,self).__init__()
        self.conda_prefix = conda_prefix
        self.path = path
        self.env = env
        self.standard_env = standard_env
        self.output_path = output_path
        self.tmp_dir = tmp_dir
        self.threshold = threshold
        self.gap_penalty = gap_penalty
        self.gap_number = gap_number
        self.use_v = use_v
        self.keep_dists=keep_dists
        
        self.tmp_filename='tmp_ismart_data.txt'
        self.name = 'ismart'
        
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


class IsmartClusterer(BaseTCRClusterer):
    
    def __init__(self,
                 options
                ):
        super(IsmartClusterer,self).__init__(options)
        self.columns_keep = self._select_out_columns()
        self.check_create_tmpdir()
        self.distance_approach = True 
        
    def check_create_tmpdir(self):
        if not os.path.exists(self.options.tmp_dir):
            os.makedirs(self.options.tmp_dir)
        return None
    
    def _select_out_columns(self):
        columns_keep=['cdr3_TRB']
        if self.options.use_v:
            columns_keep+=['v_gene_TRB']
        return columns_keep
    
    def _df_to_tmp_ismart(self,df):
        df = df[self.columns_keep] 
        
        if self.options.use_v:
            ref_map = ReferenceAAMapper(chain='TRB')
            df['v_gene_TRB'] = df['v_gene_TRB'].map(lambda x : ref_map.collect_v_name(x))
        
        df = df.dropna()
        out_path = os.path.join(self.options.tmp_dir,self.options.tmp_filename)
        df.to_csv(
            out_path,
            sep='\t',
            header=False,
            index=False
        )
        return out_path
    
    def build_cmd(
                        self,
                        data_path,
                        output_path
                        ):
        cmd = 'python '
        cmd += self.options.path
        cmd += ' -f '
        cmd += data_path
        cmd += ' -o '
        cmd += output_path
        cmd += ' -t '
        cmd += '{:.1f}'.format(self.options.threshold)
        cmd += ' -g '
        cmd += '{:d}'.format(self.options.gap_penalty)
        cmd += ' -n '
        cmd += '{:d}'.format(self.options.gap_number)
        if not self.options.use_v:
            cmd += ' -v '
        if self.options.keep_dists:
            cmd += ' --KeepPairwiseMatrix '
        cmd = cmd.strip()
        return cmd
    
    def _pull_in_results(self,path):
        df = pd.read_table(
                    path,
                    header=0,
                    names=self.columns_keep+['cluster']
            )
        return df
    
    def _get_outfile_name(self):
        expected_output_name = self.options.tmp_filename.split('.')[0]+'_ClusteredCDR3s_'+str(self.options.threshold)+'.txt'
        return expected_output_name 
    
    def _clear_out_files(self):
        expected_output_name = self._get_outfile_name()
        pa = os.path.join(
            self.options.output_path, 
            expected_output_name
        )
        if os.path.exists(pa):
            os.remove(pa)
        return None
    
    def _run_clustering(self,
                        df,
                        barcode_column='barcode',
                        clono_column='clono_id',
                        save_to_output=True
                       ):
        
        save_to_output = True
        #print("Warning iSMART requires save_to_output=True, being set to True")
        
        self._clear_out_files()
        
        data_path = self._df_to_tmp_ismart(df)
        outpath = self.options.output_path
       
        if not os.path.exists(self.options.output_path):
            os.makedirs(self.options.output_path)
            
        if not outpath.endswith('/'):
            outpath = outpath+'/'
            
        cmd = self.build_runtime_cmd(data_path,outpath) 
        self._run_cmd(cmd)
        return None
    
    def _assign_cluster_info(self,df_out,df,clono_column):
        if not self.options.use_v:
            cdr3_cluster_map = {
                cdr:cl for cdr,cl in zip(df_out['cdr3_TRB'],df_out['cluster'])
            }
            df['cluster'] = df['cdr3_TRB'].map(cdr3_cluster_map)
        else:
            ref_map = ReferenceAAMapper(chain='TRB')
            df['v_gene_TRB_mapped'] = df['v_gene_TRB'].map(lambda x : ref_map.collect_v_name(x))
            vcdr3_cluster_map = {
                (v,cdr):cl for v,cdr,cl in zip(df_out['v_gene_TRB'],df_out['cdr3_TRB'],df_out['cluster'])
            }
            # print(vcdr3_cluster_map)
            df['cluster'] = df.apply(
                lambda x: vcdr3_cluster_map.get((x['v_gene_TRB_mapped'],x['cdr3_TRB'])),
                axis=1
            )
        
        
        cl_size_map = {v:c for v,c in df_out['cluster'].value_counts().iteritems()}
        df['cluster_size_orig'] = df['cluster'].map(cl_size_map)
        
        return df
    
    def apply_clustering(self,
                         df,
                         clono_column
                        ):
        expected_output_name = self._get_outfile_name()
        df_out = self._pull_in_results(
                        os.path.join(
                            self.options.output_path, 
                            expected_output_name
                        )
        )
       
        df_cl = self._assign_cluster_info(df_out,df,clono_column)
        return df_cl
    
