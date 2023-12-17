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
from abc import ABC,abstractmethod,abstractclassmethod
import pandas as pd
import json
import os
import subprocess
from umap import UMAP
from sklearn.decomposition import PCA

class BaseTCRClusterer(ABC):
    
    def __init__(self,options):
        self.options = options
        self.distance_matrix = None
        self.features = None
        self.internal_ids = None # seq ids of the rows/cols of distance
        self.internal_ids_name = None
        self.distance_approach = None # to be bool in concrete
        self.df_run = None
        
    def _collapse_to_unique(self,
                           df,
                           clono_column='clono_id'
                          ):
        df_uniq = df.drop_duplicates(subset=[clono_column])
        return df_uniq
        
    @abstractmethod 
    def _run_clustering(self,
                        df,
                        barcode_column='barcode',
                        clono_column='clono_id',
                        save_to_output=True
                       ):
        pass
    
    @abstractmethod    
    def apply_clustering(self,
                         df,
                         clono_column
                        ):
        pass
    
    def build_conda_init_cmd(self):
        cmd = 'source '
        cmd += self.options.conda_prefix
        cmd += '/etc/profile.d/conda.sh'
        return cmd
    
    def build_conda_activation_cmd(self):
        cmd = 'conda activate '
        cmd += self.options.env
        return cmd
    
    def build_runtime_cmd(
                        self,
                        data_path,
                        output_path
                        ):
        cmd = self.build_conda_init_cmd()
        cmd += ' && '
        cmd += self.build_conda_activation_cmd()
        cmd += ' && '
        cmd += self.build_cmd(data_path,output_path)
        return cmd
    
    def _run_cmd(self,cmd):
        print("run : ", cmd)
        proc = subprocess.Popen(cmd,
                         shell=True, 
                         stdout=subprocess.PIPE, 
                         stderr=subprocess.PIPE 
                        )
        res=proc.communicate()
        if len(res[0])<1 or len(res[1])>0:
            print("Warning : might be an error here")
            print(res[0])
            print(res[1])
        return None
        
    def run_simple_clustering(self,
        df,
        clono_column='clono_id',
        adjust_df_inplace=False
    ):
        if not adjust_df_inplace:
            df_in = df.copy()
        else:
            df_in = df
        self._run_clustering(
            df_in,
            clono_column=clono_column,
            save_to_output=True
        )
        df_cl = self.apply_clustering(df_in,clono_column)
        df_cl.cluster.fillna(-1, inplace=True)
        return df_cl
            
        
class Options(ABC):
    
    def __init__(self):
        self.name='None'
        self.note=dict()
        
    def add_note(self,note):
        """ add a note to the options for later checking details
        
        Maybe this could be a dict, nested json on output, etc
        
        parameters
        ----------
        
        note: dict of str: (str or int)
        
        """
        self.note.update(note)
    
    def to_json(self,path):
        outpath = os.path.join(path,self.name+'_options.json')
        with open(outpath,'w') as f:
            json.dump(self.__dict__, f)
    
    #TODO
    @abstractclassmethod
    def from_json(cls,json_path):
        pass
    
    def __repr__(self):
        return '\n'.join([k+' : '+str(v) for k,v in self.__dict__.items()])