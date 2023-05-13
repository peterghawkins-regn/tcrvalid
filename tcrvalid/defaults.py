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
""" default paths to data, models, output directories etc

One should adjust these depending on their install.
We are setting some of these to repo specific locations
"""
import os

__all__ = [
    'keras_model_base_path',
    'data_path_full_trb_spark',
    'data_path_small_trb_spark',
    'data_path_full_tra_spark',
    'data_path_full_trb',
    'data_path_small_trb',
    'data_path_full_tra',
    'trbv_ref_path',
    'labelled_data_path',
    'figure_output_dir'
]

#DB_base = '/dbfs/mnt/dev-zone0/user/peter.hawkins/projects/TCR_VAE/'
#JH_base = '/data/home/peter.hawkins/projects/tcrvalid_testing/tcrvalid_'

# CHANGE TO SYSTEM CHOICE
#base = JH_base
base = os.path.dirname(os.path.abspath(__file__))

keras_model_base_path = os.path.join(base,'logged_models')

# note that now spark and regular are same paths but leave for consistency
data_path_full_trb_spark = {case:os.path.join(base,'data','tts_full_TRB_'+case) for case in ['tr','va','te']}
data_path_small_trb_spark = {case:os.path.join(base,'data','tts_TRB_'+case) for case in ['tr','va','te']}
data_path_full_tra_spark = {case:os.path.join(base,'data','tts_full_TRA_'+case) for case in ['tr','va','te']}

data_path_full_trb = {case:os.path.join(base,'data','tts_full_TRB_'+case) for case in ['tr','va','te']}
data_path_small_trb = {case:os.path.join(base,'data','tts_TRB_'+case) for case in ['tr','va','te']}
data_path_full_tra = {case:os.path.join(base,'data','tts_full_TRA_'+case) for case in ['tr','va','te']}

trbv_ref_path = os.path.join(base,'data','TRBV_reference.csv')

labelled_data_path = os.path.join(base,'data','antigen_reference_tcrs.csv')

figure_output_dir = os.path.join(base,'figoutput_121523')