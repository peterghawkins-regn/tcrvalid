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
from setuptools import setup, find_packages
from glob import glob
import os

    
lm_full = glob('tcrvalid/logged_models/**/*', recursive=True)
lm_f = [f for f in lm_full if not os.path.isdir(f)]
model_paths = [os.sep.join(p.split(os.sep)[1:]) for p in lm_f]
print(model_paths)

tcrvalid_data=[
    'data/*.csv',
    'data/tts_full_TRA_te/*',
    'data/tts_TRB_te/*',
]+model_paths

print(tcrvalid_data)

setup(
    name="tcrvalid",
    author="Regeneron Pharmaceuticals",
    description="smooth and disentangled TCR representations for clustering and classification",
    version="0.0.1",
    #packages=find_packages(),
    packages=['tcrvalid'],
    package_dir={'tcrvalid': 'tcrvalid'},
    package_data={
        'tcrvalid': tcrvalid_data
    },
    install_requires=[
        'matplotlib==3.4.2',
        'mlflow-skinny==1.24.0',
        'numba==0.55.1',
        'pandas==1.2.4',
        'scikit-learn==0.24.1',
        'scipy==1.6.2',
        'seaborn==0.11.1',
        'tensorflow==2.8.0',
        'protobuf==3.17.2',
        'biopython==1.76',
        'weblogo==3.7.12',
        'datasets==2.7.1',
        'Pillow==8.2.0',
        'umap-learn==0.5.3'
    ],

)