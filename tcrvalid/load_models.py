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
import tensorflow.keras as keras
import mlflow
import pathlib
import os

from .defaults import *

__all__ = [
    'mlflow_encoders_TRA',
    'mlflow_encoders_TRB',
    'keras_encoders_TRA',
    'keras_encoders_TRB',
    'load_logged_models',
    'load_named_models'
]

#keras_model_base_path = '/dbfs/mnt/dev-zone0/user/peter.hawkins/projects/TCR_VAE/logged_models/'
keras_model_names = []
for p in pathlib.Path(keras_model_base_path).glob('*/*/encoder_kr'):
    strp = str(p)
    name = os.sep.join(strp.split(os.sep)[-3:-1])
    #print(name)
    keras_model_names.append(name)  
    
mlflow_encoders_TRA = {
    '1_2': 'runs:/db09d9a9dc75472d88255acff170686d/encoder'
}
mlflow_encoders_TRB = {
    '0_0': 'runs:/745b58a62e4d44d7a557c25d2396197b/encoder',
    '1_1': 'runs:/d605574cb9534b46ae6f3aefefee3aa4/encoder',
    '1_2': 'runs:/1832da0dd8d442ce80e8e4e5adec9136/encoder',
    '1_5': 'runs:/d84d7c463ac747c3bf274238af4b7b58/encoder',
    '1_10': 'runs:/0d73b60c6e2840da825806f61a0e3a47/encoder',
    '1_20': 'runs:/e605f2b46e7e4e5f9b1f28f00bb7d3a6/encoder',
    '1_2_full': 'runs:/f15881586e134682a092b5072d637063/encoder',
    '1_2_full_40': 'runs:/baff9b5b867d46ceaf2ac70d5c61e1df/encoder'
}

mlflow_decoders_TRA = dict()
for k,v in mlflow_encoders_TRA.items():
    mlflow_decoders_TRA[k] = '/'.join(v.split('/')[:-1])+'/decoder'    

mlflow_decoders_TRB = dict()
for k,v in mlflow_encoders_TRB.items():
    mlflow_decoders_TRB[k] = '/'.join(v.split('/')[:-1])+'/decoder'     

keras_encoders_TRA = {}
keras_decoders_TRA = {}
keras_encoders_TRB = {}
keras_decoders_TRB = {}
for name in keras_model_names:
    if name.startswith('TRA'):
        keras_encoders_TRA[name.split('/')[1]] = os.path.join(keras_model_base_path,name,'encoder_kr') #keras.models.load_model(base_path+name+'/encoder_kr')
        keras_decoders_TRA[name.split('/')[1]] = os.path.join(keras_model_base_path,name,'decoder_kr')
    else:
        keras_encoders_TRB[name.split('/')[1]] = os.path.join(keras_model_base_path,name,'encoder_kr')
        keras_decoders_TRB[name.split('/')[1]] = os.path.join(keras_model_base_path,name,'decoder_kr')


def load_logged_models(model_dict,as_keras=True):
    """ Load a dictionary of named models and paths into models dictionary

    parameters
    ----------
    model_dict: dictionary
        A dictionary with model names as keys and either:
         - paths to load keras models as values
         - paths to load MLFlow models from as values.
        
    as_keras: optional, bool, default = False
        If true load keras model. Else load mlflow model (default)

    returns
    -------
    loaded_models: dictionary
        A dictionary keyed my model names and with vlaues as mlflow models
    """
    loaded_models = dict()
    if not as_keras:
        for k,v in model_dict.items():
            loaded_models[k] = mlflow.pyfunc.load_model(model_uri=v)
    else:
        for k,v in model_dict.items():
            loaded_models[k] = keras.models.load_model(v)  
    return loaded_models

def load_named_models(model_names,chain='TRB',as_keras=True, encoders=True):
    """ Load a dictionary of named models and paths into models dictionary

    parameters
    ----------
    model_dict: dictionary
        A dictionary with model names as keys and either:
         - paths to load keras models as values
         - paths to load MLFlow models from as values.
        
    as_keras: optional, bool, default = False
        If true load keras model. Else load mlflow model (default)
        
    encoders: optional, bool, default=True
        If true load encoder models (default), else load decoders

    returns
    -------
    loaded_models: dictionary
        A dictionary keyed my model names and with vlaues as mlflow models
    """
    loaded_models = dict()
    
    if encoders:
        if chain=='TRB':
            if not as_keras:
                model_dict = {k:v for k,v in mlflow_encoders_TRB.items() if k in model_names}
            else:
                model_dict = {k:v for k,v in keras_encoders_TRB.items() if k in model_names}
        else:
            if not as_keras:
                model_dict = {k:v for k,v in mlflow_encoders_TRA.items() if k in model_names}
            else:
                model_dict = {k:v for k,v in keras_encoders_TRA.items() if k in model_names}
    else:
        if chain=='TRB':
            if not as_keras:
                model_dict = {k:v for k,v in mlflow_decoders_TRB.items() if k in model_names}
            else:
                model_dict = {k:v for k,v in keras_decoders_TRB.items() if k in model_names}
        else:
            if not as_keras:
                model_dict = {k:v for k,v in mlflow_decoders_TRA.items() if k in model_names}
            else:
                model_dict = {k:v for k,v in keras_decoders_TRA.items() if k in model_names}
            
    
    if not as_keras:
        for k,v in model_dict.items():
            loaded_models[k] = mlflow.pyfunc.load_model(model_uri=v)
    else:
        for k,v in model_dict.items():
            loaded_models[k] = keras.models.load_model(v)  
    return loaded_models




