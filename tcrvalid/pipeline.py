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
import tensorflow as tf
from datasets import load_dataset, Dataset
import pathlib

from .physio_embedding import SeqArrayDictConverter

mapping = SeqArrayDictConverter()

def mapping_to_physio_no_weights(tensor_of_seqs):
    seqs = tensor_of_seqs.numpy()
    seqs = [s.decode('utf-8') for s in seqs]
    x = mapping.seqs_to_array(seqs, maxlen=100)
    x_tf = tf.convert_to_tensor(x,dtype=tf.float16)
    return (x_tf, x_tf) 

# note assumes size - like this for speed
def fast_mapping_to_physio_no_weights(tensor_of_seqs):
    seqs = tensor_of_seqs.numpy()
    seqs = [s.decode('utf-8') for s in seqs]
    x = mapping.fast_seqs_to_array(seqs,28)
    x_tf = tf.convert_to_tensor(x,dtype=tf.float16)
    #print(len(x)) #print works as wrapped in py_func - can use to check # of calls to this fun
    # i.e to check how much data collected in advance
    return (x_tf, x_tf)

def mapping_to_physio_insert_weights(tensor_of_seqs):
    pass

def fast_mapping_to_physio_insert_weights(tensor_of_seqs,tensor_of_sampleW):
    seqs = tensor_of_seqs.numpy()
    #seqs2 = tensor_of_seqs2.numpy()
    #seqs2=seqs2.astype(float)
    seqs = [s.decode('utf-8') for s in seqs]
    x = mapping.seqs_to_array(seqs, maxlen=28)

    #sample_weight = tf.convert_to_tensor(seqs2,dtype=tf.float32)
    x_tf = tf.convert_to_tensor(x,dtype=tf.float16)

    #print(len(x)) #print works as wrapped in py_func - can use to check # of calls to this fun
    # i.e to check how much data collected in advance
    return (x_tf,x_tf,tensor_of_sampleW)
    #return (x_tf, x_tf) # input, and output are same for VAE
    
def reshape_w_sampleW(tensor_of_seqs,tensor_of_sampleW):
    #x_tf =  tf.reshape(tensor_of_seqs,[tf.shape(tensor_of_seqs)[0], 8, 28])
    x_tf =  tf.reshape(tensor_of_seqs,[tf.shape(tensor_of_seqs)[0], 28, 8])

    #print(len(x_tf)) #print works as wrapped in py_func - can use to check # of calls to this fun
    # i.e to check how much data collected in advance
    return (x_tf,x_tf,tensor_of_sampleW)

def get_parquet_dataset(pq_dir):
    files = []
    for p in pathlib.Path(pq_dir).glob('*.parquet'):
        files.append(str(p))
    dataset = load_dataset("parquet", data_files=files)
    return dataset

def get_default_mapper(col_name):
    return tf.py_function(
        fast_mapping_to_physio_no_weights, 
        inp = [x[col_name]], 
        Tout=(tf.float16, tf.float16) 
    )

def get_tf_dataset(pq_dir,batch_size,feature_col,mapper=None,cache=False):
    """ 

    notes
    ------
    - may want to add in shuffle..?
    - assumes only one column - likely will be more in future?

    """
    dataset = get_parquet_dataset(pq_dir)
    ds = dataset['train'].to_tf_dataset(batch_size=batch_size)
    ds_fast = ds.map(
    lambda x: tf.py_function(
      fast_mapping_to_physio_no_weights, 
      inp = [x[feature_col]], 
      Tout=(tf.float16, tf.float16) 
    ),
    num_parallel_calls = tf.data.AUTOTUNE
    )
    if cache:
        ds_fast = ds_fast.cache()
    ds_fast = ds_fast.prefetch(tf.data.AUTOTUNE)
    return ds_fast

def get_tf_dataset_SampleW(pq_dir,batch_size,feature_col1,feature_col2,mapper=None,cache=False):
    """ 

    notes
    ------
    - may want to add in shuffle..?
    - assumes only one column - likely will be more in future?

    """
    dataset = get_parquet_dataset(pq_dir)
    ds = dataset['train'].to_tf_dataset(batch_size=batch_size)
    ds_fast = ds.map(
    lambda x: tf.py_function(
      fast_mapping_to_physio_insert_weights, 
      inp = [x[feature_col1],x[feature_col2]], 
      Tout=(tf.float16, tf.float16,tf.float32) 
    ),
    num_parallel_calls = tf.data.AUTOTUNE
    )
    if cache:
        ds_fast = ds_fast.cache()
    ds_fast = ds_fast.prefetch(tf.data.AUTOTUNE)
    return ds_fast

def get_tf_dataset_SampleW_preFeats(pq_dir,batch_size,feature_col1,feature_col2,mapper=None,cache=False):
    """ 

    notes
    ------
    - may want to add in shuffle..?
    - assumes only one column - likely will be more in future?

    """
    dataset = get_parquet_dataset(pq_dir)
    ds = dataset['train'].to_tf_dataset(batch_size=batch_size)
    ds_fast = ds.map(
        lambda x: tf.py_function(
            reshape_w_sampleW, 
            inp = [x[feature_col1],x[feature_col2]], 
            Tout=(tf.float32, tf.float32,tf.float32) 
        ),
        num_parallel_calls = tf.data.AUTOTUNE
    )
    if cache:
        ds_fast = ds_fast.cache()
    ds_fast = ds_fast.prefetch(tf.data.AUTOTUNE)
    return ds_fast


















