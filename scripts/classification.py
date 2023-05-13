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
""" script showing example classification with uncertainty estimation """
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import sklearn
import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
import sklearn
import os
import sys

from tcrvalid.load_models import *
from tcrvalid.plot_utils import set_simple_rc_params
from tcrvalid.physio_embedding import SeqArrayDictConverter
from tcrvalid.data_subsetting import *
from tcrvalid.defaults import *

# ----------------- set up chosen models and data to use --------------------------------
from_keras = True #expect True in github code
load_mini_trb = True #expect True in github code, for paper results: False (very similar)
pull_using_spark = False #expect False in github. if true, use spark to read in data, easier to use pandas for small dataset

model_names = {
    'TRA': ['1_2'],
    'TRB': ['1_2_full_40']
}

if pull_using_spark:
    if not load_mini_trb:
        trb_test_pq = data_path_full_trb_spark['te']
    else:
        trb_test_pq = data_path_small_trb_spark['te']
    tra_test_pq = data_path_full_tra_spark['te']
else:
    if not load_mini_trb:
        trb_test_pq = data_path_full_trb['te']
    else:
        trb_test_pq = data_path_small_trb['te']    
    tra_test_pq = data_path_full_tra['te']

print(tra_test_pq, trb_test_pq)    
    
# -----------------------------------------------------------------------------------------
# below are several helper functions used for this task
# -----------------------------------------------------------------------------------------

def get_labelled_data(df_labelled,trb_model=None,tra_model=None):
    """ convert raw labelled TCRs to TCRVALID representations
    
    parameters
    ----------
    
    df_labeled: pd.DataFrame
        raw labeled TCR data to use
        
    trb_model: tf model, optional, default=None
        model for collecting representations for TRB chain. If None will
        use only the TRA model representations. If TRB and TRA models provided
        will concatenate features.
        
    tra_model: tf model, optional, default=None
        model for collecting representations for TRA chain. If None will
        use only the TRB model representations. If TRB and TRA models provided
        will concatenate features.
    
    returns
    -------
    (features, labels): tuple
    
    """
    if trb_model is not None:
        f_l_trb = mapping.seqs_to_array(df_labelled.pre_feature_TRB.values,maxlen=28)
        x_l_trb,_,_ = trb_model.predict(f_l_trb)
    if tra_model is not None:
        f_l_tra = mapping.seqs_to_array(df_labelled.pre_feature_TRA.values,maxlen=28)
        x_l_tra,_,_ = tra_model.predict(f_l_tra)

    y_l = df_labelled.label.values

    if tra_model is not None and trb_model is not None:
        print(x_l_trb.shape)
        print(x_l_tra.shape)
        x_l = np.concatenate([x_l_trb, x_l_tra],axis=1)
    elif tra_model is None:
        x_l = x_l_trb
    elif trb_model is None:
        x_l = x_l_tra
    else:
        raise ValueError()

    return x_l, y_l

def get_unlabelled_data(te_seq_df_trb, te_seq_df_tra, trb_model=None,tra_model=None):
    """ Collect unlabeled raw data as TCRVALID features
    
    parameters
    ----------
    
    te_seq_df_trb: pd.DataFrame
        raw unlabeled TRB TCR data
        
    te_seq_df_tra: pd.DataFrame
        raw unlabeled TRA TCR data
        
    trb_model: tf model, optional, default=None
        model for collecting representations for TRB chain. If None will
        use only the TRA model representations. If TRB and TRA models provided
        will concatenate features.
        
    tra_model: tf model, optional, default=None
        model for collecting representations for TRA chain. If None will
        use only the TRB model representations. If TRB and TRA models provided
        will concatenate features.
    
    returns
    -------
    features: np.array
    
    """
    if trb_model is not None:
        f_u_trb = mapping.seqs_to_array(te_seq_df_trb.cdr2_cdr3, maxlen=28)
        x_u_trb,_,_ = trb_model.predict(f_u_trb)
    if tra_model is not None:
        f_u_tra = mapping.seqs_to_array(te_seq_df_tra.cdr2_cdr3,maxlen=28)
        x_u_tra,_,_ = tra_model.predict(f_u_tra)

    if tra_model is not None and trb_model is not None:
        x_u = np.concatenate([x_u_trb, x_u_tra],axis=1)
    elif tra_model is None:
        x_u = x_u_trb
    elif trb_model is None:
        x_u = x_u_tra
    else:
        raise ValueError()

    return x_u

def tts(x,y=None):
    """ train test split """
    if y is not None:
        x_tr, x_test, y_tr, y_test = train_test_split(
          x, 
          y, 
          stratify=y,
          test_size=0.1
        )
        x_train, x_vali, y_train, y_vali = train_test_split(
          x_tr, 
          y_tr, 
          stratify=y_tr,
          test_size=0.15
        )
        d_out = {
          'train': (x_train, y_train),
          'vali': (x_vali, y_vali),
          'test': (x_test, y_test)
        }
    else:
        x_tr, x_test = train_test_split(
          x, 
          test_size=0.1
        )
        x_train, x_vali = train_test_split(
          x_tr, 
          test_size=0.15
        )
        d_out = {
          'train': x_train, 
          'vali': x_vali, 
          'test': x_test,
        }
    return d_out

def ds_generator(xl,yl,xu,lsize,batch_size):
    """ Generator of labeled, unlabeled data
    
    parameters
    -----------
    xl: np.array
        labeled data features
    
    yl: np.array
        labels for labeled data
        
    xu: np.array
        unlabeled data features
        
    lsize: int
        number to use from labeled data
        
    batch_size: int
        How many per batch in total
    
    
    """
    i=0
    while i<len(xl):
        tmp_xl = xl[i:i+lsize]
        tmp_yl = yl[i:i+lsize]
        tmp_xu = xu[np.random.choice(len(xu), size=batch_size-len(tmp_xl))]
        tmp_x  = np.vstack( (tmp_xl, tmp_xu ))
        tmp_sw = np.squeeze(np.vstack( 
          (
            np.ones((tmp_xl.shape[0],1)), 
            np.zeros((tmp_xu.shape[0],1)), 
          )
        ))
        i+=lsize
        yield tmp_x, tmp_yl, tmp_sw
        
def weighted_cross_entropy(y_true,y_pred,weights):
    """

    Note, does not mean over batch - gives one value per sample
    """
    y_true = tf.cast(tf.convert_to_tensor(y_true),tf.float32)
    y_pred = tf.cast(tf.convert_to_tensor(y_pred),tf.float32)
    weights = tf.cast(tf.convert_to_tensor(weights),tf.float32)
    y_true.shape.assert_is_compatible_with(y_pred.shape)

    epsilon_ = tf.constant(
    tf.keras.backend.epsilon(), 
    y_pred.dtype.base_dtype
    )
    y_pred = tf.clip_by_value(y_pred, epsilon_, 1.0 - epsilon_)
    return -tf.reduce_sum(y_true * tf.math.log(y_pred) * weights, axis=1)


class ConfidenceClassifier(keras.Model):
    """ tf.keras model to calssify labeled data while weighting uniformity of predictions on unlabeled data
    
    parameters
    ------------
    
    dim_in: int
        dimensions of input data
        
    dim_out: int
        dimensions of the output data
        
    cw: np.array
        class weights for classification
    
    """
  
    def __init__(self,dim_in,dim_out,cw,balance=1.0):
        super(ConfidenceClassifier, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.cw = cw
        self.balance = balance

        self.network = keras.Sequential()
        self.network.add(tf.keras.Input(shape=(self.dim_in,)))
        self.network.add( keras.layers.Dropout(0.25))
        self.network.add( keras.layers.Dense(128, activation='relu') )
        self.network.add( keras.layers.Dropout(0.4))
        self.network.add( keras.layers.Dense(128, activation='relu') )
        self.network.add( keras.layers.Dropout(0.4) )
        self.network.add( keras.layers.Dense(self.dim_out, activation='softmax') )

        self.u = tf.constant([1./self.dim_out]*self.dim_out)

        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.classification_loss_tracker = keras.metrics.Mean(
            name="classification_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
          self.total_loss_tracker,
          self.classification_loss_tracker,
          self.kl_loss_tracker,
        ]
    
    def __call__(self,x,training=False):
        return self.network(x,training=False)
    
    def train_step(self,data):
        if len(data) == 3:
            x, y, sw = data
            sw = tf.cast(sw,tf.bool)
        else:
            sw = None
            x, y = data

        with tf.GradientTape() as tape:
            preds = self.network(x,training=True)
            preds_in = tf.boolean_mask(preds, sw)
            preds_out = tf.boolean_mask(preds, tf.math.logical_not(sw))
            # for each of say 32 samples, then mean over that minibatch
            oh_y = tf.one_hot(tf.cast(y,tf.int64), depth=self.dim_out)
            wce = tf.reduce_mean(weighted_cross_entropy(oh_y,preds_in,self.cw))
            # for each of say 512-32 samples, then mean over that mini-batch
            kl_loss = tf.reduce_mean(keras.metrics.kl_divergence( self.u , preds_out))
            total_loss = wce + self.balance*kl_loss

    
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.classification_loss_tracker.update_state(wce)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "classification_loss": self.classification_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
  
    def test_step(self,data):
        if len(data) == 3:
            x, y, sw = data
            sw = tf.cast(sw,tf.bool)
        else:
            sw = None
            x, y = data
            
        preds = self.network(x,training=False)
        preds_in = tf.boolean_mask(preds, sw)
        preds_out = tf.boolean_mask(preds, tf.math.logical_not(sw))

        # for each of say 32 samples, then mean over that minibatch
        oh_y = tf.one_hot(tf.cast(y,tf.int64), depth=self.dim_out)
        wce = tf.reduce_mean(weighted_cross_entropy(oh_y,preds_in,self.cw))
        # for each of say 512-32 samples, then mean over that mini-batch
        kl_loss = tf.reduce_mean(keras.metrics.kl_divergence( self.u , preds_out))

        total_loss = wce + self.balance*kl_loss

        return {
          "loss": total_loss,
          "classification_loss": wce,
          "kl_loss": kl_loss
        }

def run_case(balance = 1.0, verbose=0, dim_in=32):
    """ Train and evaluate uncertainty-aware classification model 
    
    Train and evaluate model or a given 'balance' (alpha) between classification task 
    and task to make unlabeled data have uniform predictions over classes.
    
    parameters
    ------------
    
    balance: float
        "alpha" weighting of uniformity task in loss
        
    verbose: int
        passed to keras model fit to control text output
        
    dim_in: int
        Size of the input data to the classification model.
    
    """
    
    # train test split
    data_l = tts(x_l,y=y_l)
    data_u = tts(x_u)

    # construct input generator
    # batch of 512, with 32 of those from the labeled data in each batch
    datasets = dict()
    for k in data_l.keys():
        args = data_l[k][0],data_l[k][1],data_u[k],32,512
        datasets[k] = tf.data.Dataset.from_generator(
          ds_generator,
          args=args,
          output_signature=(
            tf.TensorSpec(shape=[None,dim_in],dtype=tf.float32),
            tf.TensorSpec(shape=[None,]),
            tf.TensorSpec(shape=[None,])
          )
        )

    # calculate the class weights for classification task
    labels = data_l['train'][1]
    cw = sklearn.utils.class_weight.compute_class_weight('balanced', np.unique(labels), labels)
    cw = cw.reshape((1,len(np.unique(labels))))

    # define the model - compile and fit
    model = ConfidenceClassifier(dim_in,len(peptides),cw, balance=balance)

    model.compile(
        optimizer=keras.optimizers.Adam()
    )

    callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5
    )

    history = model.fit(
        datasets['train'],
        validation_data=datasets['vali'],
        epochs=200,
        callbacks=[callback],
        verbose=verbose
    )

    # outputs on the left out test set
    outs_l = model( data_l['test'][0] )
    outs_u = model( data_u['test'] )
    
    # collect performance stastics 

    roc_test = sklearn.metrics.roc_auc_score(
        tf.one_hot(data_l['test'][1],depth=8).numpy(),
        outs_l,
        average=None,
        multi_class='ovr'
    )

    ap_test = sklearn.metrics.average_precision_score(
        tf.one_hot(data_l['test'][1],depth=8).numpy(),
        outs_l,
        average=None
    )

    ood_labels = np.hstack([
        np.ones((len(outs_l),)),
        np.zeros((len(outs_u),))
    ])

    kl_l = keras.metrics.kl_divergence( np.array([1/8]*8) , outs_l)
    kl_u = keras.metrics.kl_divergence( np.array([1/8]*8) , outs_u)

    ood_probs = np.hstack([
        kl_l,
        kl_u
    ])
    ood_probs = ood_probs/np.max(ood_probs)

    roc_ood = sklearn.metrics.roc_auc_score(
        ood_labels,
        ood_probs,
    )

    ap_ood = sklearn.metrics.average_precision_score(
        ood_labels,
        ood_probs,
    )

    performance = {
        'AUROC_test': roc_test,
        'AUROC_ood': roc_ood,
        'AP_test': ap_test,
        'AP_ood': ap_ood
    }
    return performance

def mean_over_dicts(dicts):
    outdict = dict()
    for k in dicts[0].keys():
        if dicts[0][k] is not None:
            if isinstance(dicts[0][k],(float,int)):
                outdict[k] = 0.0
            else:
                outdict[k] = np.zeros_like(dicts[0][k])
            for d in dicts:
                outdict[k] += d[k]/len(dicts)
    return outdict

def std_over_dicts(dicts):
    outdict = dict()
  
    val_keys = []
    np_keys = []
    for k in dicts[0].keys():
        if dicts[0][k] is not None:
            if isinstance(dicts[0][k],(float,int)):
                val_keys.append(k)
            else:
                np_keys.append(k)
  
    for k in val_keys:      
        outdict[k] = np.std([d[k] for d in dicts])
    for k in np_keys:
        #print(k, dicts[0][k].shape)
        tmp = np.array([d[k].reshape((len(d[k]))) for d in dicts])
        #print(tmp.shape)
        outdict[k] = np.std(tmp,axis=0)
    return outdict


# -----------------------------------------------------------------------------------------
# end of helper functions used for this task
# begin 'main'
# -----------------------------------------------------------------------------------------

# get models and data

mapping = SeqArrayDictConverter()

loaded_trb_models = load_named_models(model_names['TRB'],chain='TRB',as_keras=from_keras)
loaded_tra_models = load_named_models(model_names['TRA'],chain='TRA',as_keras=from_keras)

if pull_using_spark:
    sdf_trb=spark.read.format("parquet").load(trb_test_pq)
    te_seq_trb_df = sdf_trb.select('cdr2_no_gaps','cdr2_cdr3').limit(100000).toPandas()

    sdf_tra=spark.read.format("parquet").load(tra_test_pq)
    te_seq_tra_df = sdf_tra.select('cdr2_no_gaps','cdr2_cdr3').limit(100000).toPandas()
else:
    te_seq_trb_df = pd.read_parquet(trb_test_pq, columns=['cdr2_no_gaps','cdr2_cdr3']).head(100000)
    te_seq_tra_df = pd.read_parquet(tra_test_pq, columns=['cdr2_no_gaps','cdr2_cdr3']).head(100000)

df = pd.read_csv(labelled_data_path)
sources = sources_['both']
chains = chains_['TR']
feature = features_['CDR23']
subset_df = make_subset(df,sources=sources,chains=chains,feature=feature,min_size=3, max_len=28)

# use only antigens with at least 100 unique TCRs
vc = subset_df.value_counts('peptide')
peptides = vc[vc>100].index
df_labelled = subset_df[subset_df['peptide'].isin(peptides)]

pep_dict = {p:i for i,p in enumerate(peptides)}
df_labelled['label'] = df_labelled['peptide'].map(pep_dict)

tra_model = loaded_tra_models['1_2']
trb_model = loaded_trb_models['1_2_full_40']

# convert raw data to TCRVALID features

x_l, y_l = get_labelled_data(
  df_labelled,
  trb_model=trb_model,
  tra_model=tra_model
)

x_u = get_unlabelled_data(
  te_seq_trb_df, 
  te_seq_tra_df,
  trb_model=trb_model,
  tra_model=tra_model
)

# ----------for each balance in 'balances' train and evaluate model ----------

all_perfs = dict()

# balances =  [0.0,0.1,1.0,10.0,30.0,100.0]
balances =  [0.0,1.0,30.0]
for b in balances:
    perfs = []
    for i in range(5):
        perfs.append(run_case(balance=b,dim_in=32))
    all_perfs[b] = perfs
    
i_pep_dict = {i:p for p,i in pep_dict.items()}
    
# ---------- collect results ----------

roc_means_dict = dict()
for k,perfs in all_perfs.items():
    roc_means_dict[k] = dict()
    for kk,v in mean_over_dicts(perfs).items():
        if kk.startswith('AUROC'):
            if kk!='AUROC_ood':
                for i,val in enumerate(v):
                    roc_means_dict[k][i_pep_dict[i]] = val
            else:
                roc_means_dict[k]['OOD'] = v

roc_stds_dict = dict()
for k,perfs in all_perfs.items():
    roc_stds_dict[k] = dict()
    for kk,v in std_over_dicts(perfs).items():
        if kk.startswith('AUROC'):
            if kk!='AUROC_ood':
                for i,val in enumerate(v):
                    roc_stds_dict[k][i_pep_dict[i]] = val
            else:
                roc_stds_dict[k]['OOD'] = v
    
    
ap_means_dict = dict()
for k,perfs in all_perfs.items():
    ap_means_dict[k] = dict()
    for kk,v in mean_over_dicts(perfs).items():
        if kk.startswith('AP'):
            if kk!='AP_ood':
                for i,val in enumerate(v):
                    ap_means_dict[k][i_pep_dict[i]] = val
            else:
                ap_means_dict[k]['OOD'] = v    
    
ap_stds_dict = dict()
for k,perfs in all_perfs.items():
    ap_stds_dict[k] = dict()
    for kk,v in std_over_dicts(perfs).items():
        if kk.startswith('AP'):
            if kk!='AP_ood':
                for i,val in enumerate(v):
                    ap_stds_dict[k][i_pep_dict[i]] = val
            else:
                ap_stds_dict[k]['OOD'] = v  
    
print(pd.DataFrame(roc_means_dict).round(decimals=3).to_latex())  
print(pd.DataFrame(ap_means_dict).round(decimals=3).to_latex())  
# finally save the outputs as csv and latex text strings
