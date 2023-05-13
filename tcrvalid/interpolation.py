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
import numpy as np
from abc import ABC,abstractmethod

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import seaborn as sns
sns.set(context='talk')
from sklearn.decomposition import PCA
import mlflow
import pathlib
import pandas as pd
from scipy import stats

from .physio_embedding import SeqArrayDictConverter

__all__ = [
    'get_ref_cdr2s',
    'get_ref_cdr2_pairs',
    'CDR2DistanceTracker',
    'CDR3DistanceTracker',
    'FeatureTracker',
    'mean_statistic',
    'mean_startendnorm_statistic',
    'LeastSquaresStatistic',
    'SpearmanStatistic',
    'Interpolator',
    'MetaInterpolation',
    'tcr_pair_gen'
]

def get_ref_cdr2s(trbv_ref_path, mapping=None):
    if mapping is None:
        mapping = SeqArrayDictConverter()
    trbv_ref_df = pd.read_csv(trbv_ref_path,index_col=0)
    trbv_ref_df = trbv_ref_df[trbv_ref_df.cdr2_no_gaps.map(lambda x: len(x)==6 and '*' not in x)].reset_index()
    ref_cdr2s = mapping.seqs_to_array(list(trbv_ref_df.cdr2_no_gaps),maxlen=6)
    ref_cdr2_seqs = list(trbv_ref_df.cdr2_no_gaps)   
    return ref_cdr2s, ref_cdr2_seqs
   
def get_ref_cdr2_pairs(trbv_ref_path,mapping=None):
    ref_cdr2s, ref_cdr2_seqs = get_ref_cdr2s(trbv_ref_path, mapping=mapping)
    ref_cdr2_pairs = [(ref_cdr2_seqs[i], ref_cdr2_seqs[j]) for i in range(len(ref_cdr2_seqs)) for j in range(i+1, len(ref_cdr2_seqs)) ] 
    return ref_cdr2_pairs

class CDR2DistanceTracker:
    """ Track (minimal) distance to possible CDR2s 

    Collect method checks for a TCR representation x, the distance
    (defined by chosen metric) between x's CDR2 region to all possible 
    CDR2 representations (IMGT). It then returns the minimal distance
    to of these.

    parameters
    -----------
    ref_cdr2: array
    array of all possible cdr2s in the chosen mapping context (e.g Meiler)

    cdr2_len: int
    Length of CDR2. Currently should be 6. We only use CDR2s of length
    6, as this is the dominant case.

    metric: callable
    metric function which takes two arrays and returns distances between
    all pairs of features. e.g sklearn pairwise distanece functions.

    """

    def __init__(self,ref_cdr2s,cdr2_len,metric):
        self.ref_cdr2s = ref_cdr2s
        self.cdr2_len = cdr2_len
        self.metric = metric

    def update_state(self,x):
        pass

    def collect(self,x):
        """ Collect the minimal distance to a true CDR2

        parameters:
        -----------
        x: np.array
          TCR representation

        """
        ref_dists = []
        for i in range(len(self.ref_cdr2s)):
            ref_dists.append(np.sum(np.diag(self.metric(x[:self.cdr2_len,:],self.ref_cdr2s[i,:self.cdr2_len,:]))))
        return np.min(ref_dists)
    
    
class CDR3DistanceTracker:
    """ Track the change in the CDR3 representation

    Comparison of the representation is made to a fixed CDR3.
    This can be provided on initialization of the Tracker.
    It can also be updated using update_state(x)

    parameters
    ----------
    ref_seq_rep: np.array
    TCR representation to use as comparitor. Should be entire seq, not just CDR3.

    metric: callable
    metric function which takes two arrays and returns distances between
    all pairs of features. e.g sklearn pairwise distanece functions.

    cdr3_start: int, optional, default=7
    Index in the sequence at which to consider the CDR3 start. We exclusively use
    CDR2 of length 6 in the interpolations. So leave as 7

    cdr3_end: int, optional, default=11
    Where to track changes up to in sequence, inclusive. We use 11, thereby checking
    the distance between TCR represenstion and the reference for the first 5 residues 
    of the CDR3.

    """

    def __init__(self,ref_seq_rep,metric,cdr3_start=7,cdr3_end=11):
        self.ref_seq_rep = ref_seq_rep
        self.cdr3_start = cdr3_start
        self.cdr3_end = cdr3_end
        self.metric = metric

    def update_state(self,x):
        """ Update the reference sequence for comparison """
        self.ref_seq_rep = x[0,:,:]

    def collect(self,x):
        """ Collect distance between TCR representation and reference

        parameters
        ----------
        x: np.array
          TCR representation

        """
        d = np.sum(np.diag(
              self.metric(
                x[self.cdr3_start:(self.cdr3_end+1),:],
                self.ref_seq_rep[self.cdr3_start:(self.cdr3_end+1),:]
              )
            ))
        return d
    
class FeatureTracker:
    """ Track physicochemical features at a specific residue location

    parameters
    -----------
    residue_index: int
    index of residue (in full sequence) to track

    """

    def __init__(self,residue_index):
        self.r = residue_index

    def update_state(self,x):
        pass

    def collect(self,x):
        """ collect the 

        """
        return np.squeeze(x[self.r,:-1])

def mean_statistic(x):
    return np.mean(x)

def mean_startendnorm_statistic(x):
    """ mean of x, normalized to mean of start and end values

    For trajectory, this is helpful as it prevents penalizing based
    on overall reconstruction accuracy, which is measured seperately.

    """
    return np.mean(x/(0.5*(x[0]+x[-1])))

class LeastSquaresStatistic:
    """ Statistic applied across tracked values on interpolation

    Callabel statistic that can be applied to tracked feature.
    It uses the value of the feature across the interpolation
    and known start and end points to measure the least Squares
    Error against expect linear line.

    internally the internal_values variable holds a tuple
    (running,values,features,lse)
    - running: LSE for each feature, for each point on interpolation
    - values: features along trajectory, noamlized to expected variation
    - features: The features at each point on trajectory
    - lse: LSE at each point along interpolation, mean'ed over features.

    parameters
    -----------
    feats1: np.array
    Feature array at begeinning of interpolation

    feats2: np.array
    Feature array at end of interpolation

    """

    def __init__(self,feats1,feats2):
        """ remove the gap feature beforehand """
        self.feats1 = feats1
        self.feats2 = feats2
        self.internal_values = None

    def __call__(self,x):
        """ Get the LSE for tracked values

        parameters
        ------------
        x: list 
          list of features over the interpolation. Entries can 
          be np.arrays.

        returns
        -------
        LSE: value
          The mean least square error on feature over interpolation
          comparing to expected linear slope.


        """
        x_stack = np.vstack(x) # N,featssize
        print(x_stack.shape)
        f1 = np.tile(self.feats1,(len(x), 1))
        f2 = np.tile(self.feats2,(len(x), 1))
        print(f2.shape)
        values = (x_stack - f1)/(f2-f1)
        running = np.zeros_like(values)
        for n in range(len(x)):
            fraction = 1.0 - n/(len(x)-1)
            running[n,:] = (values[n,:] - (1.0-fraction) )**2
        lse = np.mean(running,axis=0) #value per feature
        self.internal_values = (running,values,x_stack,lse)
        return np.mean(lse)
    
class SpearmanStatistic:
    """ Statistic applied across tracked values on interpolation

    Callable statistic that can be applied to tracked feature.
    It uses the value of the feature across the interpolation
    and known start and end points to measure the Spearman
    correlation.

    internally the internal_values variable holds a tuple
    (corr,values,features)
    - corr: correlation for each feature
    - values: features along trajectory, noamlized to expected variation
    - features: The features at each point on trajectory

    parameters
    -----------
    feats1: np.array
    Feature array at begeinning of interpolation

    feats2: np.array
    Feature array at end of interpolation

    """

    def __init__(self,feats1,feats2):
        """ remove the gap feature beforehand """
        self.feats1 = feats1
        self.feats2 = feats2
        self.internal_values = None

    def __call__(self,x):
        """ Get the Spearman Correlation for tracked values

        parameters
        ------------
        x: list 
          list of features over the interpolation. Entries can 
          be np.arrays.

        returns
        -------
        CORR: value
          The mean Spearman correlation on normalized feature
          over interpolation.
        """
        x_stack = np.vstack(x) # N,featssize
        print(x_stack.shape)
        f1 = np.tile(self.feats1,(len(x), 1))
        f2 = np.tile(self.feats2,(len(x), 1))
        print(f2.shape)
        values = (x_stack - f1)/(f2-f1)
        corr = np.zeros((x_stack.shape[1]))
        for a in range(x_stack.shape[1]):
            corr[a] = stats.spearmanr(np.arange(len(x_stack))/(len(x_stack)-1),values[:,a]).correlation #value per feature
        self.internal_values = (corr,values,x_stack)
        return np.mean(corr)
    
    
def convert_probs_max(probs):
    b=np.zeros_like(probs[0,:,:])
    b[np.arange(len(probs[0,:,:])), probs[0,:,:].argmax(1)] = 1
    return b

class Interpolator():
    """ Interpolation in latent space between two TCRs

    Project two TCRs into latent space, and walk along the line that connects them in
    the latent space. At n_steps points along the line, generate a TCR.

    For each TCR along this Interpolation, values can be tracked which quanitfy the
    trajectory in terms of e.g. smoothness.

    parameters
    -----------

    mapping: callable
    mapping which maps residues to features. This should be the same
    used to train the (V)AE


    Example
    -------
    >>>tcr1 = 'YSYEKL-ASSRAGNTEAF'
    >>>tcr2 = 'YYREEE-ASSRAGNTEAF'
    >>>interp = Interpolator(mapping)
    >>>interp.add_tracker(cdr2_d_tracker,mean_statistic,'CDR2')
    >>>interp.add_tracker(cdr3_d_tracker,mean_statistic,'CDR3')
    >>>interp.add_tracker(cdr3_d_tracker,mean_startendnorm_statistic,'CDR3_norm')
    >>>interp(
      tcr1,
      tcr2,
      enc,
      dec,
      10,
    )
    """

    def __init__(self,mapping):
        self.mapping=mapping
        self.trackers = dict()
        self.statistics = dict()
        self.tracked_values = dict()
        self.tracked_stats = dict()
        self.outs = []
        self.zs = []

    def add_tracker(self,tracker,statistic,name):
        """ Attach a tracker to the interpolation

        parameters
        ----------

        tracker: Tracker object
          Object with a 'collect()' method, which uses a TCR
          representation and returns a value/array quantifying
          some aspect of theat representation.

        statistic: callable
          A callable statistic which collates all values collected
          by the tracker over the interpolation into a single value.

        name: str
          A name to give the final statistic for the interpolation


        """
        self.trackers[name] = tracker
        self.statistics[name] = statistic

    def _reset_internal(self):
        self.tracked_values = dict()
        self.tracked_stats = dict()
        self.outs = []
        self.zs = []
        for k in self.trackers.keys():
            self.tracked_values[k] = []
            self.tracked_stats[k] = []

    def plot_trajectory(self,max_residue=False):
        """ Plot weblogo images of TCRs along the trajectory

        parameters
        ----------

        max_residue: bool, optional, default=False
          If True logo will use the most probable residue in each position
          from the generated TCR representation. If False (default) will plot 
          the logo according the probabilities of each residue given the TCR
          representation.

        """
        if len(self.outs)==0:
            print(
                'warning: no trajectory outputs:'+\
                'check keep_outputs was true when interpolation was called'
            )
            return None

        f,axes = plt.subplots(len(self.outs),1,figsize=(5,20))
        for j in range(len(self.outs)):
            probs = self.mapping.array_to_probs(self.outs[j],inv_manhattan) 
            if max_residue:
                probs = convert_probs_max(probs)
            else:
                probs = probs[0,:,:]
            logo_from_probs(
                probs,
                self.mapping,
                color_scheme='chemistry',
                units='probability',
                ax=axes[j]
            )
        return f

    def __call__(self,tcr1,tcr2,enc,dec,n_steps,keep_outputs=True):
        """ Apply the interpolation and gather statistics

        Project TCR1 and 2 into latent space, walk along linear line between them
        in latent space. At n_steps points (end point inclusive), generate TCRs using 
        the model decoder. For each TCR representation along the trajectory collect the 
        all values related to each of the attached Trackers. At the end of the
        interpolation gather statistic over the entire trajectory for each tracked 
        property using its assocaiated statistic.


        parameters
        -----------

        tcr1: str
          tcr input sequence at start of interpolation

        tcr2: str
          tcr input sequence at end of interpolation

        enc: mlflow model
          Encoder model

        dec: mlflow model
          Decoder model

        n_steps: int
          Number of steps to use on interpolation. Inclusive of end points.

        keep_outputs: bool, optional, default=True
          If True, keep the representations along the trajectory. 
          If False, do not. This does not impact Tracked statistics.
          Required if wish to subsequently plot the trajectory logos.

        returns
        ---------
        tracked_stats: dict
          A dictionary of tracked statistics. keys are names provided
          with each tracker addded to the Interpolator. Values are the 
          stastic of each tracked set of values. 

        """
        # reset state of trackers
        self._reset_internal()

        # begin interpolation
        tcr1_rep = self.mapping.seqs_to_array([tcr1],maxlen=28)
        tcr2_rep = self.mapping.seqs_to_array([tcr2],maxlen=28)
        z1,_,_ = enc.predict(tcr1_rep)
        z2,_,_ = enc.predict(tcr2_rep)

        # some trackers require a baseline representation for
        # comparison
        for k,v in self.trackers.items():
            v.update_state(tcr1_rep)

        for n in range(0,n_steps):
            fraction = 1.0 - n/(n_steps-1)
            #print((n/(n_steps-1)), (1.0 - n/(n_steps-1)) )
            z = fraction*z1 + (1.0 - fraction)*z2
            x_out = dec.predict(z)

            for k,v in self.trackers.items():
                self.tracked_values[k].append(v.collect(x_out[0,:,:]))

            if keep_outputs:
                self.outs.append(x_out)
                self.zs.append(z)

        for k,v in self.statistics.items():
            self.tracked_stats[k] = v(np.array(self.tracked_values[k]))

        return self.tracked_stats
    
    
class MetaInterpolation:
    """ Interpolate between many sets of TCRs and track statistics

    parameters
    -----------

    mapping: callable
    A mapping from residues to features

    tracker_tuples: list, optional, default=None
    List of tuples of (tracker, statistic, name). See 
    Interpolator.add_tracker() for details.

    """

    def __init__(self, mapping, tracker_tuples=None):
        #     self.tcr_pairs = tcr_pairs
        #     self.enc = enc
        #     self.dec = dec
        #     self.n_steps = n_steps
        self.tracker_tuples = tracker_tuples
        self.interpolator= None
        #     self.results = []
        # cretae interpolator and add trackers to it
        self._setup_interpolator(mapping)


    def _setup_interpolator(self,mapping):
        self.interpolator = Interpolator(mapping)
        for tracker,stat,name in self.tracker_tuples:
            self.interpolator.add_tracker(tracker,stat,name)
        return None

    def __call__(self,tcr_pairs,enc,dec,n_steps):
        """ do interpolation over many pairs of TCRS

        parameters
        ------------

        tcr_pairs: Iterable (list, generator, etc)
          iterable of tuples (tcr1, tcr2), wher each tcr
          is a string.

        enc: mlflow model
          Encoder model

        dec: mlflow model
          Decoder model

        n_steps: int
          Number of steps to use on interpolation. Inclusive of end points.


        returns:
        --------

        df: pd.DataFrame
          Dataframe of the results of each of the tracked statistics (columns)
          over the interpolation (rows)

        """
        self.results = []
        for tcr1,tcr2 in tcr_pairs:
            stats = self.interpolator(tcr1,tcr2,enc,dec,n_steps,keep_outputs=False)
            self.results.append(stats)
        return pd.DataFrame(self.results)
    

def tcr_pair_gen(seq_df,ref_cdr2_pairs,n_cdr3,n_cdr2_pair):
    """ Generator of pairs of TCRs for interpolations

    For each CDR3, make pairs of tcrs with the same CDR3 and 
    multiple sets of possible CDR2s.

    Total number of TCR pairs will be:
    n_cdr3 * n_cdr2_pair

    parameters
    ----------

    seq_df: pd.DataFrame
    Dataframe of TCRs to grab CDR3s from

    ref_cdr2_pairs: list
    List of all pairs (tuples) of CDR2s of length 6

    n_cdr3: int
    Number of CDR3s to use

    n_cdr2_pair: int
    Number of CDR2 pairs to use per CDR3


    """
    for i,s in enumerate(seq_df['cdr3']):
        if i>n_cdr3:
            break
        for j in range(n_cdr2_pair):
            r = np.random.randint(0,len(ref_cdr2_pairs))
            cdr2_pair = ref_cdr2_pairs[r]
            yield (cdr2_pair[0]+'-'+s, cdr2_pair[1]+'-'+s)
    

    
    
    
    
    
    