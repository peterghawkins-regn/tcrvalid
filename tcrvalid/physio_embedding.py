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
from sklearn.preprocessing import StandardScaler
import sklearn.metrics.pairwise as pairwise
from scipy import special
import pandas as pd
# from Bio.Alphabet import Reduced

import matplotlib.pyplot as plt

from PIL import Image
import io

import weblogo

from numba import jit
from numba.typed import Dict

MEILER_PROPERTY_NAMES = [
  'steric',
  'polarizability',
  'volume',
  'hydrophobicity',
  'isoelectric',
  'helix_prob',
  'sheet_prob'
]

MEILER_FEATURES = {
    'C':np.array([1.77, 0.13, 2.43,  1.54,  6.35, .17, 0.41]),
    'S':np.array([1.31, 0.06, 1.60, -0.04,  5.70, .20, 0.28]),
    'T':np.array([3.03, 0.11, 2.60,  0.26,  5.60, .21, 0.36]),
    'P':np.array([2.67, 0.00, 2.72,  0.72,  6.80, .13, 0.34]),
    'A':np.array([1.28, 0.05, 1.00,  0.31,  6.11, .42, 0.23]),
    'G':np.array([0.00, 0.00, 0.00,  0.00,  6.07, .13, 0.15]),
    'N':np.array([1.60, 0.13, 2.95, -0.60,  6.52, .21, 0.22]),
    'D':np.array([1.60, 0.11, 2.78, -0.77,  2.95, .25, 0.20]),
    'E':np.array([1.56, 0.15, 3.78, -0.64,  3.09, .42, 0.21]),
    'Q':np.array([1.56, 0.18, 3.95, -0.22,  5.65, .36, 0.25]),
    'H':np.array([2.99, 0.23, 4.66,  0.13,  7.69, .27, 0.30]),
    'R':np.array([2.34, 0.29, 6.13, -1.01, 10.74, .36, 0.25]),
    'K':np.array([1.89, 0.22, 4.77, -0.99,  9.99, .32, 0.27]),
    'M':np.array([2.35, 0.22, 4.43,  1.23,  5.71, .38, 0.32]),
    'I':np.array([4.19, 0.19, 4.00,  1.80,  6.04, .10, 0.45]),
    'L':np.array([2.59, 0.19, 4.00,  1.70,  6.04, .39, 0.31]),
    'V':np.array([3.67, 0.14, 3.00,  1.22,  6.02, .27, 0.49]),
    'F':np.array([2.94, 0.29, 5.89,  1.79,  5.67, .30, 0.38]),
    'Y':np.array([2.94, 0.30, 6.47,  0.96,  5.66, .25, 0.41]),
    'W':np.array([3.21, 0.41, 8.08,  2.25,  5.94, .32, 0.42]),
    'U':np.array([0.,0.,0.,0.,0.,0.,0.])
}

ATCHLEY_PROPERTY_NAMES = [
  'FACTOR.I (exposure)',
  'FACTOR.II (2ndry struc)',
  'FACTOR.III (size)',
  'FACTOR.IV (aa composition)',
  'FACTOR.V (charge)',
]

ATCHLEY_FEATURES = {
  'A': np.array([-0.591, -1.302, -0.733, 1.570,	-0.146]),
  'C': np.array([-1.343, 0.465,	-0.862,	-1.020,	-0.255]),
  'D': np.array([ 1.050, 0.302,	-3.656, -0.259,	-3.242]),
  'E': np.array([1.357, -1.453,	1.477, 0.113, -0.837]),
  'F': np.array([-1.006, -0.590, 1.891,	-0.397,	0.412]),
  'G': np.array([-0.384, 1.652,	1.330, 1.045, 2.064]),
  'H': np.array([0.336,	-0.417,	-1.673, -1.474, -0.078]),
  'I': np.array([-1.239, -0.547, 2.131, 0.393, 0.816]),
  'K': np.array([1.831,	-0.561,	0.533, -0.277,	1.648]),
  'L': np.array([-1.019, -0.987, -1.505, 1.266, -0.912]),
  'M': np.array([-0.663, -1.524, 2.219, -1.005,	1.212]),
  'N': np.array([0.945,	0.828, 1.299, -0.169,	0.933]),
  'P': np.array([0.189,	2.081, -1.628, 0.421, -1.392]),
  'Q': np.array([0.931,	-0.179,	-3.005, -0.503, -1.853]),
  'R': np.array([1.538,	-0.055,	1.502, 0.440, 2.897]),
  'S': np.array([-0.228, 1.399,	-4.760, 0.670, -2.647]),
  'T': np.array([-0.032, 0.326,	2.213, 0.908, 1.313]),
  'V': np.array([-1.337, -0.279, -0.544, 1.242, -1.262]),
  'W': np.array([-0.595, 0.009,	0.672, -2.128, -0.184]),
  'Y': np.array([0.260,	0.830, 3.097, -0.838, 1.512])
}

color_schemes = {
  'chemistry' : [
      weblogo.colorscheme.SymbolColor("GSTYC", "green", "polar"),
      weblogo.colorscheme.SymbolColor("NQ", "purple", "neutral"),
      weblogo.colorscheme.SymbolColor("KRH", "blue", "basic"),
      weblogo.colorscheme.SymbolColor("DE", "red", "acidic"),
      weblogo.colorscheme.SymbolColor("PAWFLIMV", "black", "hydrophobic"),
      weblogo.colorscheme.SymbolColor("-", "#FF00FF", "gap"),
  ],
  'hydrophobicity' : [
      weblogo.colorscheme.SymbolColor("RKDENQ", "blue", "hydrophilic"),
      weblogo.colorscheme.SymbolColor("SGHTAP", "green", "neutral"),
      weblogo.colorscheme.SymbolColor("YVMCLFIW", "black", "hydrophobic"),
      weblogo.colorscheme.SymbolColor("-", "#FF00FF", "gap"),
  ],
  'charge' : [
    weblogo.colorscheme.SymbolColor("KRH", "blue", "positive"),
    weblogo.colorscheme.SymbolColor("DE", "red", "negative"),
    weblogo.colorscheme.SymbolColor("-", "#FF00FF", "gap"),
  ]
}

class AminoFeatures:
    def __init__(self,feature_dict=MEILER_FEATURES, property_names=MEILER_PROPERTY_NAMES, zscale=True,add_gap=True):
        self.feature_dict = feature_dict.copy() #MEILER_FEATURES.copy()
        self.names =property_names #MEILER_PROPERTY_NAMES
        self.chars = sorted([k for k in self.feature_dict.keys()])
        self.add_gap = add_gap

        # might not want to add gaps until normalized
        # but also need feature array in place before scaling...
        self.feat_array = self._construct_feature_array()
        if zscale:
            self.scale_features()

        if add_gap:
            for ch in self.chars:
                self.feature_dict[ch] = np.concatenate((self.feature_dict[ch], [0.0]))
            self.chars = self.chars+['-']
            self.feature_dict['-'] = np.concatenate( ( np.zeros(len(self.feature_dict[self.chars[0]])-1), [1.0]) )
            self.names += ['gap']
            #print(self.feature_dict)
            # must rebuild feature_array
            self.feat_array = self._construct_feature_array()

        self.fast_feature_dict = Dict()
        for k, v in self.feature_dict.items():
            self.fast_feature_dict[k] = v

    def _construct_feature_array(self):
        feats = np.zeros(
          (
            len(self.feature_dict),
            len(self.feature_dict[self.chars[0]])
          )
        )
        print(feats.shape)
        for i,ch in enumerate(self.chars):
            feats[i,:] = self.feature_dict[ch]
        return feats

    def scale_features(self):
        scaler = StandardScaler()
        self.feat_array = scaler.fit_transform(self.feat_array)
        for i,ch in enumerate(self.chars):
            self.feature_dict[ch] = self.feat_array[i]
        return None

    def __getitem__(self,key):
        return self.feature_dict[key]

    @property
    def feature_size(self):
        return len(self.feature_dict[self.chars[0]])

    @property
    def is_gapped(self):
        return self.add_gap
    
    
def seq_to_array(seq,dictionary,maxlen=None):
    if maxlen is None:
        maxlen=len(seq)
    width = len(dictionary[list(dictionary.keys())[0]])
    x=np.zeros((maxlen,width))
    for j,amino in enumerate(seq):
        x[j,:] = dictionary[amino]
    return x

def feat_arr_similarity(metric,x,feats):
    return metric(x,feats)

def array_to_seq_quick(x,metric,feats,chars):
    sims = feat_arr_similarity(metric,x,feats)
    outseq = np.array(chars)[np.argmax(sims,axis=1)]
    return ''.join(outseq)

def array_to_seq(x,metric,dictionary=None,feats=None,chars=None,maxlen=None):
    if dictionary is None:
        if feats is None or chars is None:
            raise ValueError("one of inputs must be non-None")
        else:
            if maxlen is None:
                maxlen=len(seq)
            sims = feat_arr_similarity(metric,x,feats)
    else:
        feats = np.zeros((len(dictionary),len(iter(dictionary).next())))
        chars = list(dictionary.keys())
        for i,ch in enumerate(chars):
            feats[i,:] = dictionary[ch]
        sims = feat_arr_similarity(metric,x,feats)
    outseq = np.array(chars)[np.argmax(sims,axis=1)]
    return ''.join(outseq)

def cosine_positive(x,y):
    sims = pairwise.cosine_similarity(x,y)
    return np.maximum(sims,0.0)

def inv_manhattan(x,y):
    return (1.0/(pairwise.manhattan_distances(x,y) + 1e-6))

def manhattan(x,y):
    return pairwise.manhattan_distances(x,y)

@jit(nopython=True)
def full_fast_loop(seqs,features,feature_size,maxlen,is_gapped):
    x = np.zeros((len(seqs),maxlen,feature_size))
    if is_gapped:
        x[:,:,-1] = 1.0 # now extra chars at end will appear as gaps
    for i in range(len(seqs)):
        for j in range(len(seqs[i])):
            try:
                x[i,j,:] = features[seqs[i][j]]
            except:
                x[i,j,:] = features['U']
                x[i,j,:] = features['*']
    return x

meiler_features = AminoFeatures(zscale=True)

class SeqArrayDictConverter:
  
    def __init__(self,features=meiler_features):
        self.features = features
    
    def seqs_to_array(self,seqs,maxlen=None):
        if maxlen is None:
            maxlen=max([len(s) for s in seqs])
            #print(maxlen)
        x = np.zeros((len(seqs),maxlen,self.features.feature_size))
        if self.features.is_gapped:
            x[:,:,-1] = 1.0 # now extra chars at end will appear as gaps
        for i,seq in enumerate(seqs):
            for j,amino in enumerate(seq):
                x[i,j,:] = self.features[amino]
        return x
  
    def fast_seqs_to_array(self,seqs,maxlen):
        return full_fast_loop(
          seqs,
          self.features.fast_feature_dict,
          self.features.feature_size,
          maxlen,
          self.features.is_gapped
        )
  
    def array_to_seqs(self,x,metric):
        seqs = []
        for i in range(len(x)):
            seqs.append(array_to_seq_quick(
                x[i,:,:],
                metric,
                self.features.feat_array,
                self.features.chars
            )           
          )
        return seqs
  
    def array_to_probs(self,x,metric,method='linear'):
        """ probability over AAs at each position in each input `image` 

        should be able to use this to construct a logo of the output
        """
        assert(len(x.shape)==3)
        probs = np.zeros((x.shape[0],x.shape[1],len(self.features.chars)))
        print(probs.shape)
        for i in range(len(x)):
            sims = metric(
                x[i,:,:],
                self.features.feat_array
            )
            print(sims.shape)
            if method=='linear':
                sims_p = sims/np.sum(sims,keepdims=True,axis=1)
            elif method=='softmax':
                print('in softmax: shape=', sims.shape)
                sims_p = special.softmax(sims,axis=1)
                print('in softmax: sims_p shape=', sims_p.shape)
            else:
                raise ValueError('unknown method: {}'.format(method))
            probs[i,:,:] = sims_p
        return probs
    
def plot_weights(x,features,cmap='bwr',vmin=-3,vmax=3,ax=None,figsize=(10,5)):
    if ax is None:
        f,ax = plt.subplots(1,1,figsize=figsize)
    else:
        f = ax.figure
    sm = ax.pcolor(x.T,vmin=vmin,vmax=vmax,cmap=plt.get_cmap(cmap))
    ax.set_yticks(0.5+np.arange(features.feature_size))
    ax.set_yticklabels(features.names)
    cbar = f.colorbar(sm)
    cbar.set_label('weight')
    return f

def plot_probs(probs,features,cmap='Reds',vmin=0,vmax=1.0,ax=None,figsize=(10,5)):
    if ax is None:
        f,ax = plt.subplots(1,1,figsize=figsize)
    else:
        f = ax.figure
    sm = ax.pcolor(probs.T,vmin=vmin,vmax=vmax,cmap=plt.get_cmap(cmap))
    ax.set_yticks(0.5+np.arange(len(features.chars)))
    ax.set_yticklabels(features.chars)
    cbar = f.colorbar(sm)
    cbar.set_label('prob') 
    return f

def get_color_scheme(scheme,mapping_protein_alphabet):
    return weblogo.colorscheme.ColorScheme(
      color_schemes[scheme],
      alphabet=mapping_protein_alphabet,
    )

def logo_from_probs(probs,mapping,color_scheme='chemistry',units='probability',ax=None,figsize=(8,2),return_pdf=False):
    mapping_protein_alphabet = weblogo.seq.Alphabet(
        ''.join(mapping.features.chars), 
        zip(
          ''.join(mapping.features.chars).lower(),
          ''.join(mapping.features.chars)
        )
    )
    logodata = weblogo.logo.LogoData.from_counts(
        mapping_protein_alphabet, 
        np.round(probs*1000).astype(int)
    ) 
    colors = get_color_scheme(color_scheme,mapping_protein_alphabet)
    logooptions = weblogo.LogoOptions(
        show_fineprint = False,
        unit_name = units,
        color_scheme = colors,
    )
    logoformat = weblogo.logo_formatter.LogoFormat(logodata, logooptions)
    png = weblogo.logo_formatter.png_print_formatter(logodata, logoformat)
    image = Image.open(io.BytesIO(png))

    pdf = weblogo.logo_formatter.pdf_formatter(logodata, logoformat)
    #   image = Image.open(io.BytesIO(pdf))
    #   print(type(pdf))

    made_f = False
    if ax is None:
        f,ax = plt.subplots(1,1,figsize=figsize)
        made_f = True
    ax.imshow(image)
    ax.axis('off')

    if not return_pdf:
        if made_f:
            return f
        else:
            return None
    else:
        if made_f:
            return f, pdf
        else:
            return None, pdf




