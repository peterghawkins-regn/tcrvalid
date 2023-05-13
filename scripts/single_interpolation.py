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
import pandas as pd
from umap import UMAP
import matplotlib.pyplot as plt
import seaborn as sns

import sys

from tcrvalid.load_models import *
from tcrvalid.plot_utils import set_simple_rc_params
from tcrvalid.physio_embedding import SeqArrayDictConverter,manhattan
from tcrvalid.interpolation import *
from tcrvalid.defaults import *

set_simple_rc_params()

from_keras=True

tcr1 = 'YYEKEE-ASSRAGNTEAF'
tcr2 = 'SYGVNS-ASSRAGNTEAF'
interp_steps = 100


ref_cdr2s,ref_cdr2_seqs = get_ref_cdr2s(trbv_ref_path)
loaded_models = load_named_models(['0_0','1_2'],chain='TRB',as_keras=from_keras)
loaded_decoders = load_named_models(['0_0','1_2'],chain='TRB',as_keras=from_keras,encoders=False)

cdr2_d_tracker = CDR2DistanceTracker(ref_cdr2s,6,manhattan)
cdr3_d_tracker = CDR3DistanceTracker(None,manhattan)

tracker_tuples = [
  (cdr2_d_tracker,mean_statistic,'CDR2'),
  (cdr3_d_tracker,mean_statistic,'CDR3'),
  (cdr3_d_tracker,mean_startendnorm_statistic,'CDR3_norm')
]

mapping = SeqArrayDictConverter()
interp = Interpolator(mapping)
interp.add_tracker(cdr2_d_tracker,mean_statistic,'CDR2')
interp.add_tracker(cdr3_d_tracker,mean_statistic,'CDR3')
interp.add_tracker(cdr3_d_tracker,mean_startendnorm_statistic,'CDR3_norm')

f,axes = plt.subplots(2,1,figsize=(3,3.5))

cols = {
  '0_0': 'k',
  '1_2': 'r',
}
names = {
  '0_0': 'AE',
  '1_2': r'$\beta=1$'+','+r'$C=2$'
}

for k in ['0_0','1_2']: 
    rd = interp(
        tcr1,
        tcr2,
        loaded_models[k],
        loaded_decoders[k],
        interp_steps,
      )
    axes[0].plot(
        interp.tracked_values['CDR2'],
        color=cols[k],
        label=names[k] #+' ; '+r'$C_{CDR2}=$' + '{:.1f}'.format(rd['CDR2'])
    )
    axes[1].plot(
        interp.tracked_values['CDR3']/(0.5*(interp.tracked_values['CDR3'][0]+ interp.tracked_values['CDR3'][-1])),
        color=cols[k],
        label=names[k] #+' ; '+r'$D_{CDR3,norm}=$' + '{:.1f}'.format(rd['CDR3_norm'])
    )
    print(k, interp.tracked_values['CDR2'])

for ax in axes:
    ax.set_xlim([-0.5,interp_steps+1])
    
axes[0].legend()

axes[0].set_ylim([0,32])
axes[1].set_ylim([0,12])

axes[0].set_ylabel(r'$C^{*}$')
axes[1].set_ylabel(r'$D$')

axes[1].set_xlabel('step '+r'$i$')
axes[0].set_xticklabels([])

plt.tight_layout()
#savefig()