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
import matplotlib as mpl

def set_simple_rc_params():
    mpl.rcParams['lines.linewidth'] = 2
    mpl.rcParams['font.weight'] = 'bold'
    mpl.rcParams['font.size'] = 8.0

    mpl.rcParams['axes.facecolor'] = 'w'
    mpl.rcParams['axes.edgecolor'] = 'k'
    mpl.rcParams['axes.linewidth'] = 1.5
    mpl.rcParams['axes.grid'] = False
    mpl.rcParams['axes.labelweight'] = 'bold'
    mpl.rcParams['axes.labelsize'] = 'medium'


    mpl.rcParams['xtick.bottom'] = True
    mpl.rcParams['ytick.left'] = True

    mpl.rcParams['xtick.major.size'] = 5.0
    mpl.rcParams['ytick.major.size'] = 5.0

    mpl.rcParams['xtick.labelsize'] = 8.0
    mpl.rcParams['ytick.labelsize'] = 8.0
    mpl.rcParams['legend.fontsize'] = 'small'

    mpl.rcParams['axes.titlesize'] = 'medium'
    mpl.rcParams['axes.titleweight'] = 'bold'