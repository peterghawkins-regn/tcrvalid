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
import json
import shutil
from Bio.pairwise2 import align
from Bio.SubsMat import MatrixInfo as matlist


BLOSUM = matlist.blosum62


AA_TO_NUC = {
    "A": "GCT", #list("GCT,GCC,GCA,GCG".split(",")),
    "R": "CGT", #list("CGT,CGC,CGA,CGG,AGA,AGG".split(",")),
    "N": "AAT", #list("AAT,AAC".split(",")),
    "D": "GAT", #list("GAT,GAC".split(",")),
    "C": "TGT", #list("TGT,TGC".split(",")),
    "Q": "CAA", # list("CAA,CAG".split(",")),
    "E": "GAA", #list("GAA,GAG".split(",")),
    "G": "GGT", #list("GGT,GGC,GGA,GGG".split(",")),
    "H": "CAT,", #list("CAT,CAC".split(",")),
    "I": "ATT", #list("ATT,ATC,ATA".split(",")),
    "L": "TTA", #list("TTA,TTG,CTT,CTC,CTA,CTG".split(",")),
    "K": "AAA", #list("AAA,AAG".split(",")),
    "M": "ATG", #list("ATG".split(",")),
    "F": "TTT", #list("TTT,TTC".split(",")),
    "P": "CCT", #list("CCT,CCC,CCA,CCG".split(",")),
    "S": "TCT", #list("TCT,TCC,TCA,TCG,AGT,AGC".split(",")),
    "T": "ACT", #list("ACT,ACC,ACA,ACG".split(",")),
    "W": "TGG", #list("TGG".split(",")),
    "Y": "TAT", #list("TAT,TAC".split(",")),
    "V": "GTT" #list("GTT,GTC,GTA,GTG".split(",")),
}

def load_json_to_dict(json_path):
    with open(json_path) as f:
        json_dict = json.load(f)
    return json_dict

def seq_pair_blosum_score(seq1,seq2,score_only=True,gap=-6):
    return align.globalds(seq1,seq2,
                          BLOSUM,
                          gap,gap,
                          score_only=score_only)

def move_file(src,dest):
    shutil.move(src,dest)
    
