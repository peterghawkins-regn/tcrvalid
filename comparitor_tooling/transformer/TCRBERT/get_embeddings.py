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
import os
import sys
import logging
import argparse
import collections
import pandas as pd
import anndata as ad
import scanpy as sc


sys.path.append("lib/tcr-bert/tcr")

import featurization as ft
import model_utils
import utils



def build_parser():
    """CLI parser"""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "infile",
        type=str,
        help="Input file. If column-delimited, assume first column is sequences",
    )
    parser.add_argument("outfile", type=str, help="Output file to write")
    parser.add_argument(
        "-m",
        "--mode",
        choices=["B", "AB"],
        type=str,
        default="B",
        help="Input TRB or TRA/TRB pairs",
    )
    parser.add_argument(
        "--transformer",
        type=str,
        default="wukevin/tcr-bert",
        help="Path to transformer or huggingface model identifier",
    )
    parser.add_argument(
        "-l", "--layer", type=int, default=-1, help="Transformer layer to use"
    )
    parser.add_argument(
        "-r", "--res", type=float, default=32, help="Leiden clustering resolution"
    )
    parser.add_argument(
        "-g",
        "--gpu",
        required=False,
        default=None,
        type=int,
        help="GPU to run on. If not given or no GPU available, default to CPU",
    )

    return parser

parser = build_parser()
args = parser.parse_args()

embeddings = None
if args.mode == "B":
    trbs = utils.dedup(
        [trb.split("\t")[0] for trb in utils.read_newline_file(args.infile)]
    )
    trbs = [x for x in trbs if ft.adheres_to_vocab(x)]
    logging.info(f"Read in {len(trbs)} unique valid TCRs from {args.infile}")
    obs_df = pd.DataFrame(trbs, columns=["TCR"])
    embeddings = model_utils.get_transformer_embeddings(
        model_dir=args.transformer,
        seqs=trbs,
        layers=[args.layer],
        method="mean",
        device=args.gpu,
    )
elif args.mode == "AB":
    raise NotImplementedError
assert embeddings is not None
#print(type(embeddings))
#print(embeddings)
#try it with the eighth layer for -l
#try the embed_and_cluster function for the labeled datasets
#Write descriptive text on QC pipeline

df = pd.DataFrame(embeddings)
df.to_csv(args.outfile)
print("Successfully executed")