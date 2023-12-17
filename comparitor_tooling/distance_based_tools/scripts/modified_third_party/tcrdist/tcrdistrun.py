# TCRdist run script with only key elements being called
#
# TCRdist is used under its license and copyright
#
# MIT License
#
# Copyright (c) 2017 Philip Harlan Bradley and Jeremy Chase Crawford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# from basic import *
import sys
import os

## what gets checked to decide if we re-run certain steps?
##
## _*.dist
## _random_nbrdists.tsv
## _cdr3_motifs_{epitope}.log
## _tree_AB_*png
##
PATH_TO_TCRDIST = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    '..',
    '..',
    '..',
    'lib',
    'tcr-dist'
)
assert os.path.isdir( PATH_TO_TCRDIST )
sys.path.append(PATH_TO_TCRDIST)
from basic import *

MAX_MOTIFS_TIME_IN_SECONDS = 24 * 60 * 60 ## a day

with Parser(locals()) as p:
    p.str('pair_seqs_file').described_as('Name of the pair_seqs file (input option #1)')
    p.str('parsed_seqs_file').described_as('Name of the parsed_seqs file (input option #2)')
    p.str('clones_file').described_as('Name of the clones file (input option #3)')
    p.flag('only_parsed_seqs')
    p.flag('find_cdr3_motifs_in_parallel').described_as('Will spawn multiple simultaneous CDR3 motif finding jobs')
    p.flag('only_clones')
    p.flag('constant_seed')
    p.str('extra_make_really_tall_trees_args').shorthand('mrtt_args')
    p.str('make_tall_trees_color_scheme').default('probs')
    p.str('extra_find_clones_args').shorthand('fc_args')
    p.str('organism').required()
    p.str('webdir').described_as('Location where the index.html summary output file will be generated. Default is <clones_file>_web/')
    p.str('distance_params')
    p.int('min_quality_for_singletons').default(20).described_as('Minimum CDR3 region quality score for singleton clones')
    p.float('seed_threshold_for_motifs')
    p.flag('make_fake_quals').described_as("(for --pair_seqs_file I/O, arg is passed to read_pair_seqs.py) Create a fictitious quality string for each nucleotide sequence")
    p.flag('make_fake_ids').described_as("(for --pair_seqs_file I/O, arg passed to read_pair_seqs.py) Create an id for each line based on index in the pair_seqs file")
    p.flag('make_fake_alpha').described_as("Create a fictitious alpha chain sequence")
    p.flag('make_fake_beta').described_as("Create a fictitious beta chain sequence")
    p.flag('force')
    p.flag('borderline_motifs').described_as("Option to relax the CDR3 motif finding significance thresholds; may be helpful for small datasets")
    p.flag('webstatus')
    p.flag('dry_run')
    p.flag('intrasubject_nbrdists').described_as('Include TCRs from the same subject when computing the nbrdist (aka NNdistance) score')
    p.flag('consistentfigcolors')
    p.flag('no_probabilities').described_as('Assign a probability of 1 to all TCRs.')
    p.str('dist_chains').default('A B AB').described_as('The chains over which the distance calcn will be performed; default is all three ("A B AB")')
    p.float('radius_factor').default(0.0)
    p.set_help_prefix("""

################
INPUT
################

This script will run a pipeline of analysis tools starting from three possible input filetypes:

    pair_seqs_file: A .tsv (tab-separated values) file with info on alpha and beta chain
       sequences and quality scores, epitope, subject, and id.
       Required fields: id epitope subject a_nucseq b_nucseq a_quals b_quals
       For further details, run "python read_pair_seqs.py -h"
       In particular,
          * if you don't have quality score info you can add --make_fake_quals and fictitious
            quality scores will be created.
          * if you don't have ids, you can add --make_fake_ids and default ids will be created based
            on position in the file.


    parsed_seqs_file: A processed sequence file with V and J genes assigned, CDR3s parsed, etc. Will be produced
       as an intermediate when you start from a pair_seqs_file and could be then reused to re-run downstream
       analyses. Could also be generated from the output of other tools (eg MIXCR, conversion scripts yet to come).

    clones_file: A processed version of the parsed_seqs_file in which per-TCR probabilities have been assigned and
       clones identified. One line per clone. Most of the scripts in the pipeline take a clones_file as input.

    Use the corresponding command line option to point to the type of file you have.


#################
OUTPUT
#################

    Running the pipeline will generate a slew of different files. The best place to start will be the html output
    which can be found in the file

    <clones_file>_web/index.html

    where <clones_file> is the name of the clones_file produced by or input to the pipeline. To change the directory
    where index.html will live, use the --webdir command line option.


#################
NOTES
#################

    The pipeline has been tested on datasets as large as 5000 sequences, but it does start to get slow.
    Eventually we would like to code some of the time-intensive steps in C/C++/other. If this would help
    you, let me know!

    CDR3 motif finding in particular is slow for larger datasets. If you have multiple cores available,
    consider using the option:

       --find_cdr3_motifs_in_parallel

    Also, be aware of the --min_quality_for_singletons flag which can cause some (singleton) TCR clones
    to be filtered out if they have bad sequence read quality scores. The default is 20.


#################
SUPPORT/FEEDBACK
#################

    This analysis pipeline is a work in progress. Please direct questions/suggestions to

    pbradley@fredhutch.org

    Thank you!


""")

## imports are slow if we only wanted the --help output, so do these now
try:
    import numpy
except:
    print '[ERROR] Failed to import the python module scipy-- is it installed? I really need it.'
    exit()

try:
    import scipy
except:
    print '[ERROR] Failed to import the python module scipy-- is it installed? I really need it.'
    exit()

try:
    import matplotlib
except:
    print '[ERROR] Failed to import the python module matplotlib-- is it installed? I really need it.'
    exit()

try:
    import sklearn
except:
    print """
=============================================================================
[ERROR] failed to import the python module sklearn (scikit-learn)
[ERROR] Some analyses (kernelPCA plots, adjusted_mutual_information) will fail
[ERROR] Take a look at http://scikit-learn.org/stable/install.html
=============================================================================
"""

import time
import os
import subprocess
import sys
# from paths import path_to_scripts, path_to_tablesorter_files

path_to_scripts = PATH_TO_TCRDIST
path_to_tablesorter_files = path_to_scripts+'/external/tablesorter'

# print path_to_scripts
# print path_to_tablesorter_files

if pair_seqs_file:
    assert not parsed_seqs_file
    if not os.path.isfile(pair_seqs_file):
        print "Error: file " + pair_seqs_file + " does not exist."
        sys.exit()
    else:
        #checkinput(pair_seqs_file)
        #pair_seqs_file = pair_seqs_file[:-4]+ "_cleanedinput.tsv"
        parsed_seqs_file = pair_seqs_file[:-4]+'_parsed_seqs.tsv'

if parsed_seqs_file:
    assert not clones_file
    if not pair_seqs_file:
        if not os.path.isfile(parsed_seqs_file):
            print "Error: file " + parsed_seqs_file + " does not exist."
            sys.exit()
        #else:
            #checkinput(parsed_seqs_file)
            #parsed_seqs_file = parsed_seqs_file[:-4]+ "_cleanedinput.tsv"
    probs_file = parsed_seqs_file[:-4]+'_probs.tsv'
    clones_file = '{}_mq{}_clones.tsv'.format( probs_file[:-4], min_quality_for_singletons )

if distance_params:
    distance_params_args = ' --distance_params {} '.format( distance_params )
else:
    distance_params_args = ' '

if constant_seed:
    constant_seed_args = ' --constant_seed '
else:
    constant_seed_args = ' '

if not webdir:
    webdir = '{}_web'.format(clones_file[:-4])

if webdir.endswith('/'):webdir = webdir[:-1]

if not exists(webdir) and not only_parsed_seqs and not only_clones:
    os.mkdir(webdir)


if not only_parsed_seqs and not only_clones:
    files = glob(path_to_tablesorter_files+'/*')
    for file in files:
        #print 'copying tablesorter file:',file
        system('cp {} {}'.format( file, webdir ) )

webfile = '{}/index.html'.format(webdir)

if not ( only_clones or only_parsed_seqs ):
    print '\nWill generate summary output file: {}\n'.format(webfile)

webdir_contains_input_files = ( os.path.dirname(os.path.normpath(os.path.realpath( webfile ))) ==
                                os.path.dirname(os.path.normpath(os.path.realpath( clones_file ))) )

if webstatus:
    out = open(webfile,'w')
    out.write("""<!doctype html>
<title>Running</title>
<h1>Analysis is in progress, sorry for the delay, try reloading from time to time</h1>
""")
    out.close()

all_logfiles = []
all_errfiles = []

def run(cmd):
#     print cmd
    if not dry_run:
        print cmd
        if webstatus: ## we want a continuously updating index.html file
            outwebstatus = open(webfile,'a')
            outwebstatus.write('<h2>Running:</h2>\n{}<br><br>\n'.format(cmd))
            outwebstatus.close()
        system(cmd)
        cmdl = cmd.split()
        if len(cmdl)>=4 and cmdl[-4] == '>' and cmdl[-2] == '2>':
            logfile = cmdl[-3]
            errfile = cmdl[-1]
            all_logfiles.append( logfile )
            all_errfiles.append( errfile )
            if webstatus:
                errlines = '<br>'.join( popen('tail '+errfile).readlines() )
                outwebstatus = open(webfile,'a')
                outwebstatus.write('<i>Last few stderr lines from run:</i><br><br>{}<br><br>\n'\
                                   .format(errlines))
                outwebstatus.close()

# print "here"

if pair_seqs_file and ( force or not exists( parsed_seqs_file ) ):
#     print "read in file"
    cmd = 'python {}/read_pair_seqs.py {} {} {} {} --organism {} --infile {} --outfile {} -c > {}.log 2> {}.err'\
          .format( path_to_scripts,
                   ' --make_fake_ids ' if make_fake_ids else '',
                   ' --make_fake_quals ' if make_fake_quals else '',
                   ' --make_fake_alpha ' if make_fake_alpha else '',
                   ' --make_fake_beta ' if make_fake_beta else '',
                   organism, pair_seqs_file, parsed_seqs_file, parsed_seqs_file, parsed_seqs_file )
    run( cmd )

if only_parsed_seqs:
    exit()

if parsed_seqs_file and ( force or not exists( clones_file ) ):
    if no_probabilities:
        noprobsarg = "--no_probabilities"
    else:
        noprobsarg = " "
    ## compute probs
    if force or not exists( probs_file ):
        cmd = 'python {}/compute_probs.py --organism {}  --infile {} --outfile {} {}  -c --filter --add_masked_seqs > {}.log 2> {}.err'\
              .format( path_to_scripts, organism, parsed_seqs_file, probs_file, noprobsarg, probs_file, probs_file )
        run(cmd)

    ## find the clones
    if force or not exists( clones_file ) or extra_find_clones_args:
        cmd = 'python {}/find_clones.py {} --organism {}  --infile {} --outfile {}  -c --min_quality_for_singletons {} > {}.log 2> {}.err'\
            .format( path_to_scripts, extra_find_clones_args if extra_find_clones_args else ' ',
                     organism, probs_file, clones_file, min_quality_for_singletons, clones_file, clones_file )
        run(cmd)

assert exists( clones_file )

if only_clones:
    exit()

all_clones = parse_tsv_file( clones_file, ['epitope','subject'], ['cdr3a'], False )

epitopes = all_clones.keys()[:]
epitopes.sort()


## make a mouse table

cmd = 'python {}/make_mouse_table.py --clones_file {} > {}_mmt.log 2> {}_mmt.err'\
      .format( path_to_scripts, clones_file, clones_file, clones_file )
run(cmd)


## precompute some info on gene frequencies
cmd = 'python {}/analyze_gene_frequencies.py --organism {}  --clones_file {} > {}_agf.log 2> {}_agf.err'\
      .format( path_to_scripts, organism, clones_file, clones_file, clones_file )
run(cmd)


## make gene plots (entropy, relentropy, ami, covariation, pie charts of gene usage) and VJ pairings
# cmd = 'python {}/make_gene_plots.py {} --organism {}  --clones_file {} --use_color_gradients > {}_mgp.log 2> {}_mgp.err'\
#     .format( path_to_scripts, ' --consistentfigcolors '*consistentfigcolors,
#              organism, clones_file, clones_file, clones_file )
# run(cmd)


## compute distances
#distfiles = glob('{}_*.dist'.format(clones_file[:-4]))

cmd = 'python {}/compute_distances.py {} {} --organism {} --clones_file {} --dist_chains {} > {}_cd.log 2> {}_cd.err'\
    .format( path_to_scripts, distance_params_args, ' --intrasubject_nbrdists '*intrasubject_nbrdists,
             organism, clones_file, dist_chains, clones_file, clones_file )
run(cmd)
# if force or not distfiles: run(cmd)

## this prints out the tcrdist clusters
## TODO: adapt make_tall_trees to save file of clone_id -> cluster number?
## make tall trees
## --junction_bars removed from the command
if 'AB' in dist_chains:
    ABs = 'AB'
elif 'A' in dist_chains:
    ABs = 'A'
elif 'B' in dist_chains:
    ABs = 'B'
else:
    raise ValueError('Unknown dist_chains value : {}'.format(dist_chains))

cmd = 'python {}/make_tall_trees.py {} --organism {} --color_scheme {} --clones_file {} --ABs {} --radius_factor {} > {}_mtt.log 2> {}_mtt.err'\
      .format( path_to_scripts, constant_seed_args, organism, make_tall_trees_color_scheme,
               clones_file, ABs, radius_factor, clones_file, clones_file )
run(cmd)