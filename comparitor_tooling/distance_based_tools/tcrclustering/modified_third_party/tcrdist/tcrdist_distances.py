# TCRDIST distance creation with parallelized options
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


import sys
import os

path_ = os.path.dirname(os.path.realpath(__file__))


PATH_TO_TCRDIST = os.path.join(
    path_,
    '../../../lib/tcr-dist'
)
assert os.path.isdir( PATH_TO_TCRDIST )
sys.path.append(PATH_TO_TCRDIST)
import json
# from basic import *
# from all_genes import all_genes, gap_character
# from amino_acids import amino_acids
# from tcr_distances_blosum import blosum, bsd4


rep_dists_path = os.path.join(path_,'../../../data/tcrdist_gene_info/rep_dists.json')
with open(rep_dists_path,'r') as f:
    rep_dists = json.load(f)
                                            
gap_character = '.'

amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', \
               'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']

blosum = {('S', 'W'): -3, ('G', 'G'): 6, ('E', 'M'): -2, ('A', 'N'): -2, ('A', 'Y'): -2, ('W', 'Q'): -2, ('V', 'N'): -3, ('F', 'K'): -3, ('G', 'E'): -2, ('E', 'D'): 2, ('W', 'P'): -4, ('I', 'T'): -1, ('F', 'D'): -3, ('K', 'V'): -2, ('C', 'Y'): -2, ('G', 'D'): -1, ('T', 'N'): 0, ('W', 'W'): 11, ('S', 'S'): 4, ('K', 'C'): -3, ('E', 'F'): -3, ('N', 'L'): -3, ('A', 'K'): -1, ('Q', 'P'): -1, ('F', 'G'): -3, ('D', 'S'): 0, ('C', 'V'): -1, ('V', 'T'): 0, ('H', 'P'): -2, ('P', 'V'): -2, ('I', 'Q'): -3, ('F', 'V'): -1, ('W', 'T'): -2, ('H', 'F'): -1, ('P', 'D'): -1, ('Q', 'R'): 1, ('D', 'Q'): 0, ('K', 'Q'): 1, ('D', 'F'): -3, ('V', 'W'): -3, ('T', 'C'): -1, ('A', 'F'): -2, ('T', 'H'): -2, ('A', 'Q'): -1, ('Q', 'T'): -1, ('V', 'F'): -1, ('F', 'C'): -2, ('C', 'R'): -3, ('V', 'P'): -2, ('H', 'T'): -2, ('E', 'L'): -3, ('F', 'R'): -3, ('I', 'G'): -4, ('C', 'Q'): -3, ('Y', 'V'): -1, ('T', 'A'): 0, ('T', 'V'): 0, ('Q', 'V'): -2, ('S', 'K'): 0, ('K', 'K'): 5, ('E', 'N'): 0, ('N', 'T'): 0, ('A', 'H'): -2, ('A', 'C'): 0, ('V', 'S'): -2, ('Q', 'H'): 0, ('H', 'S'): -1, ('Q', 'Y'): -1, ('P', 'N'): -2, ('I', 'Y'): -1, ('P', 'G'): -2, ('F', 'N'): -3, ('H', 'N'): 1, ('K', 'H'): -1, ('N', 'W'): -4, ('S', 'Y'): -2, ('W', 'N'): -4, ('D', 'Y'): -3, ('E', 'Q'): 2, ('K', 'Y'): -2, ('S', 'G'): 0, ('Y', 'S'): -2, ('G', 'R'): -2, ('A', 'L'): -1, ('A', 'G'): 0, ('T', 'K'): -1, ('T', 'P'): -1, ('M', 'V'): 1, ('Q', 'L'): -2, ('E', 'S'): 0, ('H', 'W'): -2, ('I', 'D'): -3, ('K', 'F'): -3, ('N', 'A'): -2, ('T', 'I'): -1, ('Q', 'N'): 0, ('K', 'W'): -3, ('S', 'C'): -1, ('Y', 'Y'): 7, ('G', 'V'): -3, ('L', 'V'): 1, ('A', 'R'): -1, ('M', 'R'): -1, ('Y', 'L'): -1, ('D', 'C'): -3, ('P', 'P'): 7, ('D', 'H'): -1, ('Q', 'Q'): 5, ('I', 'V'): 3, ('P', 'F'): -4, ('I', 'A'): -1, ('F', 'F'): 6, ('K', 'T'): -1, ('L', 'T'): -1, ('S', 'Q'): 0, ('W', 'F'): 1, ('D', 'A'): -2, ('E', 'Y'): -2, ('K', 'A'): -1, ('Q', 'S'): 0, ('A', 'D'): -2, ('L', 'R'): -2, ('T', 'S'): 1, ('A', 'V'): 0, ('M', 'N'): -2, ('Q', 'D'): 0, ('E', 'P'): -1, ('V', 'V'): 4, ('D', 'N'): 1, ('I', 'S'): -2, ('P', 'M'): -2, ('H', 'D'): -1, ('I', 'L'): 2, ('K', 'N'): 0, ('L', 'P'): -3, ('Y', 'I'): -1, ('N', 'I'): -3, ('T', 'Q'): -1, ('Q', 'F'): -3, ('S', 'M'): -1, ('E', 'R'): 0, ('Q', 'W'): -2, ('G', 'N'): 0, ('L', 'Y'): -1, ('L', 'N'): -3, ('A', 'S'): 1, ('D', 'T'): -1, ('S', 'T'): 1, ('P', 'S'): -1, ('V', 'R'): -3, ('D', 'K'): -1, ('P', 'H'): -2, ('H', 'C'): -3, ('Q', 'I'): -3, ('H', 'H'): 8, ('I', 'I'): 4, ('L', 'W'): -2, ('L', 'L'): 4, ('D', 'R'): -2, ('S', 'I'): -2, ('D', 'I'): -3, ('E', 'A'): -1, ('K', 'I'): -3, ('Q', 'K'): 1, ('T', 'D'): -1, ('A', 'W'): -3, ('Y', 'R'): -2, ('M', 'F'): 0, ('S', 'P'): -1, ('H', 'Q'): 0, ('Y', 'N'): -2, ('I', 'P'): -3, ('E', 'C'): -4, ('H', 'G'): -2, ('P', 'E'): -1, ('Q', 'M'): 0, ('H', 'L'): -3, ('L', 'S'): -2, ('L', 'H'): -3, ('N', 'Q'): 0, ('T', 'Y'): -2, ('K', 'G'): -2, ('S', 'E'): 0, ('Y', 'E'): -2, ('W', 'R'): -3, ('V', 'M'): 1, ('N', 'R'): 0, ('G', 'F'): -3, ('F', 'Y'): 3, ('L', 'Q'): -2, ('M', 'Y'): -1, ('A', 'P'): -1, ('S', 'N'): 1, ('C', 'L'): -1, ('L', 'F'): 0, ('D', 'W'): -4, ('S', 'L'): -2, ('P', 'R'): -2, ('P', 'K'): -1, ('Y', 'G'): -3, ('C', 'K'): -3, ('H', 'K'): -1, ('Q', 'A'): -1, ('I', 'F'): 0, ('K', 'D'): -1, ('N', 'C'): -3, ('L', 'D'): -4, ('Y', 'K'): -2, ('S', 'A'): 1, ('W', 'V'): -3, ('E', 'I'): -3, ('V', 'I'): 3, ('Q', 'C'): -3, ('T', 'G'): -2, ('T', 'L'): -1, ('L', 'M'): 2, ('A', 'T'): 0, ('C', 'H'): -3, ('P', 'Y'): -3, ('S', 'H'): -1, ('H', 'Y'): 2, ('E', 'K'): 1, ('C', 'G'): -3, ('I', 'C'): -1, ('Q', 'E'): 2, ('K', 'R'): 2, ('T', 'E'): -1, ('L', 'K'): -2, ('M', 'W'): -1, ('N', 'Y'): -2, ('N', 'H'): 1, ('V', 'E'): -2, ('Q', 'G'): -2, ('Y', 'D'): -3, ('F', 'Q'): -3, ('G', 'Y'): -3, ('L', 'I'): 2, ('M', 'Q'): 0, ('R', 'A'): -1, ('C', 'D'): -3, ('S', 'V'): -2, ('D', 'D'): 6, ('S', 'D'): 0, ('P', 'C'): -3, ('C', 'C'): 9, ('W', 'K'): -3, ('I', 'N'): -3, ('K', 'L'): -2, ('N', 'K'): 0, ('L', 'G'): -4, ('M', 'S'): -1, ('R', 'C'): -3, ('R', 'D'): -2, ('V', 'A'): 0, ('W', 'I'): -3, ('T', 'T'): 5, ('F', 'M'): 0, ('L', 'E'): -3, ('M', 'M'): 5, ('R', 'E'): 0, ('W', 'H'): -2, ('S', 'R'): -1, ('E', 'W'): -3, ('P', 'Q'): -1, ('H', 'A'): -2, ('Y', 'A'): -2, ('E', 'H'): 0, ('R', 'F'): -3, ('I', 'K'): -3, ('N', 'E'): 0, ('T', 'M'): -1, ('T', 'R'): -1, ('M', 'T'): -1, ('G', 'S'): 0, ('L', 'C'): -1, ('R', 'G'): -2, ('Y', 'M'): -1, ('N', 'F'): -3, ('Y', 'Q'): -1, ('N', 'P'): -2, ('R', 'H'): 0, ('W', 'M'): -1, ('C', 'N'): -3, ('V', 'L'): 1, ('F', 'I'): 0, ('G', 'Q'): -2, ('L', 'A'): -1, ('M', 'I'): 1, ('R', 'I'): -3, ('W', 'L'): -2, ('D', 'G'): -1, ('D', 'L'): -4, ('I', 'R'): -3, ('C', 'M'): -1, ('H', 'E'): 0, ('Y', 'W'): 2, ('G', 'P'): -2, ('W', 'C'): -2, ('M', 'P'): -2, ('N', 'S'): 1, ('G', 'W'): -2, ('M', 'K'): -1, ('R', 'K'): 2, ('D', 'E'): 2, ('K', 'E'): 1, ('R', 'L'): -2, ('A', 'I'): -1, ('V', 'Y'): -1, ('W', 'A'): -3, ('Y', 'F'): 3, ('T', 'W'): -2, ('V', 'H'): -3, ('F', 'E'): -3, ('M', 'E'): -2, ('R', 'M'): -1, ('E', 'T'): -1, ('H', 'R'): 0, ('P', 'I'): -3, ('F', 'T'): -2, ('C', 'I'): -1, ('H', 'I'): -3, ('G', 'T'): -2, ('I', 'H'): -3, ('R', 'N'): 0, ('C', 'W'): -2, ('W', 'G'): -2, ('N', 'M'): -2, ('M', 'L'): 2, ('G', 'K'): -2, ('M', 'G'): -3, ('K', 'S'): 0, ('E', 'V'): -2, ('N', 'N'): 6, ('V', 'K'): -2, ('R', 'P'): -2, ('A', 'M'): -1, ('W', 'E'): -3, ('F', 'W'): 1, ('C', 'F'): -2, ('V', 'D'): -3, ('F', 'A'): -2, ('G', 'I'): -4, ('M', 'A'): -1, ('R', 'Q'): 1, ('C', 'T'): -1, ('W', 'D'): -4, ('H', 'V'): -3, ('S', 'F'): -2, ('P', 'T'): -1, ('F', 'P'): -4, ('C', 'E'): -4, ('H', 'M'): -2, ('I', 'E'): -3, ('G', 'H'): -2, ('R', 'R'): 5, ('K', 'P'): -1, ('C', 'S'): -1, ('D', 'V'): -3, ('M', 'H'): -2, ('M', 'C'): -1, ('R', 'S'): -1, ('D', 'M'): -3, ('E', 'E'): 5, ('K', 'M'): -1, ('V', 'G'): -3, ('R', 'T'): -1, ('A', 'A'): 4, ('V', 'Q'): -2, ('W', 'Y'): 2, ('F', 'S'): -2, ('G', 'M'): -3, ('C', 'P'): -3, ('E', 'G'): -2, ('I', 'W'): -3, ('P', 'A'): -1, ('F', 'L'): 0, ('C', 'A'): 0, ('G', 'L'): -4, ('R', 'V'): -3, ('T', 'F'): -2, ('Y', 'P'): -3, ('M', 'D'): -3, ('G', 'C'): -3, ('R', 'W'): -3, ('N', 'D'): 1, ('N', 'V'): -3, ('V', 'C'): -1, ('A', 'E'): -1, ('Y', 'H'): 2, ('D', 'P'): -1, ('G', 'A'): 0, ('R', 'Y'): -2, ('P', 'W'): -4, ('Y', 'C'): -2, ('P', 'L'): -3, ('F', 'H'): -1, ('I', 'M'): 1, ('Y', 'T'): -2, ('N', 'G'): 0, ('W', 'S'): -3}

## convert from a similarity to a "distance", not necessarily a true metric as reported by the DEV lines in __main__ output
bsd4 = {}

for a in amino_acids:
    for b in amino_acids:
        #print(a,b)
        bab = blosum[(a,b)]
        if a==b:
            bsd4[(a,b)] = 0
        else: ## different
            assert bab<4
            if bab<0:
                bsd4[(a,b)] = 4
            else:
                bsd4[(a,b)] = 4-bab

class DistanceParams:
    def __init__(self, config_string=None ):
        self.gap_penalty_v_region = 4
        self.gap_penalty_cdr3_region = 12 # same as gap_penalty_v_region=4 since weight_cdr3_region=3 is not applied
        self.weight_v_region = 1
        self.weight_cdr3_region = 3
        self.distance_matrix = bsd4
        self.align_cdr3s = False
        self.trim_cdr3s = True
        self.scale_factor = 1.0
        if config_string:
            l = config_string.split(',')
            for tag,val in [x.split(':') for x in l ]:
                if tag == 'gap_penalty_cdr3_region':
                    self.gap_penalty_cdr3_region = float(val)
                elif tag == 'gap_penalty_v_region':
                    self.gap_penalty_v_region = float(val)
                elif tag == 'weight_cdr3_region':
                    self.weight_cdr3_region = float(val)
                elif tag == 'weight_v_region':
                    self.weight_v_region = float(val)
                elif tag == 'scale_factor':
                    self.scale_factor = float(val)
                elif tag == 'align_cdr3s':
                    assert val in ['True','False']
                    self.align_cdr3s = ( val == 'True' )
                elif tag == 'trim_cdr3s':
                    assert val in ['True','False']
                    self.trim_cdr3s = ( val == 'True' )
                else:
                    print('unrecognized tag:',tag)
                    assert False
            print( 'config_string: {} self: {}'.format( config_string, self ) )

    def __str__(self):
        return 'DistanceParams: gap_penalty_v_region= {} gap_penalty_cdr3_region= {} weight_v_region= {} weight_cdr3_region= {} align_cdr3s= {} trim_cdr3s= {}'\
            .format( self.gap_penalty_v_region, self.gap_penalty_cdr3_region,
                     self.weight_v_region, self.weight_cdr3_region,
                     self.align_cdr3s, self.trim_cdr3s )

default_distance_params = DistanceParams()

def blosum_character_distance( a, b, gap_penalty, params ):
    if a== gap_character and b == gap_character:
        return 0
    elif a == '*' and b == '*':
        return 0
    elif a == gap_character or b == gap_character or a=='*' or b=='*':
        return gap_penalty
    else:
        # assert a in amino_acids
        # assert b in amino_acids
        # maxval = min( blosum[(a,a)], blosum[(b,b)] )
        # return maxval - blosum[(a,b)]
        return params.distance_matrix[ (a,b) ]

def blosum_sequence_distance( aseq, bseq, gap_penalty, params ):
    assert len(aseq) == len(bseq)
    dist = 0.0
    for a,b in zip(aseq,bseq):
        if a == ' ':
            assert b== ' '
        else:
            dist += blosum_character_distance( a, b, gap_penalty, params )
    return dist

def align_cdr3s( a, b, gap_character ):
    if len(a) == len(b):
        return (a[:],b[:])

    if len(a)<len(b): ## s0 is the shorter sequence
        s0,s1 = a,b
    else:
        s0,s1 = b,a

    lendiff = len(s1)-len(s0)

    best_score=-1000
    best_gappos=0 # in case len(s0) == 1

    # the gap comes after s0[gappos]

    for gappos in range(len(s0)-1):
        score=0
        for i in range(gappos+1):
            score += blosum[ (s0[i],s1[i]) ]
        for i in range(gappos+1,len(s0)):
            score += blosum[ (s0[i],s1[i+lendiff]) ]
        if score>best_score:
            best_score = score
            best_gappos = gappos
    ## insert the gap
    s0 = s0[:best_gappos+1] + gap_character*lendiff + s0[best_gappos+1:]

    assert len(s0) == len(s1)

    if len(a)<len(b): ## s0 is the shorter sequence
        return ( s0, s1 )
    else:
        return ( s1, s0 )

## align
##
##   shortseq[        ntrim: gappos   ] with longseq[ ntrim: gappos ] and
##   shortseq[ -1*remainder: -1*ctrim ] with longseq[ -1*remainder: -1*ctrim ]
##
## but be careful about negative indexing if ctrim is 0
##
## the gap comes after position (gappos-1) ie there are gappos amino acids before the gap
##
##
## DOES NOT INCLUDE THE GAP PENALTY
##
def sequence_distance_with_gappos( shortseq, longseq, gappos, params ):
    ntrim = 3 if params.trim_cdr3s else 0
    ctrim = 2 if params.trim_cdr3s else 0
    remainder = len(shortseq)-gappos
    dist = 0.0
    count =0
    #print("in d with gappos; ntrim={}, gappos={}".format(ntrim,gappos))
    if ntrim < gappos:
        for i in range(ntrim,gappos):
            #print i,shortseq[i],longseq[i],params.distance_matrix[(shortseq[i],longseq[i])]
            dist += params.distance_matrix[ (shortseq[i], longseq[i] ) ]
            count += 1
    #print 'sequence_distance_with_gappos1:',gappos,ntrim,ctrim,remainder,dist
    if ctrim < remainder:
        for i in range(ctrim, remainder):
            #print -1-i,shortseq[-1-i],longseq[-1-i],params.distance_matrix[(shortseq[-1-i],longseq[-1-i])]
            dist += params.distance_matrix[ (shortseq[-1-i], longseq[-1-i] ) ]
            count += 1
    #print 'sequence_distance_with_gappos2:',gappos,ntrim,ctrim,remainder,dist
    return dist,count


def weighted_cdr3_distance( seq1, seq2, params ):
    shortseq,longseq = (seq1,seq2) if len(seq1)<=len(seq2) else (seq2,seq1)

    ## try different positions of the gap
    lenshort = len(shortseq)
    lenlong = len(longseq)
    lendiff = lenlong - lenshort

#    assert lenshort>3 ##JCC testing
    assert lenshort > 1##JCC testing
    assert lendiff>=0
    if params.trim_cdr3s:
        assert lenshort > 3+2 ## something to align... NOTE: Minimum length of cdr3 protein carried into clones file is currently set in the read_sanger_data.py script!

    if not params.align_cdr3s:
        ## if we are not aligning, use a fixed gap position relative to the start of the CDR3
        ## that reflects the typically longer and more variable-length contributions to
        ## the CDR3 from the J than from the V. For a normal-length
        ## CDR3 this would be after the Cys+5 position (ie, gappos = 6; align 6 rsds on N-terminal side of CDR3).
        ## Use an earlier gappos if lenshort is less than 11.
        ##
        gappos = min( 6, 3 + (lenshort-5)//2 )
        #print("gappos in weighted = ", gappos)
        best_dist,count = sequence_distance_with_gappos( shortseq, longseq, gappos, params )

    else:
        ## the CYS and the first G of the GXG are 'aligned' in the beta sheet
        ## the alignment seems to continue through roughly CYS+4
        ## ie it's hard to see how we could have an 'insertion' within that region
        ## gappos=1 would be a insertion after CYS
        ## gappos=5 would be a insertion after CYS+4 (5 rsds before the gap)
        ## the full cdr3 ends at the position before the first G
        ## so gappos of len(shortseq)-1 would be gap right before the 'G'
        ## shifting this back by 4 would be analogous to what we do on the other strand, ie len(shortseq)-1-4
        min_gappos = 5
        max_gappos = len(shortseq)-1-4
        while min_gappos>max_gappos:
            min_gappos -= 1
            max_gappos += 1
        for gappos in range( min_gappos, max_gappos+1 ):
            dist, count = sequence_distance_with_gappos( shortseq, longseq, gappos, params )
            if gappos>min_gappos:
                assert count==best_count
            if gappos == min_gappos or dist < best_dist:
                best_dist = dist
                best_gappos = gappos
                best_count = count
        #print 'align1:',shortseq[:best_gappos] + '-'*lendiff + shortseq[best_gappos:], best_gappos, best_dist
        #print 'align2:',longseq, best_gappos, best_dist


    ## Note that weight_cdr3_region is not applied to the gap penalty
    ##
    return  params.weight_cdr3_region * best_dist + lendiff * params.gap_penalty_cdr3_region



# def compute_all_v_region_distances( organism, params ):
#     rep_dists = {}
#     for chain in 'AB': # don't compute inter-chain distances
#         repseqs = []
#         for id,g in all_genes[organism].iteritems():
#             if g.chain == chain and g.region == 'V':
#                 merged_loopseq = ' '.join( g.cdrs[:-1])
#                 repseqs.append( ( id, merged_loopseq  ) )
#                 rep_dists[ id ] = {}

#         for r1,s1 in repseqs:
#             for r2,s2 in repseqs:
#                 #if r1[2] != r2[2]: continue
#                 rep_dists[r1][r2] = params.weight_v_region * \
#                                     blosum_sequence_distance( s1, s2, params.gap_penalty_v_region, params )

#     return rep_dists

# def compute_distance(t1,t2,chains,rep_dists,distance_params): #    t1/2 = [ va_reps, vb_reps, l['cdr3a'], l['cdr3b'] ]
#     dist=0.0
#     if 'A' in chains:
#         dist += min( ( rep_dists[x][y] for x in t1[0] for y in t2[0] ) ) +\
#                 weighted_cdr3_distance( t1[2], t2[2], distance_params )
#     if 'B' in chains:
#         dist += min( ( rep_dists[x][y] for x in t1[1] for y in t2[1] ) ) +\
#                 weighted_cdr3_distance( t1[3], t2[3], distance_params )
#     return distance_params.scale_factor * dist

def compute_distance(t1,t2,distance_params): #    t1/2 = [ va_reps, vb_reps, l['cdr3a'], l['cdr3b'] ]
    dist=0.0
    #if 'A' in chains:
    dist += min( ( rep_dists[x][y] for x in t1[0] for y in t2[0] ) ) +\
            weighted_cdr3_distance( t1[2], t2[2], distance_params )
    #if 'B' in chains:
    dist += min( ( rep_dists[x][y] for x in t1[1] for y in t2[1] ) ) +\
            weighted_cdr3_distance( t1[3], t2[3], distance_params )
    return int(distance_params.scale_factor * dist)




# def compute_auc( l0, l1, sign_factor=1 ):
#     ## l0 are the true positives, l1 are the false positives
#     ## if sign_factor==1 then lower scores are better, otherwise it's the opposite
#     ##
#     if not l0:
#         return 0.0, [0,1], [0,0]
#     elif not l1:
#         return 1.0, [0,0,1], [0,1,1]

#     l = [ (sign_factor*x,0) for x in l0 ] + [ (sign_factor*x,-1) for x in l1 ] ## in ties, take the false positive first
#     l.sort()

#     xvals = []
#     yvals = []

#     counts = [0,0]
#     totals = [len(l0),len(l1)]

#     area=0.0
#     width = 1.0/totals[1]
#     for ( score, neg_tcr_class ) in l:
#         tcr_class = -1*neg_tcr_class
#         counts[ tcr_class ] += 1
#         xval = float( counts[1] ) / totals[1]
#         yval = float( counts[0] ) / totals[0]
#         xvals.append( xval )
#         yvals.append( yval )
#         if tcr_class==1: area += yval * width

#     return area,xvals,yvals

# def get_rank( val, l ): ## does not require that the list l is sorted
#     num_lower = 0
#     num_upper = 0

#     epsilon = 1e-6

#     lower_neighbor = val-10000
#     upper_neighbor = val+10000

#     for x in l:
#         if x<val-epsilon:
#             num_lower += 1
#             lower_neighbor = max( lower_neighbor, x )
#         elif x>val+epsilon:
#             num_upper += 1
#             upper_neighbor = min( upper_neighbor, x )

#     total = len(l)
#     num_equal = total - num_lower - num_upper
#     assert num_equal >=0

#     if num_upper == 0:
#         return 100.0

#     elif num_lower == 0:
#         return 0.0

#     else:
#         assert upper_neighbor>lower_neighbor
#         interp = (val-lower_neighbor)/(upper_neighbor-lower_neighbor)

#         #if num_equal>0:print 'num_equal:',num_equal

#         interp_num_lower = num_lower + interp * ( 1 + num_equal )

#         return (100.0*interp_num_lower)/total



# ## negative nbrdist_percentile means take exactly -nbrdist_percentile topn
# def sort_and_compute_nbrdist_from_distances( l, nbrdist_percentile, dont_sort=False ):
#     if not dont_sort: l.sort()
#     assert l[0]<=l[-1]
#     if nbrdist_percentile<0:
#         n = max( 1, min(len(l), -1*nbrdist_percentile ) )
#     else:
#         n = max(1, ( nbrdist_percentile * len(l) )/100 )
#     return sum( l[:n])/float(n)

# ## negative nbrdist_percentile means take exactly -nbrdist_percentile topn
# def sort_and_compute_weighted_nbrdist_from_distances( l, nbrdist_percentile, dont_sort=False ):
#     if not dont_sort: l.sort()
#     assert l[0]<=l[-1]
#     if nbrdist_percentile<0:
#         n = max( 1, min(len(l), -1*nbrdist_percentile ) )
#     else:
#         n = max(1, ( nbrdist_percentile * len(l) )/100 )

#     total_wt = 0.0
#     nbrdist=0.0
#     for i,val in enumerate( l[:n] ):
#         wt = 1.0 - float(i)/n
#         total_wt += wt
#         nbrdist += wt * val

#     return nbrdist / total_wt



# # if __name__ == '__main__': ## hacking
# #     a,b = "DVGYKL DPAGNTGKL".split()
# #     #a,b = "GEGSNNRI GYNTNTGKL".split()
# #     #a,b = "GDRYAQGL GDVDYAQGL".split()
# #     print align_cdr3s( a,b,'.')
# #     exit()


# if __name__ == '__main__':
#     # generate an input file for tcr-dist calculation in C++
#     #
#     params = DistanceParams()

#     for aa in amino_acids:
#         print('AAdist',aa,)
#         for bb in amino_acids:
#             print( '{:.3f}'.format( bsd4[(aa,bb)] ) )
#         #print

#     rep_dists = compute_all_v_region_distances( 'human', params )

#     ## this part only works with the classic db (getting chain from id[2] is bad for gammadelta)
#     vb_genes = [ x for x in rep_dists.keys() if x[2] == 'B' ]
#     vb_genes.sort()

#     print('num_v_genes',len(vb_genes))

#     for v1 in vb_genes:
#         print('Vdist',v1)
#         for v2 in vb_genes:
#             print('{:.3f}'.format(rep_dists[v1][v2]))
        

