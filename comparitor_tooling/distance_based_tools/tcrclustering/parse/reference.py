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

from Bio import SeqIO
import os

DEFAULT_REF_PATH = os.path.join(
    os.path.dirname(__file__),
    '../../data/reference'
)

def bio_seq_to_clean_str(seq):
    s = str(seq)
    s = s.replace('.','')
    return s

def grab_fasta_dict(fasta_path):
    record = SeqIO.index(fasta_path, "fasta")
    r_dict = {
        k.split('|')[1]:bio_seq_to_clean_str(v._seq)
        for k,v in record.items()
    }
    return r_dict

def collect_gene_from_dict(gene,g_dict):
        # TODO - IF DV in gene, but no slash before it - add it
        #print("input: ",gene)
        try:
            v = g_dict[gene]
        except KeyError:
            if '*' not in gene:
                try:
                    #print("trying: ",gene+'*01')
                    v = g_dict[gene+'*01']
                except KeyError:
                    try:
                        close_keys = [k for k in g_dict.keys() 
                                      if k.startswith(gene.split('*')[0]) and k.endswith('*01')]
                        #print("last: ",close_keys[0])
                        v = g_dict[close_keys[0]]
                    except IndexError:
                        try:
                            close_keys = [k for k in g_dict.keys() 
                                          if k.startswith(gene.split('*')[0].split('-')[0]) and k.endswith('*01')]
                            #print("last: ",close_keys[0])
                            v_try = close_keys[0]
                            v = g_dict[close_keys[0]]
                        except IndexError:
                            v=None
            else:
                try:
                    close_keys = [k for k in g_dict.keys() 
                                  if k.startswith(gene.split('*')[0]) and k.endswith('*01')]
                    #print("last: ",close_keys[0])
                    v = g_dict[close_keys[0]]
                except IndexError:
                    try:
                        close_keys = [k for k in g_dict.keys() 
                                      if k.startswith(gene.split('*')[0].split('-')[0]) and k.endswith('*01')]
                        #print("last: ",close_keys[0])
                        v_try = close_keys[0]
                        v = g_dict[close_keys[0]]
                    except IndexError:
                        v=None
        return v 
    
def collect_gene_name_from_dict(gene,g_dict):
    # TODO - IF DV in gene, but no slash before it - add it
    #print("input: ",gene)
    if gene is None:
        return None
    try:
        v = g_dict[gene]
        v_name = gene
    except KeyError:
        if '*' not in gene:
            try:
                #print("trying: ",gene+'*01')
                v = g_dict[gene+'*01']
                v_name = gene+'*01'
            except KeyError:
                try:
                    # not sure this case ever really used, but leave for now
                    close_keys = [k for k in g_dict.keys() 
                                  if k.startswith(gene.split('*')[0]) and k.endswith('*01')]
                    #print("last: ",close_keys[0])
                    v = g_dict[close_keys[0]]
                    v_name = close_keys[0]
                except IndexError:
                    try:
                        # if has "01" or something instead of "1" after the family, e.g 12-03
                        # but also the case like TRBV06-01 instead of TRBV6-1 - both the 6 and 1 should not have leading 0
                        # split around '-' and remove TRBV from 0th. if 2 chars and first is 0, remove the zero. or int() then format.
                        split = gene.split('-')
                        family = split[0][4:]
                        subfamily = split[1]
                        #print(family, subfamily)
                        if family[0]=='0':
                            family=family[1:]
                        if subfamily[0]=='0':
                            subfamily=subfamily[1:]
                        #print(family, subfamily)
                        gene_try = 'TRBV'+family+'-'+subfamily+'*01'
                        #print(gene_try)
                        v = g_dict[gene_try]
                        v_name = gene_try
                    except (IndexError,KeyError):
                        try:
                            # cases from same family, but different after dash
                            # this may not be reasonable, but should be somewhat close
                            close_keys = [k for k in g_dict.keys() 
                                          if k.startswith(gene.split('*')[0].split('-')[0]) and k.endswith('*01')]
                            #print("last: ",close_keys[0])
                            v_try = sorted(close_keys)[0] #sorted so always the same
                            v = g_dict[v_try]
                            v_name = v_try
                        except IndexError:
                            v_name=None
        else:
            try:
                close_keys = [k for k in g_dict.keys() 
                              if k.startswith(gene.split('*')[0]) and k.endswith('*01')]
                #print("last: ",close_keys[0])
                v = g_dict[close_keys[0]]
                v_name = close_keys[0]
            except IndexError:
                try:
                    close_keys = [k for k in g_dict.keys() 
                                  if k.startswith(gene.split('*')[0].split('-')[0]) and k.endswith('*01')]
                    #print("last: ",close_keys[0])
                    v_try = sorted(close_keys)[0]
                    v = g_dict[v_try]
                    v_name = v_try
                except IndexError:
                    v_name=None
    return v_name
    
class ReferenceAAMapper:
    
    def __init__(self,
                 chain='TRB',
                 ref_path=DEFAULT_REF_PATH
                ):
        self.v_dict = grab_fasta_dict(
                            os.path.join(
                                ref_path,
                                chain+'V_aa.fasta'
                                )
        )
        self.j_dict = grab_fasta_dict(
                            os.path.join(
                                ref_path,
                                chain+'J_aa.fasta'
                                )
        )
        
    def collect_v_name(self,v_gene):
        return collect_gene_name_from_dict(v_gene,self.v_dict)
        
    def collect_j_name(self,j_gene):
        return collect_gene_name_from_dict(j_gene,self.j_dict)
        
    def collect_v_aa(self,v_gene):
        return collect_gene_from_dict(v_gene,self.v_dict)
#         print("input: ",v_gene)
#         try:
#             v = self.v_dict[v_gene]
#         except KeyError:
#             if '*' not in v_gene:
#                 try:
#                     print("trying: ",v_gene+'*01')
#                     v = self.v_dict[v_gene+'*01']
#                 except KeyError:
#                     try:
#                         close_keys = [k for k in self.v_dict.keys() 
#                                       if k.startswith(v_gene.split('*')[0]) and k.endswith('*01')]
#                         print("last: ",close_keys[0])
#                         v = self.v_dict[close_keys[0]]
#                     except IndexError:
#                         try:
#                             close_keys = [k for k in self.v_dict.keys() 
#                                           if k.startswith(v_gene.split('*')[0].split('-')[0]) and k.endswith('*01')]
#                             print("last: ",close_keys[0])
#                             v_try = close_keys[0]
#                             v = self.v_dict[close_keys[0]]
#                         except IndexError:
#                             v=None
#             else:
#                 try:
#                     close_keys = [k for k in self.v_dict.keys() 
#                                   if k.startswith(v_gene.split('*')[0]) and k.endswith('*01')]
#                     print("last: ",close_keys[0])
#                     v = self.v_dict[close_keys[0]]
#                 except IndexError:
#                     try:
#                         close_keys = [k for k in self.v_dict.keys() 
#                                       if k.startswith(v_gene.split('*')[0].split('-')[0]) and k.endswith('*01')]
#                         print("last: ",close_keys[0])
#                         v_try = close_keys[0]
#                         v = self.v_dict[close_keys[0]]
#                     except IndexError:
#                         v=None
#         return v
        
#         if '*' not in v_gene:
#             v_gene+='*01'
#         return self.v_dict[v_gene]
            
    def collect_j_aa(self,j_gene):
        return collect_gene_from_dict(j_gene,self.j_dict)
#         if '*' not in j_gene:
#             j_gene+='*01'
#         return self.j_dict[j_gene]

    def join_v_cdr3(self,v_gene,cdr3):
        """ v gene name + cdr as aa """
        v = self.collect_v_aa(v_gene)
        if v is None:
            return None
        p=6
        tmp_p = 6
        while tmp_p > 0:
            f_idx = v[-p:].find(cdr3[:tmp_p])
            if f_idx!=-1:
                return v[:-(p-f_idx)]+cdr3
            tmp_p-=1
        print("WARNING: could not map V&CDR3 !!!!!!!!")
        print("v = ",v)
        print('cdr3 = ', cdr3)
        return None
    
    def join_vcdr_j(self,vcdr,j_gene):
        """ vcdr3: str of v+cdr3 matched, v_gene  is gene name"""
        j = self.collect_j_aa(j_gene)
        if j is None:
            return None
        p=13
        tmp_p = 13
        while tmp_p > 0:
            f_idx = j[:p].find(vcdr[-tmp_p:])
            if f_idx!=-1:
                return vcdr+j[f_idx:]
            tmp_p-=1
        print("WARNING: could not map VCDR3 to J !!!!!!!!")
        print('vcdr = ',vcdr)
        print('j = ',j)
        return None
    
    def __call__(self,v_gene,j_gene,cdr3):
        vcdr = self.join_v_cdr3(v_gene,cdr3)
        if vcdr is None:
            return None
        vcdr = self.join_vcdr_j(vcdr,j_gene)
        return vcdr
            
def prepare_df_genes(df,chains=['TRB','TRA']):
    reference_mappers = {ch:ReferenceAAMapper(chain=ch) for ch in chains}
    for locus in chains:
        df = df[df['v_gene_'+locus].map(lambda x: x.startswith(locus))] #QC step 0
        df['v_gene_'+locus] = df['v_gene_'+locus].map(lambda x : reference_mappers[locus].collect_v_name(x))
        df['j_gene_'+locus] = df['j_gene_'+locus].map(lambda x : reference_mappers[locus].collect_j_name(x))
        df = df[df['cdr3_'+locus].map(lambda x: len(x)>5)] # extra QC step
        df = df[~df['v_gene_'+locus].isna()] # TCRdist lookup reqiures a V that matches reference
    return df