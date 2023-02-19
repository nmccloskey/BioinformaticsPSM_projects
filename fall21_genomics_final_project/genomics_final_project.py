# genomics final project

# import modules
from Bio import Entrez
from Bio import SeqIO
import os
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 5'-3'
tel = 'TTAGGG'

class Species():
# species class to contain sequences and calculate overall telomeric proportion of genome
    def __init__(self,name=None,accns=None):
        self.name = name # species name
        self.accns = accns # accession numbers
        self.seqs = [] # list of sequence objects
        self.sorted_seqs = [] # list of sorted sequence objects
        self.spec_total_tel_seg_length = 0 # total length of telomeric segments of all sequences in bp
        self.spec_tel_prop = 0 # telomeric proportion of genome
    
    def extract_seqrecs(self):
        # fetch data from NCBI and write relevant info to text file
        path = os.getcwd() + '\\seq_data\\' + self.name
        if os.path.isdir(path):
            num = len(os.listdir(path))
        else:
            num = 0
            os.mkdir(path)
        handle = Entrez.efetch(db="nucleotide", id=self.accns, rettype="fasta", retmode="text", email='nsm@temple.edu')
        print('seqrecs obtained')
        for i,seqrec in enumerate(SeqIO.parse(handle,'fasta')):
            print('creating new seq object')
            seq = Seq()
            org_name = self.name
            seq.extract_info(i,num,org_name,path,seqrec)
        handle.close()

    def process_seqrecs(self):
        # process info from text files written by above function
        # allows data processing without waiting for Entrez
        path = os.getcwd() + '\\seq_data\\' + self.name
        for fname in os.listdir(path):
            fo = open(os.path.join(path, fname),'r')
            data = fo.readlines()
            sid = data[1].rstrip('\n')
            length = int(data[2].rstrip('\n'))
            tel_count = int(data[3].rstrip('\n'))
            ind_string = data[4]
            tel_inds = []
            temp = []
            for m in re.finditer(r'[0-9]+',ind_string):
                temp.append(int(m.group()))
                if len(temp) == 2:
                    tel_inds.append(tuple(temp))
                    temp = []
            # create sequence object with these features
            seq = Seq(sid=sid, length=length, tel_count=tel_count, tel_inds=tel_inds)
            # let sequence fill itself out
            seq.process_info()
            # append to species
            self.seqs.append(seq)
            fo.close()
        # determine telomeric proportion of genome
        species_tel_seg_length = 0
        species_seq_length = 0
        for seq in self.seqs:
            species_tel_seg_length += seq.total_seq_tel_seg_length
            species_seq_length += seq.length
        self.spec_total_tel_seg_length = species_tel_seg_length
        self.spec_tel_prop = species_tel_seg_length / species_seq_length

    def sort_seqrecs(self,attribute,reverse=True):
        # sort sequences by an attribute
        if attribute == 'length':
            self.seqs.sort(key=lambda x: x.length, reverse=reverse)
        if attribute == 'sid':
            self.seqs.sort(key=lambda x: x.sid, reverse=reverse)
        if attribute == 'total_seq_tel_seg_length':
            self.seqs.sort(key=lambda x: x.total_seq_tel_seg_length, reverse=reverse)
        if attribute == 'seq_tel_prop':
            self.seqs.sort(key=lambda x: x.seq_tel_prop, reverse=reverse)
        # sort by chromosome per order of accession numbers
        if attribute == 'chrom':
            for accn,seq in zip(self.accns,self.seqs):
                if accn == seq.sid:
                    self.sorted_seqs.append(seq)

class Seq():
# sequence class to contain info of individual sequences and calculate telomeric proportion of sequence
    def __init__(self,sid=None,length=0,tel_count=0,tel_inds=[]):
        # from extract info
        self.sid = sid # sequence ID
        self.length = length # length of sequence (bps)
        self.tel_count = tel_count # count of instances of TTAGGG in sequence
        self.tel_inds = tel_inds # indices of (TTAGGG)+
        # from process info
        self.tel_seg_inds = [] # indices of segments repeating at least k times
        self.tel_seg_count = 0 # count of segments consisting of at least k repeats
        self.total_seq_tel_seg_length = 0 # total telomeric length of sequence
        self.seq_tel_prop = 0 # proportion of sequence that is telomeric
        
    def extract_info(self,i,num,org_name,path,seqrec):
        # find seqID, length, count of TTAGGG, and indices of (TTAGGG)+
        s = str(seqrec.seq)
        print('processing string')
        self.length = len(s)
        self.sid = seqrec.id
        self.tel_count = len(re.findall(tel,s))
        # find segments of at least 1 TTAGGG
        for m in re.finditer(r'(TTAGGG)+',s):
            self.tel_inds.append(m.span())
        # write info to file
        fname = org_name + '_seq_' + str(i+num+1) + '_accn_' + str(self.sid) + '.txt'
        with open(os.path.join(path,fname),'w') as file:
            file.write(org_name+'\n'+str(self.sid)+'\n'+str(self.length)+'\n'+str(self.tel_count)+'\n'+str(self.tel_inds))
        
    def process_info(self):       
        # determine tel_seg_inds, tel_seg_count, total telomeric length, and telomeric proportion of sequence for a given k value
        for span in self.tel_inds:
            seg = span[1] - span[0]
            if seg / 6 >= k:
                self.tel_seg_inds.extend([span[0],span[1]])
                self.total_seq_tel_seg_length += seg
        self.tel_seg_count = len(self.tel_seg_inds)
        self.seq_tel_prop = self.total_seq_tel_seg_length / self.length

# create lists of accession numbers
accns_rn=['NC_0513%d.1'%i for i in range(36,58)]
accns_rr=['NC_0461%d.1'%i for i in range(54,74)]
accns_an=['NC_0476%d.1'%i for i in range(58,80)]
accns_mm = ['NC_000067.7','NC_000068.8'] + ['NC_0000%d.7'%i for i in range(69,86)] + ['NC_000086.8','NC_000087.8']
accns_ms = ['CM0040%d.1'%i for i in range(94,100)] + ['CM00410%d.1'%i for i in range(4)] + ['CM00410%d.1'%i for i in range(5,10)] + ['CM00411%d.1'%i for i in range(4)] + ['CM004104.1']
accns_pl = ['CM0260%d.1'%i for i in range(41,65)]
accns_pmb = ['NC_056008.1','NC_056009.1']+['NC_0560%d.1'%i for i in range(10,31)]
accns_aa = ['NC_052047.1','NC_052048.2','NC_052049.1','NC_052050.1','NC_052051.1','NC_052052.2','NC_052053.1','NC_052054.1','NC_052055.2','NC_052056.1','NC_052057.2','NC_052058.2','NC_052059.1','NC_052060.1','NC_052061.1','NC_052063.2','NC_052064.1','NC_052065.1']
accns_oc = ['CM000%d.1'%i for i in range(790,812)]
accns_hh = ['CM0274%d.1'%i for i in range(13,46)]
accns_hs = ['NC_000001.11','NC_000002.12','NC_000003.12','NC_000004.12','NC_000005.10','NC_000006.12','NC_000007.14','NC_000008.11','NC_000009.12','NC_000010.11','NC_000011.10','NC_000012.12','NC_000013.11','NC_000014.9','NC_000015.10','NC_000016.10','NC_000017.11','NC_000018.10','NC_000019.10','NC_000020.11','NC_000021.9','	NC_000022.11','NC_000023.11','NC_000024.10']

# # create species objects and extract info - commented out because this is only done once
# # lab rat
# # mRatBN7.2
# # chroms 1-20, X, Y
# Rattus_norvegicus = Species(name='Rattus_norvegicus',accns=accns_rn)
# # black rat
# # Rrattus_CSIRO_v1
# # chroms 1-18, X, Y
# Rattus_rattus = Species(name='Rattus_rattus',accns=accns_rr)
# # African grass rat
# # mArvNil1.pat.X
# # chroms 1-21, X
# Arvicanthis_niloticus = Species(name="Arvicanthis_niloticus",accns=accns_an)
# # lab mouse
# # GRCm39
# # chroms 1-19, X, Y
# Mus_musculus = Species(name='Mus_musculus',accns=accns_mm)
# # Algerian mouse, or western Mediterranean mouse
# # SPRET_EiJ_v1
# # chroms 1-19, X
# Mus_spretus = Species(name='Mus_spretus',accns=accns_ms) # X is CM004104.1
# # white-footed mouse
# # UCI_PerLeu_2.1
# # weird chrom list
# Peromyscus_leucopus = Species(name='Peromyscus_leucopus',accns=accns_pl)
# # prairie deer mouse
# # HU_Pman_2.1.3
# # chroms 1-23, X
# Peromyscus_maniculatus_bairdii = Species(name='Peromyscus_maniculatus_bairdii',accns=accns_pmb)
# # European water vole
# # mArvAmp1.2
# # chroms 1-18, X
# Arvicola_amphibius = Species(name='Arvicola_amphibius',accns=accns_aa)
# # rabbit
# # OryCun2.0
# # chroms 1-21, X
# Oryctolagus_cuniculus = Species(name='Oryctolagus_cuniculus',accns=accns_oc)
# # capybara
# # Hydrochoerus_hydrochaeris_HiC
# # chroms 1-33
# Hydrochoeris_hydrochaeris = Species(name='Hydrochoeris_hydrochaeris',accns=accns_hh)
# # human
# # GRCh38.p13
# # chroms 1-22, X, Y
# Homo_sapiens_k7 = Species(name='Homo_sapiens',accns=accns_hs)
  
# # extract info
# Rattus_norvegicus.extract_seqrecs()
# Rattus_rattus.extract_seqrecs()
# Arvicanthis_niloticus.extract_seqrecs()
# Mus_musculus.extract_seqrecs()
# Mus_spretus.extract_seqrecs()
# Peromyscus_leucopus.extract_seqrecs()
# Arvicola_amphibius.extract_seqrecs() # no chrom 16
# Oryctolagus_cuniculus.extract_seqrecs()
# Hydrochoeris_hydrochaeris.extract_seqrecs()
# Homo_sapiens.extract_seqrecs() # NC_000023.11 is X and NC_000024.10 is Y
# # 11/23/21
# Peromyscus_maniculatus_bairdii.extract_seqrecs()

# create species objects for process_seqrecs
k = 7
Rattus_norvegicus_k7 = Species(name='Rattus_norvegicus',accns=accns_rn)
Rattus_rattus_k7 = Species(name='Rattus_rattus',accns=accns_rr)
Arvicanthis_niloticus_k7 = Species(name="Arvicanthis_niloticus",accns=accns_an)
Mus_musculus_k7 = Species(name='Mus_musculus',accns=accns_mm)
Mus_spretus_k7 = Species(name='Mus_spretus',accns=accns_ms) # X is CM004104.1
Peromyscus_leucopus_k7 = Species(name='Peromyscus_leucopus',accns=accns_pl)
Peromyscus_maniculatus_bairdii_k7 = Species(name='Peromyscus_maniculatus_bairdii',accns=accns_pmb)
Arvicola_amphibius_k7 = Species(name='Arvicola_amphibius',accns=accns_aa)
Oryctolagus_cuniculus_k7 = Species(name='Oryctolagus_cuniculus',accns=accns_oc)
Hydrochoeris_hydrochaeris_k7 = Species(name='Hydrochoeris_hydrochaeris',accns=accns_hh)
Homo_sapiens_k7 = Species(name='Homo_sapiens',accns=accns_hs)
# make list of species
sl_k7 = [Rattus_norvegicus_k7,Rattus_rattus_k7,Arvicanthis_niloticus_k7,Mus_musculus_k7,Mus_spretus_k7,Peromyscus_leucopus_k7,Peromyscus_maniculatus_bairdii_k7,Arvicola_amphibius_k7,Oryctolagus_cuniculus_k7,Hydrochoeris_hydrochaeris_k7,Homo_sapiens_k7]
for s in sl_k7:
    s.process_seqrecs()
    s.sort_seqrecs(attribute='chrom',reverse=False)
short_sl_k7 = sl_k7.copy()
short_sl_k7.remove(Rattus_norvegicus_k7)
short_sl_k7.remove(Arvicanthis_niloticus_k7)

# same for different k
k = 14
Rattus_norvegicus_k14 = Species(name='Rattus_norvegicus',accns=accns_rn)
Rattus_rattus_k14 = Species(name='Rattus_rattus',accns=accns_rr)
Arvicanthis_niloticus_k14 = Species(name="Arvicanthis_niloticus",accns=accns_an)
Mus_musculus_k14 = Species(name='Mus_musculus',accns=accns_mm)
Mus_spretus_k14 = Species(name='Mus_spretus',accns=accns_ms) # X is CM004104.1
Peromyscus_leucopus_k14 = Species(name='Peromyscus_leucopus',accns=accns_pl)
Peromyscus_maniculatus_bairdii_k14 = Species(name='Peromyscus_maniculatus_bairdii',accns=accns_pmb)
Arvicola_amphibius_k14 = Species(name='Arvicola_amphibius',accns=accns_aa)
Oryctolagus_cuniculus_k14 = Species(name='Oryctolagus_cuniculus',accns=accns_oc)
Hydrochoeris_hydrochaeris_k14 = Species(name='Hydrochoeris_hydrochaeris',accns=accns_hh)
Homo_sapiens_k14 = Species(name='Homo_sapiens',accns=accns_hs)
# list of species
sl_k14 = [Rattus_norvegicus_k14,Rattus_rattus_k14,Arvicanthis_niloticus_k14,Mus_musculus_k14,Mus_spretus_k14,Peromyscus_leucopus_k14,Peromyscus_maniculatus_bairdii_k14,Arvicola_amphibius_k14,Oryctolagus_cuniculus_k14,Hydrochoeris_hydrochaeris_k14,Homo_sapiens_k14]
for s in sl_k14:
    s.process_seqrecs()
    s.sort_seqrecs(attribute='chrom',reverse=False)
short_sl_k14 = sl_k14.copy()
short_sl_k14.remove(Rattus_norvegicus_k14)
short_sl_k14.remove(Arvicanthis_niloticus_k14)

# lists for graphing
str_list = ['R.norvegicus','R.rattus','A.niloticus','M.musculus','M.spretus','P.leucopus','P.bairdii','A.amphibius','O.cuniculus','H.hydrochaeris','H.sapiens']
short_str_list = ['R.r.','M.m.','M.s.','P.l.','P.m.b.','A.a.','O.c.','H.h.','H.s.']

# Figure 1 - Telomeric Proportion of Genome across all Species at both ks
fig, ax1 = plt.subplots()
X_axis = np.arange(len(str_list))
X_axis1 = np.arange(len(short_str_list))
left, bottom, width, height = [0.47, 0.36, 0.4, 0.4]
plt.xticks(ha='right',rotation=60, rotation_mode='anchor')
ax2 = fig.add_axes([left, bottom, width, height])
ax1.bar(X_axis,[s.spec_tel_prop * 100 for s in sl_k7],label='k=7')
ax1.bar(X_axis,[s.spec_tel_prop * 100 for s in sl_k14],label='k=14')
ax1.set_xticks(X_axis)
ax1.set_xticklabels(str_list)
ax2.bar(X_axis1,[s.spec_tel_prop * 100 for s in short_sl_k7],label='k=7')
ax2.bar(X_axis1,[s.spec_tel_prop * 100 for s in short_sl_k14],label='k=14')
ax2.set_xticks(X_axis1)
ax2.set_xticklabels(short_str_list)
ax2.set_title('without outliers')
ax2.legend()
ax1.set_title('Telomeric Proportion of Genome across all Species')
ax1.set_xlabel('species')
ax1.set_ylabel('telomeric percentage of genome (%)')
plt.xticks(ha='right',rotation=60, rotation_mode='anchor')
plt.savefig('figure1.jpeg',bbox_inches='tight')
plt.show()

# Figure 2 - Total Telomere Length across all Species at both ks
fig, ax1 = plt.subplots()
X_axis = np.arange(len(str_list))
X_axis1 = np.arange(len(short_str_list))
left, bottom, width, height = [0.47, 0.36, 0.4, 0.4]
plt.xticks(ha='right',rotation=60, rotation_mode='anchor')
ax2 = fig.add_axes([left, bottom, width, height])
ax1.bar(X_axis,[s.spec_total_tel_seg_length / 1000 for s in sl_k7],label='k=7')
ax1.bar(X_axis,[s.spec_total_tel_seg_length / 1000 for s in sl_k14],label='k=14')
ax1.set_xticks(X_axis)
ax1.set_xticklabels(str_list)
ax2.bar(X_axis1,[s.spec_total_tel_seg_length / 1000 for s in short_sl_k7],label='k=7')
ax2.bar(X_axis1,[s.spec_total_tel_seg_length / 1000 for s in short_sl_k14],label='k=14')
ax2.set_xticks(X_axis1)
ax2.set_xticklabels(short_str_list)
ax2.set_title('without outliers')
ax2.legend()
ax1.set_title('Total Telomere Length in Genome across all Species')
ax1.set_xlabel('species')
ax1.set_ylabel('telomeric length in genome (kb)')
plt.xticks(ha='right',rotation=60, rotation_mode='anchor')
plt.savefig('figure2.jpeg',bbox_inches='tight')
plt.show()

# Figure 3 - Telomeric Proportion of Chromosome for three Rat Species for both ks
f, (ax1,ax2) = plt.subplots(2,1,sharex=True)
X = [str(i) for i in range(1,22)] + ['X'] + ['Y']
yrn7 = [seq.seq_tel_prop * 100 for seq in Rattus_norvegicus_k7.seqs]
yrn7.insert(-3,0)
yrr7 = [seq.seq_tel_prop * 100 for seq in Rattus_rattus_k7.seqs]
for i in range(3):
    yrr7.insert(-3,0)
yan7 = [seq.seq_tel_prop * 100 for seq in Arvicanthis_niloticus_k7.seqs] + [0]
X_axis = np.arange(len(X))
ax1.bar(X_axis, yan7, label = 'A. niloticus',alpha=0.7)
ax1.bar(X_axis, yrn7, label = 'R. norvegicus',alpha=0.7,color='purple')
ax1.bar(X_axis, yrr7, label = 'R. rattus',color='green')
ax1.set_xticks(X_axis)
ax1.set_ylabel('telomeric %, k=7')
ax1.set_title('Telomeric Proportion of Chromosome for three Rat Species')
ax1.legend()
yrn14 = [seq.seq_tel_prop * 100 for seq in Rattus_norvegicus_k14.seqs]
yrn14.insert(-3,0)
yrr14 = [seq.seq_tel_prop * 100 for seq in Rattus_rattus_k14.seqs]
for i in range(3):
    yrr14.insert(-3,0)
yan14 = [seq.seq_tel_prop * 100 for seq in Arvicanthis_niloticus_k14.seqs] + [0]
ax2.bar(X_axis, yan14, label = 'A. niloticus',alpha=0.7)
ax2.bar(X_axis, yrn14, label = 'R. norvegicus',alpha=0.7,color='purple')
ax2.bar(X_axis, yrr14, label = 'R. rattus',color='green')
ax2.set_xticks(X_axis)
ax2.set_xticklabels(X)
ax2.set_ylabel('telomeric %, k=14')
ax2.set_xlabel('chromosome')
plt.savefig('figure3.jpeg',bbox_inches='tight')
plt.show()

# # Figure 3b - same plot as Figure 3 - Telomeric Proportion of Chromosome for three Rat Species for both ks - this presentation is too cluttered
# f, ax = plt.subplots()
# X = [str(i) for i in range(1,22)] + ['X'] + ['Y']
# yrn7 = [seq.seq_tel_prop * 100 for seq in Rattus_norvegicus_k7.seqs]
# yrn7.insert(-3,0)
# yrr7 = [seq.seq_tel_prop * 100 for seq in Rattus_rattus_k7.seqs]
# for i in range(3):
#     yrr7.insert(-3,0)
# yan7 = [seq.seq_tel_prop * 100 for seq in Arvicanthis_niloticus_k7.seqs] + [0]
# X_axis = np.arange(len(X))
# ax.bar(X_axis-0.2, yan7, 0.4, label = 'A. niloticus k=7',alpha=0.7)
# ax.bar(X_axis-0.2, yrn7, 0.4, label = 'R. norvegicus k=7',alpha=0.7)
# ax.bar(X_axis-0.2, yrr7, 0.4, label = 'R. rattus k=7')
# ax.set_title('Telomeric Proportion of Chromosome for three Rat Species')
# yrn14 = [seq.seq_tel_prop * 100 for seq in Rattus_norvegicus_k14.seqs]
# yrn14.insert(-3,0)
# yrr14 = [seq.seq_tel_prop * 100 for seq in Rattus_rattus_k14.seqs]
# for i in range(3):
#     yrr14.insert(-3,0)
# yan14 = [seq.seq_tel_prop * 100 for seq in Arvicanthis_niloticus_k14.seqs] + [0]
# ax.bar(X_axis+0.2, yan14, 0.4, label = 'A. niloticus k=14',alpha=0.7)
# ax.bar(X_axis+0.2, yrn14, 0.4, label = 'R. norvegicus k=14',alpha=0.7)
# ax.bar(X_axis+0.2, yrr14, 0.4, label = 'R. rattus k=14')
# ax.legend()
# ax.set_xticks(X_axis)
# ax.set_xticklabels(X)
# ax.set_ylabel('telomeric percentage')
# ax.set_xlabel('chromosome')
# plt.savefig('figure2b.jpeg',bbox_inches='tight')
# plt.show()

# Figure 4 - Telomere Length by Chromosome for three Rat Species for both ks
f, (ax1,ax2) = plt.subplots(2,1,sharex=True)
X = [str(i) for i in range(1,22)] + ['X'] + ['Y']
yrn7 = [seq.total_seq_tel_seg_length / 1000 for seq in Rattus_norvegicus_k7.seqs]
yrn7.insert(-3,0)
yrr7 = [seq.total_seq_tel_seg_length / 1000 for seq in Rattus_rattus_k7.seqs]
for i in range(3):
    yrr7.insert(-3,0)
yan7 = [seq.total_seq_tel_seg_length / 1000 for seq in Arvicanthis_niloticus_k7.seqs] + [0]
X_axis = np.arange(len(X))
ax1.bar(X_axis, yan7, label = 'A. niloticus',alpha=0.7)
ax1.bar(X_axis, yrn7, label = 'R. norvegicus',alpha=0.7,color='purple')
ax1.bar(X_axis, yrr7, label = 'R. rattus',color='green')
ax1.set_xticks(X_axis)
ax1.set_ylabel('kb, k=7')
ax1.set_title('Telomere Length by Chromosome for three Rat Species')
ax1.legend()
yrn14 = [seq.total_seq_tel_seg_length / 1000 for seq in Rattus_norvegicus_k14.seqs]
yrn14.insert(-3,0)
yrr14 = [seq.total_seq_tel_seg_length / 1000 for seq in Rattus_rattus_k14.seqs]
for i in range(3):
    yrr14.insert(-3,0)
yan14 = [seq.total_seq_tel_seg_length / 1000 for seq in Arvicanthis_niloticus_k14.seqs] + [0]
ax2.bar(X_axis, yan14, label = 'A. niloticus',alpha=0.7)
ax2.bar(X_axis, yrn14, label = 'R. norvegicus',alpha=0.7,color='purple')
ax2.bar(X_axis, yrr14, label = 'R. rattus',color='green')
ax2.set_xticks(X_axis)
ax2.set_xticklabels(X)
ax2.set_ylabel('kb, k=14')
ax2.set_xlabel('chromosome')
plt.savefig('figure4.jpeg',bbox_inches='tight')
plt.show()

# Figure 5 - Telomeric Proportion of Chromosome for three Mouse Species for both ks
f, (ax1,ax2) = plt.subplots(2,1,sharex=True)
X = [str(i) for i in range(1,24)] + ['X'] + ['Y']
ymm7 = [seq.seq_tel_prop * 100 for seq in Mus_musculus_k7.seqs]
for i in range(4):
    ymm7.insert(-3,0)
yms7 = [seq.seq_tel_prop * 100 for seq in Mus_spretus_k7.seqs] + [0]
for i in range(4):
    yms7.insert(-3,0)
ypmb7 = [seq.seq_tel_prop * 100 for seq in Peromyscus_maniculatus_bairdii_k7.seqs] + [0]
X_axis = np.arange(len(X))
ax1.bar(X_axis, yms7, label = 'M. spretus',alpha=0.7)
ax1.bar(X_axis, ymm7, label = 'M. musculus',alpha=0.7,color='purple')
ax1.bar(X_axis, ypmb7, label = 'P. bairdii',color='green')
ax1.set_xticks(X_axis)
ax1.set_ylabel('telomeric %, k=7')
ax1.set_title('Telomeric Proportion of Chromosome for three Mouse Species')
ax1.legend()
ymm14 = [seq.seq_tel_prop * 100 for seq in Mus_musculus_k14.seqs]
for i in range(4):
    ymm14.insert(-3,0)
yms14 = [seq.seq_tel_prop * 100 for seq in Mus_spretus_k14.seqs] + [0]
for i in range(4):
    yms14.insert(-3,0)
ypmb14 = [seq.seq_tel_prop * 100 for seq in Peromyscus_maniculatus_bairdii_k14.seqs] + [0]
ax2.bar(X_axis, yms14, label = 'M. spretus',alpha=0.7)
ax2.bar(X_axis, ymm14, label = 'M. musculus',alpha=0.7,color='purple')
ax2.bar(X_axis, ypmb14, label = 'P. bairdii',color='green')
ax2.set_xticks(X_axis)
ax2.set_xticklabels(X)
ax2.set_ylabel('telomeric %, k=14')
ax2.set_xlabel('chromosome')
plt.savefig('figure5.jpeg',bbox_inches='tight')
plt.show()

# Figure 6 - Telomere Length by Chromosome for three Mouse Species for both ks
f, (ax1,ax2) = plt.subplots(2,1,sharex=True)
X = [str(i) for i in range(1,24)] + ['X'] + ['Y']
ymm7 = [seq.total_seq_tel_seg_length / 1000 for seq in Mus_musculus_k7.seqs]
for i in range(4):
    ymm7.insert(-3,0)
yms7 = [seq.total_seq_tel_seg_length / 1000 for seq in Mus_spretus_k7.seqs] + [0]
for i in range(4):
    yms7.insert(-3,0)
ypmb7 = [seq.total_seq_tel_seg_length / 1000 for seq in Peromyscus_maniculatus_bairdii_k7.seqs] + [0]
X_axis = np.arange(len(X))
ax1.bar(X_axis, yms7, label = 'M. spretus',alpha=0.7)
ax1.bar(X_axis, ymm7, label = 'M. musculus',alpha=0.7,color='purple')
ax1.bar(X_axis, ypmb7, label = 'P. bairdii',color='green')
ax1.set_xticks(X_axis)
ax1.set_ylabel('kb, k=7')
ax1.set_title('Telomere Length by Chromosome for three Mouse Species')
ax1.legend()
ymm14 = [seq.total_seq_tel_seg_length / 1000 for seq in Mus_musculus_k14.seqs]
for i in range(4):
    ymm14.insert(-3,0)
yms14 = [seq.total_seq_tel_seg_length / 1000 for seq in Mus_spretus_k14.seqs] + [0]
for i in range(4):
    yms14.insert(-3,0)
ypmb14 = [seq.total_seq_tel_seg_length / 1000 for seq in Peromyscus_maniculatus_bairdii_k14.seqs] + [0]
ax2.bar(X_axis, yms14, label = 'M. spretus',alpha=0.7)
ax2.bar(X_axis, ymm14, label = 'M. musculus',alpha=0.7,color='purple')
ax2.bar(X_axis, ypmb14, label = 'P. bairdii',color='green')
ax2.set_xticks(X_axis)
ax2.set_xticklabels(X)
ax2.set_ylabel('kb, k=14')
ax2.set_xlabel('chromosome')
plt.savefig('figure6.jpeg',bbox_inches='tight')
plt.show()

# create comparison tables - telomeric proportion, length, and comparison ratios of genome for all species
# rows for k=7 tables
tl = ['R.n.','R.r.','A.n.','M.m.','M.s.','P.l.','P.m.b.','A.a.','O.c.','H.h.','H.s.']
# proportion for k = 7
columns = ['TPG (%)'] + tl
props = [s.spec_tel_prop * 100 for s in sl_k7]
ratios = [ [0] * 11 for i in range(11)]
for i in range(11):
    for j in range(11):
        ratios[i][j] = round(sl_k7[i].spec_tel_prop/sl_k7[j].spec_tel_prop,3)
for i in range(11):
    ratios[i].insert(0,props[i])
df2 = pd.DataFrame(ratios,index=tl,columns=columns)
df2.to_csv('table1_k7_proportions.csv')

# length for k = 7
columns = ['TTL (kb)'] + tl
lengths = [s.spec_total_tel_seg_length / 1000 for s in sl_k7]
ratios = [ [0] * 11 for i in range(11)]
for i in range(11):
    for j in range(11):
        ratios[i][j] = round(sl_k7[i].spec_total_tel_seg_length/sl_k7[j].spec_total_tel_seg_length,3)
for i in range(11):
    ratios[i].insert(0,lengths[i])
df1 = pd.DataFrame(ratios,index=tl,columns=columns)
df1.to_csv('table2_k7_lengths.csv')

# proportion for k = 14
# first remove species with zero telomere content at this threshold
stl = ['R.n.','R.r.','A.n.','M.m.','M.s.','P.m.b.','A.a.','O.c.','H.s.']
shorter_list = sl_k14.copy()
shorter_list.remove(Peromyscus_leucopus_k14)
shorter_list.remove(Hydrochoeris_hydrochaeris_k14)
columns = ['TPG (%)'] + stl
props = [s.spec_tel_prop * 100 for s in shorter_list]
ratios = [ [0] * 9 for i in range(9)]
for i in range(9):
    for j in range(9):
        ratios[i][j] = round(shorter_list[i].spec_tel_prop/shorter_list[j].spec_tel_prop,3)
for i in range(9):
    ratios[i].insert(0,props[i])
df1 = pd.DataFrame(ratios,index=stl,columns=columns)
df1.to_csv('table3_k14_proportions.csv')

# length for k = 14
columns = ['TTL (kb)'] + stl
lengths = [s.spec_total_tel_seg_length / 1000 for s in shorter_list]
ratios = [ [0] * 9 for i in range(9)]
for i in range(9):
    for j in range(9):
        ratios[i][j] = round(shorter_list[i].spec_total_tel_seg_length/shorter_list[j].spec_total_tel_seg_length,3)
for i in range(9):
    ratios[i].insert(0,lengths[i])
df1 = pd.DataFrame(ratios,index=stl,columns=columns)
df1.to_csv('table4_k14_lengths.csv')
