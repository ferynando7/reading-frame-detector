import torch.nn as nn
import torch as th
import torch.nn.functional as F


def reverseComplement(string):
    complements = {'A':'T', 'G':'C', 'C':'G', 'T':'A'}
    revComplem = ""
    for char in string:
        revComplem = complements[char] + revComplem
    return revComplem

geneticCode = {
    "AAA": "K",
    "AAC": "N",
    "AAG": "K",
    "AAT": "N",
    "ACA": "T",
    "ACC": "T",
    "ACG": "T",
    "ACT": "T",
    "AGA": "R",
    "AGC": "S",
    "AGG": "R",
    "AGT": "S",
    "ATA": "I",
    "ATC": "I",
    "ATG": "M",
    "ATT": "I",

    "CAA": "Q",
    "CAC": "H",
    "CAG": "Q",
    "CAT": "H",
    "CCA": "P",
    "CCC": "P",
    "CCG": "P",
    "CCT": "P",
    "CGA": "R",
    "CGC": "R",
    "CGG": "R",
    "CGT": "R",
    "CTA": "L",
    "CTC": "L",
    "CTG": "L",
    "CTT": "L",

    "GAA": "E",
    "GAC": "D",
    "GAG": "E",
    "GAT": "D",
    "GCA": "A",
    "GCC": "A",
    "GCG": "A",
    "GCT": "A",
    "GGA": "G",
    "GGC": "G",
    "GGG": "G",
    "GGT": "G",
    "GTA": "V",
    "GTC": "V",
    "GTG": "V",
    "GTT": "V",

    "TAA": "*",
    "TAC": "Y",
    "TAG": "*",
    "TAT": "Y",
    "TCA": "S",
    "TCC": "S",
    "TCG": "S",
    "TCT": "S",
    "TGA": "*",
    "TGC": "C",
    "TGG": "W",
    "TGT": "C",
    "TTA": "L",
    "TTC": "F",
    "TTG": "L",
    "TTT": "F",

}

    
def translate(rna, geneticCode):

    # Input: An RNA string Pattern and the array GeneticCode.
    # Output: The translation of Pattern into an amino acid string Peptide.
    protein = ""
    for i in range(0, len(rna)-2, 3):
        codon = rna[i:i+3]
        aminoacid = geneticCode[codon]
        protein += aminoacid
    return protein

def safeTranslate(sequence, idx):

    end = len(sequence)
    protein = ""
    for i in range(0, len(sequence)-2, 3):
        codon = sequence[i:i+3]
        aminoacid = geneticCode[codon]
        protein += aminoacid
        if aminoacid == '*':
            end = i + 3
            break
    end += idx
    return end, protein


def findStartPos(sequence, idx):

    start = 0
    for i in range(idx, len(sequence)-2, 3):
        codon = sequence[i:i+3]
        if codon in ['ATG', 'GTG', 'TTG']:
            start = i 
            break
    start += 1
    return start



def translateFull(sequence):
    translations = []

    strand = '+'
    for i in range(3):
        frame = i
        start_pos = 1
        end_pos, seq = safeTranslate(sequence[i:], i)
        if end_pos == len(sequence):
            start_pos = findStartPos(seq[i:], i)
        
        if end_pos - start_pos >= 60:
            translations.append((start_pos, end_pos, frame, strand, seq))
       
    strand = '-'
    sequence = reverseComplement(sequence)
    for i in range(3):
        frame = i
        start_pos = 1
        end_pos, seq = safeTranslate(sequence[i:], i)
        if end_pos == len(sequence):
            start_pos = findStartPos(seq[i:], i)
        
        aux = start_pos
        start_pos = len(sequence) - end_pos + 1
        end_pos = len(sequence) - aux +1
        if end_pos - start_pos >= 60:
            translations.append((start_pos, end_pos, frame, strand, seq))
    return translations

class FrameClassifier(nn.Module):
    def __init__(self, FB_ESM, dropout = 0.1):
        super(FrameClassifier, self).__init__()

        self.embedding = FB_ESM
        self.classifier = nn.Linear(self.embedding.args.embed_dim,1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, data):


        x = self.embedding(x, repr_layers=[33], return_contacts=True)["representations"][33] #should give error, change to 33
       
        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(x[i, 1 : len(seq) + 1].mean(0))

        x = th.stack(sequence_representations)
       
        x = self.classifier(x)
        x = self.dropout(x)
        x = th.sigmoid(x)
        return x



def checkFeatures(features, read_start, read_end, read_strand=None):
    result = 0
    for feat in features:
        if read_strand is None:
            read_strand = feat.strand 
        if 'gene' == feat.featuretype: # and overlaps(feat.start, feat.end, read_start, read_end) and feat.strand == read_strand:
            result = 1
            break

    return result

def getCodingRegion(features, read_start, read_end):
    result = 0
    regions = []
    for feat in features:
        if 'gene' == feat.featuretype:
            set_base = set(range(feat.start, feat.end+1))
            set_read = set(range(read_start, read_end+1))
            overlap_region = set_base & set_read
         
            start = min(overlap_region)
            end = max(overlap_region)
            
            if end - start >= 75:
                regions.append((start, end, feat.strand))
    
    return regions


def overlaps(base_start, base_end, read_start, read_end, prcntg = 0.8):
    
    overlap = False

    set_base = set(range(base_start, base_end))
    set_read = set(range(read_start, read_end))

    overlap_region = set_base & set_read

    if len(overlap_region)/len(set_base) >= prcntg:
        overlap = True
    
    return overlap