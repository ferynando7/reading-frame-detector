from Bio import SeqIO



def read_fasta_single(input_file):
    fasta_sequences = SeqIO.parse(open(input_file),'fasta')
    for fasta in fasta_sequences:
        print(type(fasta))        
        name, sequence = fasta.id, str(fasta.seq)
        
    return sequence


