from esm.pretrained import load_model_and_alphabet
import torch as th
import torch.nn as nn
import click as ck
import torch.nn.functional as F
import numpy as np
import read_fasta as rf


import time   
import gffutils
from Bio import SeqIO
from utils import FrameClassifier, translate, reverseComplement, geneticCode, checkFeatures, overlaps, translateFull

import logging
logging.basicConfig(level=logging.INFO)


@ck.command()
@ck.option(
    '--length-read', '-l',
    help='length of the simulated reads')
@ck.option(
    '--err', '-e',
    help='percentage of error in the reads')
@ck.option(
    '--species', '-s',
    help='identifier of the species')

def main(length_read, err, species):

    root = 'data/reads_and_preds/'

    sequencer = '454'
    if int(length_read) == 700:
        sequencer = 'Sanger'

    sequences_file = root + length_read + '_' + err + '/' + species + '-' + sequencer + '.fna'
    out_file = root + length_read + '_' + err + '/' + species + '-' + sequencer + '.RFD.0.7.gff'

    startT = time.time()
    model, alphabet = load_model_and_alphabet("data/esm1b_t33_650M_UR50S.pt")
    end = time.time()
    logging.info(f"Loaded FB model and alphabet ({(end-startT):.{4}})")

    startT = time.time()
    batch_converter = alphabet.get_batch_converter()
    end = time.time()
    logging.info(f"Got batch converter ({(end-startT):.{4}})")

    device = "cuda:0"

    startT = time.time()
    classifier = th.load('model.pt')
    classifier.to(device)
    classifier.eval()
    end = time.time()
    logging.info(f"Loaded model ({(end-startT):.{4}})")
    
    startT = time.time()
    fasta_sequences = SeqIO.parse(open(sequences_file),'fasta')
    end = time.time()
    logging.info(f"Loaded FASTA file with sequences ({(end-startT):.{4}})")
    
    results = []
    translateTime = 0
    predictTime = 0
    for seq in fasta_sequences:        
        start, end, frame, strand, length, score, timeTranslate, timePredict = predict(str(seq.seq), batch_converter, device, classifier)
        #print(seq.id)
        # predicts = predict(str(seq.seq), batch_converter, device, classifier)
        # logging.debug(results)
        # for start, end, frame, strand, length, score in predicts:  
        predictTime += timePredict
        translateTime += timeTranslate          
        if not start is None:
            line = [seq.id, 'RFD', 'GENE', str(start), str(end), '.', strand, str(frame), str(length), str(score)]
            results.append('\t'.join(line))
        
    logging.info(f"Prediction time ({(predictTime):.{4}})")
    logging.info(f"Translation time ({(translateTime):.{4}})")

    o_file = open(out_file, 'w')
    o_file.write('\n'.join(results))

def predict(sequence, batch_converter, device, classifier):
    
    start, end, frame, strand, length, score, timeTranslate, timePredict = checkSequence(sequence, batch_converter, device, classifier)
    #results = checkSequence(sequence, batch_converter, device, classifier)

    # strands = set(map(lambda x: x[3], results))
    # if len(strands) > 1:
    #     results = []
    return start, end, frame, strand, length, score, timeTranslate, timePredict
    
def checkSequence(sequence, batch_converter, device, classifier, reverse=False):
    start = None
    end = None
    frame = None
    strand = None
    length = -1
    results = []

    # potential_starts = findStartPos(sequence, window_size)
    startT = time.time()
    potentialSequences = translateFull(sequence)
    end = time.time()
    timeTranslate = end - startT

    score = 0
    
    startT = time.time()
    
   
    with th.no_grad():
        if len(potentialSequences) > 0:
            sequences = [('1', s) for _, _, _, _, s in potentialSequences]
            _, _, batch_tokens = batch_converter(sequences)
            batch_tokens = batch_tokens.to(device)
            
            preds = classifier(batch_tokens, sequences).cpu().numpy()

            cand_idx = np.argmax(preds)
            #print((cand_start, cand_end, cand_frame, cand_strand, cand_end-cand_start+1, cand_score))
            
            if  preds[cand_idx] >= 0.7:
                cand_start, cand_end, cand_frame, cand_strand, cand_seq = potentialSequences[cand_idx]
                score = preds[cand_idx]
                start = cand_start
                end = cand_end       
                frame = cand_frame
                strand = cand_strand
                length = end-start+1

    end = time.time()
    timePredict = end - startT

    return start, end, frame, strand, length, score, timeTranslate, timePredict

                                             


if __name__ == '__main__':
    main()















