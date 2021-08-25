import click as ck
import gffutils
from Bio import SeqIO
from utils import checkFeatures, overlaps, getCodingRegion

@ck.command()
@ck.option(
    '--predictor', '-p',
    help='name of the predicton (FGS or RFD)')
@ck.option(
    '--length-read', '-l',
    help='length of the simulated reads')
@ck.option(
    '--err', '-e',
    help='percentage of error in the reads')
@ck.option(
    '--species', '-s',
    help='identifier of the species')



def eval(predictor, length_read, err, species):
    root = 'data/reads_and_preds/'
    
    sequencer = '454'
    if int(length_read) == 700:
        sequencer = 'Sanger'

    input_file = root + length_read + '_' + err + '/' + species + '-' + sequencer + '.fna'
    if predictor == 'RFD':
        predictions_file = root + length_read + '_' + err + '/' + species + '-' + sequencer + '.' + predictor + '.0.7.gff'
    else:
        predictions_file = root + length_read + '_' + err + '/' + species + '-' + sequencer + '.' + predictor + '.gff'
    reference_file = 'data/refs/' + species + '.gff'

    reference_db = gffutils.create_db(reference_file, ':memory:', id_spec={'gene': 'db_xref'}, force = True)
    
    predictions_db = gffutils.create_db(predictions_file, ':memory:', id_spec=":seqid:", force = True, merge_strategy = 'replace')

    input_sequences = SeqIO.parse(open(input_file),'fasta')

    total = 0
    annotated = 0
    predicted = 0
    true_positive = 0
    for seq in input_sequences:
        total += 1
        read_start, read_end  = tuple(seq.description.split('|')[1].split(',')[2].split(';')[0].split('-'))
        read_start, read_end = (int(read_start), int(read_end[:-1]))

        features = reference_db.region(start=read_start, end=read_end)

        regions = getCodingRegion(features, read_start, read_end)

        if len(regions) > 0:
            annotated += 1
        
        pred = None
        try:
            pred = predictions_db[seq.id]
        except:
            continue

        if not pred is None:
            predicted += 1
            for start_gt, end_gt, strand_gt in regions:
                pred_overlaps = overlaps(start_gt, end_gt, read_start+pred.start-1, read_start+pred.end-1, prcntg = 0.8)
                if pred_overlaps and strand_gt == pred.strand:
                    true_positive += 1
                    break
                #elif strand_gt != pred.strand:
                    #print(start_gt, end_gt, strand_gt, seq.id, pred.strand)

        
    # print(f"Predictions file was {predictions_file}")
    # print(f'Total: {total}\nAnnotated: {annotated}\nPredicted: {predicted}\nTrue positive: {true_positive}')

    sn = true_positive/annotated
    sp = true_positive/predicted
    acc = (sn+sp)/2
    # print(f'Sensitivity: {sn}')
    # print(f'Specificity: {sp}')
    # print(f'Accuracy: {acc}')
    print(f"{predictor}\t{species}\t{length_read}\t{err}\t{sn}\t{sp}\t{acc}")
    
if __name__ == '__main__':
    eval()

    