import torch
import esm
import logging
import numpy as np
from csv import reader
from torch.utils.data import DataLoader

#logging.basicConfig(level=logging.INFO)

# Load 34 layer model
model, alphabet = esm.pretrained.esm1_t34_670M_UR50S()
batch_converter = alphabet.get_batch_converter()

# Prepare data (two protein sequences)
# open file in read mode
with open('data/only_classifier.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Get all rows of csv from csv_reader object as list of tuples
    data = list(map(tuple, csv_reader))[:20000]

logging.info(data[:2])


batch_labels, batch_strs, batch_tokens = batch_converter(data)

logging.info(batch_tokens.shape)


# Extract per-residue embeddings (on CPU)
batch_tokens = DataLoader(batch_tokens, batch_size=64, shuffle=False)


with torch.no_grad():
    token_embeddings = []

    for batch in batch_tokens:
        results = model(batch, repr_layers=[34])
        token_embeddings += results["representations"][34]

token_embeddings = np.array([embedding.detach().cpu().numpy() for embedding in token_embeddings])

# Generate per-sequence embeddings via averaging
# NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
sequence_embeddings = []
for i, (_, seq) in enumerate(data):
    sequence_embeddings.append(token_embeddings[i, 1:len(seq) + 1].mean(0))

logging.info(sequence_embeddings)

data = np.insert(sequence_embeddings, 0, batch_labels, axis=1)

np.savetxt("data/embeddings.txt", data, delimiter=",")
