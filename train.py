from esm.pretrained import load_model_and_alphabet
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from csv import reader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
import numpy as np

from torch.utils.data import DataLoader
import pandas as pd
import logging
logging.basicConfig(level=logging.DEBUG)


class FrameClassifier(nn.Module):
    def __init__(self, FB_ESM, dropout = 0.1):
        super(FrameClassifier, self).__init__()

        self.embedding = FB_ESM
        self.classifier = nn.Linear(self.embedding.args.embed_dim,1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, data):


        x = self.embedding(x, repr_layers=[33], return_contacts=True)["representations"][33] #should give error, change to 33
        logging.debug("X shape: " + str(x.shape))

        sequence_representations = []
        for i, (_, seq) in enumerate(data):
            sequence_representations.append(x[i, 1 : len(seq) + 1].mean(0))

        x = th.stack(sequence_representations)
       
        x = self.classifier(x)
        x = self.dropout(x)
        x = th.sigmoid(x)
        logging.debug(f"X shape: {x.shape}")
        logging.debug(f"max and min:  {th.max(x)}, {th.min(x)}")
        return x

## Load FB-ESM model
model, alphabet = load_model_and_alphabet("data/esm1b_t33_650M_UR50S.pt")
batch_converter = alphabet.get_batch_converter()

pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.info("FB_ESM params: " + str(pytorch_total_params) + ", " + str(pytorch_total_params_train))


def compute_roc(labels, preds):
    # Compute ROC curve and ROC area for each class
    fpr, tpr, _ = roc_curve(labels.flatten(), preds.flatten())
    roc_auc = auc(fpr, tpr)

    return roc_auc

# Prepare the data
def load_data(path):

    with open(path, 'r') as read_obj:
        # pass the file object to reader() to get the reader object
        csv_reader = reader(read_obj)
        # Get all rows of csv from csv_reader object as list of tuples
        data = list(map(tuple, csv_reader))[:200000]

    logging.info("Data size: " + str(len(data)))

    return train_test_split(data, test_size=0.2, shuffle = True)

data_train, data_test = load_data('data/only_classifier.csv')

# Create the classifier
device = "cpu" # "cuda:0"

model.to(device)

logging.debug("Model device: " + str(next(model.parameters()).device))

classifier = FrameClassifier(model)

classifier.to(device)
logging.debug("Classifier device: " + str(next(classifier.parameters()).device))


pytorch_total_params = sum(p.numel() for p in classifier.parameters())
pytorch_total_params_train = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

logging.info("Classifier params: " + str(pytorch_total_params) + ", " + str(pytorch_total_params_train))





# Train 
lr = 1e-4
l2norm = 0
n_epochs = 20
batch_size = 64
loss_func = nn.BCELoss()
optimizer = th.optim.Adam(classifier.parameters(), lr=lr, weight_decay=l2norm)


data_train, data_val = train_test_split(data_train, test_size=0.2, shuffle = True)

train_dataloader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(data_val, batch_size=batch_size, shuffle=True)

best_loss = float('inf')

classifier.train()
for epoch in range(n_epochs):

    train_loss = 0
    train_acc = 0
    train_total = 0
    epoch_steps = 0
    for batch in train_dataloader:
        epoch_steps += 1

        batch = list(zip(*batch))
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)

        batch_tokens = batch_tokens.to(device)

        batch_labels = list(map(int, batch_labels))
        batch_labels = th.FloatTensor(batch_labels).unsqueeze(1).to(device)
        logging.debug(f"labels preds: {batch_labels.shape}" )

        optimizer.zero_grad()
        
        preds = classifier(batch_tokens, batch)

        logging.debug(f"size preds: {preds.squeeze().shape}" )
       
        loss = loss_func(preds, batch_labels) #F.cross_entropy(preds, batch_labels)
        
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
#        acc = th.sum(preds.argmax(dim=1) == batch_labels)
#        train_acc += acc.item()

        train_total += batch_labels.size(0)


    
    val_acc = 0
    val_loss = 0
    val_total = 0
    val_steps = 0

    val_labels = []
    val_preds = []
    for batch in val_dataloader:
        with th.no_grad():
            val_steps += 1

            batch = list(zip(*batch))
            batch_labels, batch_strs, batch_tokens = batch_converter(batch)

            batch_tokens = batch_tokens.to(device)

            batch_labels = list(map(int, batch_labels))
            batch_labels = th.FloatTensor(batch_labels).unsqueeze(1).to(device)
            
            preds = classifier(batch_tokens, batch)

            loss = loss_func(preds, batch_labels) #F.cross_entropy(preds, batch_labels)

            val_loss += loss.item()
          #  acc = th.sum(preds.argmax(dim=1) == batch_labels)
          #  val_acc += acc.item()

            val_total += batch_labels.size(0)
            
            val_labels = np.append(val_labels, batch_labels.cpu())
            val_preds = np.append(val_preds, preds.cpu())

    if best_loss > val_loss/val_steps:
        best_loss = val_loss/val_steps
        th.save(classifier, 'model.pt')
#        th.save(classifier.state_dict(), 'model.pth')

    # print("Epoch {:05d} | ".format(epoch) +
    #     "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
    #         train_acc/ train_total, train_loss/epoch_steps) +
    #         "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
    #             val_acc/val_total, val_loss/val_steps))

    roc_auc = compute_roc(val_labels, val_preds)
    print("Epoch {:05d} | ".format(epoch) +
        "Train Loss: {:.4f} | ".format(
            train_loss/epoch_steps) +
            "Validation loss: {:.4f} | ".format(
                val_acc/val_total, val_loss/val_steps) + "AUC: {:.4f}".format(roc_auc))

#classifier.load_state_dict(th.load('model.pth'))
classifier = th.load('model.pt')
th.set_grad_enabled(False)
classifier.eval()

test_dataloader = DataLoader(data_test, batch_size=batch_size, shuffle=True)

test_loss = 0
test_acc = 0
test_total = 0
test_steps = 0
all_labels = []
all_preds = []
with th.no_grad():
    for batch in test_dataloader:
        test_steps += 1

        batch = list(zip(*batch))
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)

        batch_tokens = batch_tokens.to(device)

        batch_labels = list(map(int, batch_labels))
        batch_labels = th.FloatTensor(batch_labels).unsqueeze(1).to(device)

        preds = classifier(batch_tokens, batch)
        loss = loss_func(preds, batch_labels)
      
        all_labels = np.append(all_labels, batch_labels.cpu())
        all_preds = np.append(all_preds, preds.cpu())
      
        test_loss += loss.item()
        #acc = th.sum(preds.argmax(dim=1) == batch_labels)
        #test_acc += acc.item()

        test_total += batch_labels.size(0)


    roc_auc = compute_roc(all_labels, all_preds)

   

    print("Test Loss: {:.4f} | AUC: {:.4f}".format(
         test_loss/test_steps, roc_auc))


