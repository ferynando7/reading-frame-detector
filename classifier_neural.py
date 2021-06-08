from esm.pretrained import load_model_and_alphabet
import torch as th
import torch.nn as nn
from csv import reader
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import logging
logging.basicConfig(level=logging.INFO)


class FrameClassifier(nn.Module):
    def __init__(self, FB_ESM):
        super(FrameClassifier, self).__init__()

        self.embedding = FB_ESM
        self.classifier = nn.Linear(128,2)


    def forward(self, x):
        x = self.embedding(x)["representations"][34] #should give error, change to 33
        x = self.classifier(F.softmax(x))

        return x

## Load FB-ESM model
model, alphabet = load_model_and_alphabet("data/esm1b_t33_650M_UR50S.pt")
batch_converter = alphabet.get_batch_converter()

pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_params_train = sum(p.numel() for p in model.parameters() if p.requires_grad)

logging.info("FB_ESM params: " + str(pytorch_total_params) + ", " + str(pytorch_total_params_train))




# Create the classifier
classifier = FrameClassifier(model)

pytorch_total_params = sum(p.numel() for p in classifier.parameters())
pytorch_total_params_train = sum(p.numel() for p in classifier.parameters() if p.requires_grad)

logging.info("Classifier params: " + str(pytorch_total_params) + ", " + str(pytorch_total_params_train))


# Prepare the data

with open('data/only_classifier.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Get all rows of csv from csv_reader object as list of tuples
    data = list(map(tuple, csv_reader))[:20000]

data_train, data_val = train_test_split(data, test_size=0.2, shuffle = True)


# Train 
lr = 0.01
l2norm = 0
n_epochs = 20

optimizer = th.optim.Adam(classifier.parameters(), lr=lr, weight_decay=l2norm)

batch_train = DataLoader(data_train, batch_size=64, shuffle=False)

for epoch in range(n_epochs):

    for batch in batch_train:
        optimizer.zero_grad()
        batch_labels, batch_strs, batch_tokens = batch_converter(batch)

        preds = classifier(batch_tokens)

        loss = F.cross_entropy(preds, batch_labels)
        loss.backward()

        optimizer.step()

        train_acc = torch.sum(preds.argmax(dim=1) == batch_labels)
        train_acc = train_acc.item() / batch_size
        print("Epoch {:05d} | ".format(epoch) +
            "Train Accuracy: {:.4f} | Train Loss: {:.4f} | ".format(
                train_acc, loss.item()) +
            "Validation Accuracy: {:.4f} | Validation loss: {:.4f}".format(
                val_acc, val_loss.item()))