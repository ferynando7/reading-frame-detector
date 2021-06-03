from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np


path = "data/embeddings.csv"
data = np.genfromtxt(path, delimiter=',')

labels = data[:,0]
sequences = data[:, 1:]


input_train, input_val, labels_train, labels_val = train_test_split(sequences, labels, test_size=0.2)


clf = svm.SVC()

clf.fit(input_train, labels_train)


train_preds = clf.predict(input_train)
val_preds = clf.predict(input_val)


train_acc = accuracy_score(train_preds, labels_train)
val_acc = accuracy_score(val_preds, labels_val)

print("Train accuracy is: " + str(train_acc))
print("Test accuracy is: " + str(val_acc))