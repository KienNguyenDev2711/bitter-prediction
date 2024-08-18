import numpy as np
import pandas as pd
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.regularizers import l2
from keras.regularizers import L1L2
from keras.layers import Activation, Dense, BatchNormalization, Dropout, LSTM
from keras.utils import to_categorical
from keras import optimizers
from keras import regularizers
from keras.optimizers import RMSprop
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)
from rdkit.Chem import rdMolDescriptors
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold
from keras.utils import pad_sequences
from sklearn.metrics import roc_auc_score


# Custom precision function
def precision(y_true, y_pred):
    y_true = K.cast(y_true, "float32")  # Ensure y_true is float32
    y_pred = K.cast(y_pred, "float32")  # Ensure y_pred is float32
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


# Custom recall function
def recall(y_true, y_pred):
    y_true = K.cast(y_true, "float32")  # Ensure y_true is float32
    y_pred = K.cast(y_pred, "float32")  # Ensure y_pred is float32
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


# Custom F1 score function
def f1(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * (
        (precision_val * recall_val) / (precision_val + recall_val + K.epsilon())
    )


seed = 200422

data_train = pd.read_csv("dataset/bitter-or-not/true_train.csv")
data_test = pd.read_csv("dataset/bitter-or-not/true_test.csv")
data_valid = pd.read_csv("dataset/test_set/UNIMI.csv")

data_train = data_train.sample(frac=1)
data_test = data_test.sample(frac=1)
data_valid = data_valid.sample(frac=1)

data_data = pd.concat([data_train, data_test], axis=0)


def haha(x):
    if x == True:
        return 1
    else:
        return 0


data_train["Bitter"] = data_train["Bitter"].apply(lambda x: haha(x))
data_test["Bitter"] = data_test["Bitter"].apply(lambda x: haha(x))
data_valid["Bitter"] = data_valid["Bitter"].apply(lambda x: haha(x))
all_data = pd.concat([data_train, data_test, data_valid], axis=0)

aa = [Chem.MolFromSmiles(x) for x in all_data["smiles"]]
aa_sentence = [mol2alt_sentence(x, 1) for x in aa]
vocab = np.unique([x for l in aa_sentence for x in l])
num_words = len(vocab)
word_map = {}
for i in range(len(vocab)):
    word_map[vocab[i]] = i + 1
embedding_length = 32

aa_train = [Chem.MolFromSmiles(x) for x in data_train["smiles"]]
aa_sentence_train = [mol2alt_sentence(x, 1) for x in aa_train]
aa_map_train = []
aa_map_all = []
for m in aa_sentence_train:
    aa_map_train.append([word_map[x] for x in m])
    aa_map_all.append([word_map[x] for x in m])
aa_map_train_padded = pad_sequences(aa_map_train, padding="post")
aa_map_train = np.array(aa_map_train_padded)

aa_test = [Chem.MolFromSmiles(x) for x in data_test["smiles"]]
aa_sentence_test = [mol2alt_sentence(x, 1) for x in aa_test]
aa_map_test = []
for m in aa_sentence_test:
    aa_map_test.append([word_map[x] for x in m])
    aa_map_all.append([word_map[x] for x in m])

aa_map_test_padded = pad_sequences(aa_map_test, padding="post")
aa_map_test = np.array(aa_map_test_padded)

aa_valid = [Chem.MolFromSmiles(x) for x in data_valid["smiles"]]
aa_sentence_valid = [mol2alt_sentence(x, 1) for x in aa_valid]
aa_map_valid = []
for m in aa_sentence_valid:
    aa_map_valid.append([word_map[x] for x in m])
    aa_map_all.append([word_map[x] for x in m])
aa_map_valid_padded = pad_sequences(aa_map_valid, padding="post")
aa_map_valid = np.array(aa_map_valid_padded)

maxSeqLen = max([len(aa_map_train[i]) for i in range(len(aa_map_train))])


def main():
    crnn_score = []
    crnn = Sequential()
    crnn.add(Embedding(num_words + 1, embedding_length, input_length=maxSeqLen))
    crnn.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
    crnn.add(MaxPooling1D(pool_size=3))
    crnn.add(LSTM(100))
    crnn.add(Dense(1, activation="sigmoid"))
    crnn.compile(loss="binary_crossentropy", optimizer="adam", metrics=[f1])
    print(crnn.summary())

    y_data = data_train["Bitter"]
    x_data = aa_map_train
    y_data = y_data.values.reshape(-1, 1)
    x_test = aa_map_test
    x_valid = aa_map_valid
    x_data = pad_sequences(x_data, maxlen=maxSeqLen)
    x_valid = pad_sequences(x_valid, maxlen=maxSeqLen)
    x_test = pad_sequences(x_test, maxlen=maxSeqLen)
    history = crnn.fit(
        x_data,
        y_data,
        validation_data=(x_valid, data_valid["Bitter"].values.reshape(-1, 1)),
        epochs=30,
        batch_size=64,
    )
    data_history = pd.DataFrame(history.history)

    true_pred = crnn.predict(x_test)
    y_true_test = data_test["Bitter"].values.reshape(-1, 1)
    print(accuracy_score(true_pred.round(), y_true_test))
    print(f1_score(true_pred.round(), y_true_test))
    print(recall_score(true_pred.round(), y_true_test))  # SN (Sensitivity)

    phyto_tn, phyto_fp, phyto_fn, phyto_tp = confusion_matrix(
        true_pred.round(), y_true_test
    ).ravel()
    phyto_specificity = phyto_tn / (phyto_tn + phyto_fp)
    print(phyto_specificity),  # (SP) specificity
    print(average_precision_score(y_true_test, true_pred))  # AUPR


main()
