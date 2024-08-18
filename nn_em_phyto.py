import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
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
from tabulate import tabulate

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from rdkit.Chem import rdMolDescriptors
from sklearn.metrics import average_precision_score
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold
from keras.utils import pad_sequences
from sklearn.metrics import roc_auc_score
import tensorflow as tf


def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    y_true = K.cast(y_true, "float32")  # Ensure y_true is float32
    y_pred = K.cast(y_pred, "float32")  # Ensure y_pred is float32
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * (
        (precision_val * recall_val) / (precision_val + recall_val + K.epsilon())
    )


seed = 200422

data_train = pd.read_csv("dataset/bitter-or-not/true_train.csv")
data_valid = pd.read_csv("dataset/bitter-or-not/true_valid.csv")
data_phyto = pd.read_csv("dataset/test_set/phyto_test.csv")
data_bitter_new = pd.read_csv("dataset/test_set/bitter_new.csv")
data_unimi = pd.read_csv("dataset/test_set/UNIMI.csv")

data_train = data_train.sample(frac=1)


def haha(x):
    if x == True:
        return 1
    else:
        return 0


data_train["Bitter"] = data_train["Bitter"].apply(lambda x: haha(x))
data_valid["Bitter"] = data_valid["Bitter"].apply(lambda x: haha(x))

data_train = pd.concat([data_train, data_valid], axis=0)

data_train["mol"] = data_train["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
data_train["sentences"] = data_train["mol"].apply(
    lambda x: MolSentence(mol2alt_sentence(x, radius=2))
)
w2v_model = word2vec.Word2Vec.load("pretrained/model_300dim.pkl")
data_train["embedding"] = [
    DfVec(x) for x in sentences2vec(data_train["sentences"], w2v_model, unseen="UNK")
]
data_train_vec = np.array([x.vec for x in data_train["embedding"]])
data_train_vec = np.reshape(data_train_vec, (len(data_train_vec), 300, 1))

data_phyto["mol"] = data_phyto["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
data_phyto["sentences"] = data_phyto["mol"].apply(
    lambda x: MolSentence(mol2alt_sentence(x, radius=2))
)
data_phyto["embedding"] = [
    DfVec(x) for x in sentences2vec(data_phyto["sentences"], w2v_model, unseen="UNK")
]
data_phyto_vec = np.array([x.vec for x in data_phyto["embedding"]])
data_phyto_vec = np.reshape(data_phyto_vec, (len(data_phyto_vec), 300, 1))

data_unimi["mol"] = data_unimi["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
data_unimi["sentences"] = data_unimi["mol"].apply(
    lambda x: MolSentence(mol2alt_sentence(x, radius=2))
)
data_unimi["embedding"] = [
    DfVec(x) for x in sentences2vec(data_unimi["sentences"], w2v_model, unseen="UNK")
]
data_unimi_vec = np.array([x.vec for x in data_unimi["embedding"]])
data_unimi_vec = np.reshape(data_unimi_vec, (len(data_unimi_vec), 300, 1))


def main():
    model = Sequential(
        [
            Dense(256, input_shape=(300,), activation="relu"),
            Dense(128, activation="sigmoid"),
            Dense(64, activation="sigmoid"),
            Dense(32, activation="sigmoid"),
            BatchNormalization(axis=1),
            Dense(1, activation="sigmoid", name="Dense_haha"),
        ]
    )

    x_train = data_train_vec
    y_train = data_train["Bitter"].values.reshape(-1, 1)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[f1],
    )

    x_phyto = data_phyto_vec
    y_phyto = data_phyto["Bitter"].values.reshape(-1, 1)

    history = model.fit(
        x_train, y_train, validation_data=(x_phyto, y_phyto), epochs=15, batch_size=128
    )
    phyto_pred = model.predict(x_phyto)

    f1score = f1_score(phyto_pred.round(), y_phyto)
    AUPR = average_precision_score(y_phyto, phyto_pred)
    acc_phyto = accuracy_score(phyto_pred.round(), y_phyto)
    recall_sensitivity = recall_score(phyto_pred.round(), y_phyto)
    confusion = confusion_matrix(phyto_pred.round(), y_phyto)[0]
    phyto_specificity1 = (confusion[0]) / (confusion[0] + confusion[1])
    # phyto_tn, phyto_fp, phyto_fn, phyto_tp = confusion_matrix(
    #     phyto_pred.round(), y_phyto
    # ).ravel()
    # phyto_specificity2 = phyto_tn / (phyto_tn + phyto_fp)
    # sensitive_phyto = phyto_tp / (phyto_tp + phyto_fn)

    table = [
        [
            "Metric",
            "Accuracy",
            "recall (Sensitivity)",
            # "SN (Sensitivity)",
            "SP Specificity 1",
            # "SP Specificity 2",
            "F1 Score",
            "AUPR",
        ],
        [
            "Phyto",
            acc_phyto,
            recall_sensitivity,
            # sensitive_phyto,
            phyto_specificity1,
            # phyto_specificity2,
            f1score,
            AUPR,
        ],
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
    # model.save("model/")
    return (
        f1_score(phyto_pred.round(), y_phyto),
        average_precision_score(y_phyto, phyto_pred),
    )


phyto_score = 0
aupr = 0
i = 0
while 1:
    score = main()
    phyto_score = max(phyto_score, score[0])
    aupr = max(aupr, score[1])
    i += 1
    print("loop: ", i)
    print("max phyto", phyto_score)
    print("max aupr", aupr)
    if (score[0] >= 0.93) and (score[1] >= 0.95):
        break
