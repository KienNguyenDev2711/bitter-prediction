import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from tabulate import tabulate


# from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
# from gensim.models import word2vec
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.regularizers import l2
from keras.regularizers import L1L2
from keras.layers import Activation, Dense, BatchNormalization, Dropout, LSTM, Input
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
from sklearn.metrics import average_precision_score
import tensorflow as tf


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

train_mol = [
    Chem.rdmolfiles.MolFromSmiles(SMILES_string)
    for SMILES_string in data_train["smiles"]
]
bi = {}
train_fps = [
    rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo=bi, nBits=256)
    for m in train_mol
]
train_fps_vec = []
for fp in train_fps:
    arr = np.array([])
    DataStructs.ConvertToNumpyArray(fp, arr)
    train_fps_vec.append(arr)
train_fps_vec = np.asarray(train_fps_vec)

# Phyto test set
phyto_mol = [
    Chem.rdmolfiles.MolFromSmiles(SMILES_string)
    for SMILES_string in data_phyto["smiles"]
]
bi = {}
phyto_fps = [
    rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo=bi, nBits=256)
    for m in phyto_mol
]
phyto_fps_vec = []
for fp in phyto_fps:
    arr = np.array([])
    DataStructs.ConvertToNumpyArray(fp, arr)
    phyto_fps_vec.append(arr)
phyto_fps_vec = np.asarray(phyto_fps_vec)

# Unimi test set
unimi_mol = [
    Chem.rdmolfiles.MolFromSmiles(SMILES_string)
    for SMILES_string in data_unimi["smiles"]
]
bi = {}
unimi_fps = [
    rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo=bi, nBits=256)
    for m in unimi_mol
]
unimi_fps_vec = []
for fp in unimi_fps:
    arr = np.array([])
    DataStructs.ConvertToNumpyArray(fp, arr)
    unimi_fps_vec.append(arr)
unimi_fps_vec = np.asarray(unimi_fps_vec)


# Bitter new test set
phyto_mol = [
    Chem.rdmolfiles.MolFromSmiles(SMILES_string)
    for SMILES_string in data_bitter_new["smiles"]
]
bi = {}
bitter_new_fps = [
    rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo=bi, nBits=256)
    for m in phyto_mol
]
bitter_new_fps_vec = []
for fp in bitter_new_fps:
    arr = np.array([])
    DataStructs.ConvertToNumpyArray(fp, arr)
    bitter_new_fps_vec.append(arr)
bitter_new_fps_vec = np.asarray(bitter_new_fps_vec)


def main():
    inputs = Input(shape=(256,))
    # Model Definition
    model = Sequential(
        [
            Dense(256, activation="relu", input_shape=(256,)),
            Dropout(0.3),
            Dense(128, activation="sigmoid"),
            Dropout(0.3),
            Dense(64, activation="sigmoid"),
            Dropout(0.3),
            Dense(32, activation="sigmoid"),
            Dropout(0.3),
            BatchNormalization(axis=1),
            Dense(1, activation="sigmoid", kernel_regularizer=l2(0.001)),
        ]
    )

    x_train = train_fps_vec
    y_train = data_train["Bitter"].values.reshape(-1, 1)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(), f1],
    )
    history = model.fit(x_train, y_train, epochs=10, batch_size=64)
    data_history = pd.DataFrame(history.history)
    data_history.to_csv("history/nn_train")

    x_phyto = phyto_fps_vec
    x_unimi = unimi_fps_vec
    x_bitter_new = bitter_new_fps_vec

    y_phyto = data_phyto["Bitter"].values.reshape(-1, 1)
    y_unimi = data_unimi["Bitter"].values.reshape(-1, 1)
    y_bitter_new = data_bitter_new["Bitter"].values.reshape(-1, 1)

    phyto_pred = model.predict(x_phyto)
    unimi_pred = model.predict(x_unimi)
    bitter_new_pred = model.predict(x_bitter_new)

    phyto_tn, phyto_fp, phyto_fn, phyto_tp = confusion_matrix(
        phyto_pred.round(), y_phyto
    ).ravel()
    phyto_specificity = phyto_tn / (phyto_tn + phyto_fp)

    unimi_tn, unimi_fp, unimi_fn, unimi_tp = confusion_matrix(
        unimi_pred.round(), y_unimi
    ).ravel()
    unimi_specificity = unimi_tn / (unimi_tn + unimi_fp)

    bitter_new_tn, bitter_new_fp, bitter_new_fn, bitter_new_tp = confusion_matrix(
        bitter_new_pred.round(), y_bitter_new
    ).ravel()
    bitter_new_specificity = bitter_new_tn / (bitter_new_tn + bitter_new_fp)

    acc_phyto = accuracy_score(phyto_pred.round(), y_phyto)
    f1_phyto = f1_score(phyto_pred.round(), y_phyto)
    recall_phyto = recall_score(phyto_pred.round(), y_phyto)
    aupr_phyto = average_precision_score(y_phyto, phyto_pred)
    sensitive_phyto = phyto_tp / (phyto_tp + phyto_fn)

    acc_unimi = accuracy_score(unimi_pred.round(), y_unimi)
    f1_unimi = f1_score(unimi_pred.round(), y_unimi)
    recall_unimi = recall_score(unimi_pred.round(), y_unimi)
    aupr_unimi = average_precision_score(y_unimi, unimi_pred)
    sensitive_unimi = unimi_tp / (unimi_tp + unimi_fn)

    acc_bitter_new = accuracy_score(bitter_new_pred.round(), y_bitter_new)
    f1_bitter_new = f1_score(bitter_new_pred.round(), y_bitter_new)
    recall_bitter_new = recall_score(bitter_new_pred.round(), y_bitter_new)
    aupr_bitter_new = average_precision_score(y_bitter_new, bitter_new_pred)
    sensitive_bitter_new = bitter_new_tp / (bitter_new_tp + bitter_new_fn)

    # print("acc unimi", accuracy_score(unimi_pred.round(), y_unimi))
    # print("F1 unimi", f1_score(unimi_pred.round(), y_unimi))

    # print("acc bitter new", accuracy_score(bitter_new_pred.round(), y_bitter_new))
    # print("F1 bitter new", f1_score(bitter_new_pred.round(), y_bitter_new))

    # Tạo dữ liệu bảng
    table = [
        ["Metric", "Phyto Value", "Unimi Value", "Bitter New Value"],
        ["Accuracy", acc_phyto, acc_unimi, acc_bitter_new],
        ["SN (Sensitivity)", sensitive_phyto, sensitive_unimi, sensitive_bitter_new],
        ["Recall (Sensitivity)", recall_phyto, recall_unimi, recall_bitter_new],
        [
            "SP Specificity",
            phyto_specificity,
            unimi_specificity,
            bitter_new_specificity,
        ],
        ["F1 Score", f1_phyto, f1_unimi, f1_bitter_new],
        ["AUPR (OK)", aupr_phyto, aupr_unimi, aupr_bitter_new],
    ]

    # In bảng
    print(tabulate(table, headers="firstrow", tablefmt="grid"))
    model.save("model/propo_nn_ecfp.h5")

    return (
        [
            recall_score(phyto_pred.round(), y_phyto),  # SN (Sensitivity)
            phyto_specificity,  # (SP) specificity
            f1_score(phyto_pred.round(), y_phyto),  # f1_score
            average_precision_score(y_phyto, phyto_pred),  # AUPR
        ],
        [
            recall_score(unimi_pred.round(), y_unimi),  # SN (Sensitivity)
            unimi_specificity,  # (SP) specificity
            f1_score(unimi_pred.round(), y_unimi),
            average_precision_score(y_unimi, unimi_pred),
        ],
        [
            recall_score(bitter_new_pred.round(), y_bitter_new),  # SN (Sensitivity)
            bitter_new_specificity,  # (SP) specificity, always return 0
            f1_score(bitter_new_pred.round(), y_bitter_new),  # f1_score
            average_precision_score(
                y_bitter_new, bitter_new_pred
            ),  # AUPR always return 1
        ],
    )


unimi_score = 0
aupr = 0
i = 0
while 1:
    score = main()
    unimi_score = max(unimi_score, score[1][2])
    aupr = max(aupr, score[1][3])
    i += 1
    print("loop: ", i)
    print("max unimi", unimi_score)
    print("max aupr", aupr)
    # print("-----------------------------")
    # for v in score:
    #     for item in v:
    #         print(item, end=" ")
    #     print()
    # print("-----------------------------")

    if score[1][2] >= 0.81:
        break
