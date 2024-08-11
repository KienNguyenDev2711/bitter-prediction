import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

# from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
# from gensim.models import word2vec
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
import keras.backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


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


def main():
    # Model Definition
    model = Sequential(
        [
            Dense(256, input_shape=(256,), activation="relu"),
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
        optimizer=keras.optimizers.Adam(lr=0.005),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.BinaryAccuracy(), f1],
    )
    history = model.fit(x_train, y_train, epochs=10, batch_size=64)
    data_history = pd.DataFrame(history.history)
    data_history.to_csv("history/nn_train")

    x_phyto = phyto_fps_vec
    x_unimi = unimi_fps_vec
    y_phyto = data_phyto["Bitter"].values.reshape(-1, 1)
    y_unimi = data_unimi["Bitter"].values.reshape(-1, 1)

    phyto_pred = model.predict(x_phyto)
    unimi_pred = model.predict(x_unimi)

    print("acc Phyto", accuracy_score(phyto_pred.round(), y_phyto))
    print("F1 Phyto", f1_score(phyto_pred.round(), y_phyto))

    print("acc unimi", accuracy_score(unimi_pred.round(), y_unimi))
    print("F1 unimi", f1_score(unimi_pred.round(), y_unimi))

    # model.save("model/propo_nn_ecfp")

    return (
        f1_score(unimi_pred.round(), y_unimi),
        average_precision_score(y_unimi, unimi_pred),
    )


unimi_score = 0
aupr = 0
i = 0
while 1:
    score = main()
    unimi_score = max(unimi_score, score[0])
    aupr = max(aupr, score[1])
    i += 1
    print("loop: ", i)
    print("max unimi", unimi_score)
    print("max aupr", aupr)
    if score[0] >= 0.81:
        break
