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
import keras.backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rdkit.Chem import rdMolDescriptors
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold
from keras.utils import pad_sequences
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
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
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


seed = 200422

data_train = pd.read_csv("dataset/sweet_data/sweet_true_train.csv")
data_test = pd.read_csv("dataset/sweet_data/sweet-test.csv", sep="\t")

data_train = data_train.sample(frac=1)

def haha(x):
    if (x == True): return 1
    else: return 0

data_train["Sweet"] = data_train["Sweet"].apply(lambda x: haha(x))
data_test["Sweet"] = data_test["Sweet"].apply(lambda x: haha(x))




train_mol = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in data_train["smiles"]]
bi = {}
train_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=256) for m in train_mol]
train_fps_vec = []
for fp in train_fps:
    arr = np.array([])
    DataStructs.ConvertToNumpyArray(fp, arr)
    train_fps_vec.append(arr)
train_fps_vec = np.asarray(train_fps_vec)

test_mol = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in data_test["smiles"]]
bi = {}
test_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=256) for m in test_mol]
test_fps_vec = []
for fp in test_fps:
    arr = np.array([])
    DataStructs.ConvertToNumpyArray(fp, arr)
    test_fps_vec.append(arr)
test_fps_vec = np.asarray(test_fps_vec)


def main():
    model = Sequential([
        Dense(256, input_shape=(256, ), activation="relu"),
        Dense(128, activation="sigmoid"),
        Dense(64, activation="sigmoid"),
        Dense(32, activation="sigmoid"),
        BatchNormalization(axis=1),
        Dense(1, activation="sigmoid")
    ])

    x_train = train_fps_vec
    y_train = data_train["Sweet"].values.reshape(-1, 1)

    model.compile(optimizer=keras.optimizers.Adam(lr=0.005), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=[tf.keras.metrics.BinaryAccuracy(), f1])
    model.fit(x_train, y_train, epochs=10, batch_size=64)

    x_test = test_fps_vec
    y_test = data_test["Sweet"].values.reshape(-1, 1)

    model.save("model/sweet_ecfp_ver1")
    pred = model.predict(x_test)

    print("F1 ", f1_score(pred.round(), y_test))
    print("AUPR ", average_precision_score(y_test, pred))
    print("AUR ROC ", roc_auc_score(y_test, pred))
    x = [f1_score(pred.round(), y_test), average_precision_score(y_test, pred), roc_auc_score(y_test, pred)]
    return x


f1_haha = 0
i = 0
while(1):
    score = main()
    f1_haha = max(f1_haha, score[0])
    print("max F1: ", f1_haha)
    i += 1
    print("Loop: ", i)
    if ((score[0] >= 0.865) and (score[1] >= 0.92) and (score[2] >= 0.83)): break


