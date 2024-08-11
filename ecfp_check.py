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
from sklearn.metrics import average_precision_score
import keras.backend as K
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from rdkit.Chem import rdMolDescriptors
from keras.layers import Flatten
from keras.layers import Embedding
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.preprocessing import sequence
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from keras.utils import pad_sequences
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
data_phyto = pd.read_csv("dataset/test_set/phyto_test.csv")
data_bitter_new = pd.read_csv("dataset/test_set/bitter_new.csv")
data_unimi = pd.read_csv("dataset/test_set/UNIMI.csv")

phyto_mol = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in data_phyto["smiles"]]
bi = {}
phyto_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=256) for m in phyto_mol]
phyto_fps_vec = []
for fp in phyto_fps:
    arr = np.array([])
    DataStructs.ConvertToNumpyArray(fp, arr)
    phyto_fps_vec.append(arr)
phyto_fps_vec = np.asarray(phyto_fps_vec)

unimi_mol = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in data_unimi["smiles"]]
bi = {}
unimi_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=256) for m in unimi_mol]
unimi_fps_vec = []
for fp in unimi_fps:
    arr = np.array([])
    DataStructs.ConvertToNumpyArray(fp, arr)
    unimi_fps_vec.append(arr)
unimi_fps_vec = np.asarray(unimi_fps_vec)

bitternew_mol = [Chem.rdmolfiles.MolFromSmiles(SMILES_string) for SMILES_string in data_bitter_new["smiles"]]
bi = {}
bitternew_fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(m, radius=2, bitInfo= bi, nBits=256) for m in bitternew_mol]
bitternew_fps_vec = []
for fp in bitternew_fps:
    arr = np.array([])
    DataStructs.ConvertToNumpyArray(fp, arr)
    bitternew_fps_vec.append(arr)
bitternew_fps_vec = np.asarray(bitternew_fps_vec)

model = tf.keras.models.load_model("model/Paper_ecfp_nn2", custom_objects={'f1':f1})

def confu_info(y_actual, y_hat, target):
    print(target)
    # cm = confusion_matrix(y_pred, y_real)
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)):
        if y_actual[i] == y_hat[i] == 1:
            TP += 1
        if y_hat[i] == 1 and y_actual[i] != y_hat[i]:
            FP += 1
        if y_actual[i] == y_hat[i] == 0:
            TN += 1
        if y_hat[i] == 0 and y_actual[i] != y_hat[i]:
            FN += 1
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    print("SP: ", TPR)
    print("SN: ", TNR)


pred = model.predict(unimi_fps_vec)
y_unimi = data_unimi["Bitter"].values.reshape(-1, 1)
confu_info(pred.round(), y_unimi, "Unimi")
print(pred.dtype)
print("F1 unimi", f1_score(pred.round(), y_unimi))
print("AUPR: ", average_precision_score(y_unimi, pred))

phyto_pred = model.predict(phyto_fps_vec)
y_phyto = data_phyto["Bitter"].values.reshape(-1, 1)
confu_info(phyto_pred.round(), y_phyto, "Phyto")
print("F1 phyto", f1_score(phyto_pred.round(), y_phyto))
print("AUPR: ", average_precision_score(y_phyto, phyto_pred))

bitternew = model.predict(bitternew_fps_vec)
y_bitter = data_bitter_new["Bitter"].values.reshape(-1, 1)
confu_info(bitternew.round(), y_bitter, "Bitternew")
print("F1 bitternew", f1_score(bitternew.round(), y_bitter))
print("AUPR: ", average_precision_score(y_bitter, bitternew))

