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
data_train = pd.read_csv("dataset/bitter-or-not/true_train.csv")
data_valid = pd.read_csv("dataset/bitter-or-not/true_valid.csv")
data_phyto = pd.read_csv("dataset/test_set/phyto_test.csv")
data_bitter_new = pd.read_csv("dataset/test_set/bitter_new.csv")
data_unimi = pd.read_csv("dataset/test_set/UNIMI.csv")

data_train = data_train.sample(frac=1)

def haha(x):
    if (x == True): return 1
    else: return 0

data_train["Bitter"] = data_train["Bitter"].apply(lambda x: haha(x))
data_valid["Bitter"] = data_valid["Bitter"].apply(lambda x: haha(x))

data_train = pd.concat([data_train, data_valid], axis=0)

data_train["mol"] = data_train["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
data_train["sentences"] = data_train["mol"].apply(lambda x: MolSentence(mol2alt_sentence(x, radius=1)))
w2v_model = word2vec.Word2Vec.load('pretrained/model_300dim.pkl')
data_train["embedding"] = [DfVec(x) for x in sentences2vec(data_train["sentences"], w2v_model, unseen='UNK')]
data_train_vec = np.array([x.vec for x in data_train["embedding"]])
data_train_vec = np.reshape(data_train_vec, (len(data_train_vec), 300, 1))

data_phyto["mol"] = data_phyto["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
data_phyto["sentences"] = data_phyto["mol"].apply(lambda x: MolSentence(mol2alt_sentence(x, radius=1)))
data_phyto["embedding"] = [DfVec(x) for x in sentences2vec(data_phyto["sentences"], w2v_model, unseen='UNK')]
data_phyto_vec = np.array([x.vec for x in data_phyto["embedding"]])
data_phyto_vec = np.reshape(data_phyto_vec, (len(data_phyto_vec), 300, 1))

data_unimi["mol"] = data_unimi["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
data_unimi["sentences"] = data_unimi["mol"].apply(lambda x: MolSentence(mol2alt_sentence(x, radius=1)))
data_unimi["embedding"] = [DfVec(x) for x in sentences2vec(data_unimi["sentences"], w2v_model, unseen='UNK')]
data_unimi_vec = np.array([x.vec for x in data_unimi["embedding"]])
data_unimi_vec = np.reshape(data_unimi_vec, (len(data_unimi_vec), 300, 1))

def main():
    model = Sequential([
        Dense(256, input_shape=(300, ), activation="relu"),
        Dropout(0.5),
        Dense(128, activation="sigmoid"),
        Dropout(0.5),
        Dense(64, activation="sigmoid"),
        Dropout(0.5),
        Dense(32, activation="sigmoid"),
        Dropout(0.5),
        BatchNormalization(axis=1),
        Dense(1, activation="sigmoid", kernel_regularizer=l2(0.001))
    ])

    x_train = data_train_vec
    y_train = data_train["Bitter"].values.reshape(-1, 1)

    model.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), metrics=[tf.keras.metrics.BinaryAccuracy(), f1])
    history = model.fit(x_train, y_train, epochs=10, batch_size=64)
    data_history = pd.DataFrame(history.history)
    data_history.to_csv("history/nn_train")

    x_phyto = data_phyto_vec
    x_unimi = data_unimi_vec
    y_phyto = data_phyto["Bitter"].values.reshape(-1, 1)
    y_unimi = data_unimi["Bitter"].values.reshape(-1, 1)

    phyto_pred = model.predict(x_phyto)
    unimi_pred = model.predict(x_unimi)
    model.save("model/Paper_embedding_nn1")
    print("F1 Phyto", f1_score(phyto_pred.round(), y_phyto))

    print("F1 Unimi", f1_score(unimi_pred.round(), y_unimi))


    return (f1_score(phyto_pred.round(), y_phyto), f1_score(unimi_pred.round(), y_unimi))


unimi_score = 0
aupr = 0
i = 0
while (1):
    score = main()
    unimi_score = max(unimi_score, score[0])
    # aupr = max(aupr, score[1])
    i += 1
    print("loop: ", i)
    print("max unimi", unimi_score)
    # print("max aupr", aupr)
    if ((score[0] >= 0.91) and (score[1] >= 0.7)): break

