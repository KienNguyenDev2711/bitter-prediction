import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from keras import backend as K

from sklearn.metrics import f1_score


# Định nghĩa hàm f1
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


# Đoạn mã xử lý dữ liệu
seed = 200422
data_phyto = pd.read_csv("dataset/test_set/phyto_test.csv")
data_bitter_new = pd.read_csv("dataset/test_set/bitter_new.csv")
data_unimi = pd.read_csv("dataset/test_set/UNIMI.csv")
data_train = pd.read_csv("dataset/bitter-or-not/bitter_train.csv", sep="\t")

data_unimi_vec = np.array([x.vec for x in data_unimi["embedding"]])
data_unimi_vec = np.reshape(data_unimi_vec, (len(data_unimi_vec), 300, 1))

data_bitter_new["mol"] = data_bitter_new["smiles"].apply(
    lambda x: Chem.MolFromSmiles(x)
)
data_bitter_new["sentences"] = data_bitter_new["mol"].apply(
    lambda x: MolSentence(mol2alt_sentence(x, radius=1))
)
data_bitter_new["embedding"] = [
    DfVec(x)
    for x in sentences2vec(data_bitter_new["sentences"], w2v_model, unseen="UNK")
]
data_bitter_vec = np.array([x.vec for x in data_bitter_new["embedding"]])
data_bitter_vec = np.reshape(data_bitter_vec, (len(data_bitter_vec), 300, 1))


# Định nghĩa hàm load_all_models
def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        filename = "model/stacking_embedding/model" + str(i + 1) + ".h5"
        model = load_model(filename, custom_objects={"f1": f1})
        all_models.append(model)
    return all_models


# Định nghĩa mô hình stacked
def define_stacked_model(members):
    inputs = keras.Input(shape=(300, 1))
    x = keras.layers.Flatten()(inputs)
    x = keras.layers.Dense(10, activation="relu")(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Cấu hình mô hình
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction="sum_over_batch_size"
        ),
        metrics=[tf.keras.metrics.BinaryAccuracy(), f1],
    )
    return model


# Định nghĩa hàm fit_stacked_model
def fit_stacked_model(model, inputX, inputy):
    model.fit(
        inputX,
        inputy,
        epochs=12,
    )


# Định nghĩa hàm predict_stacked_model
def predict_stacked_model(model, inputX):
    return model.predict(inputX)


# Hàm main
def main(n_members):
    members = load_all_models(n_members)
    stacked_model = define_stacked_model(members)
    fit_stacked_model(
        stacked_model, data_bitter_vec, data_bitter_new["Bitter"].values.reshape(-1, 1)
    )
    yhat = predict_stacked_model(stacked_model, data_unimi_vec)

    for i in range(len(yhat)):
        if yhat[i] > 0.2:
            yhat[i] = 1
        else:
            yhat[i] = 0

    acc = f1_score(data_phyto["Bitter"].values.reshape(-1, 1), yhat)
    yhat = np.average(yhat, axis=1)
    print(yhat)
    print("Stacked Test Accuracy:", acc)
    return acc


n_members = 4
score = main(n_members)
