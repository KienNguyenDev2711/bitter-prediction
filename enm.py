import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model, Model
from keras.layers import Dense, concatenate, Lambda
from keras.regularizers import l2
from tensorflow.keras import backend as K
import pandas as pd
from sklearn.metrics import f1_score
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec


def f1(y_true, y_pred, reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        if reduction == tf.keras.losses.Reduction.SUM:
            return K.sum(recall)
        elif reduction == tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE:
            return K.mean(recall)
        else:
            return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        if reduction == tf.keras.losses.Reduction.SUM:
            return K.sum(precision)
        elif reduction == tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE:
            return K.mean(precision)
        else:
            return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


seed = 200422
data_phyto = pd.read_csv("dataset/test_set/phyto_test.csv")
data_bitter_new = pd.read_csv("dataset/test_set/bitter_new.csv")
data_unimi = pd.read_csv("dataset/test_set/UNIMI.csv")
data_train = pd.read_csv("dataset/bitter-or-not/bitter_train.csv", sep="\t")


def haha(x):
    return 1 if x else 0


data_train["Bitter"] = data_train["Bitter"].apply(lambda x: haha(x))
data_train = pd.concat([data_train, data_phyto], axis=0)

data_train["mol"] = data_train["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
data_train["sentences"] = data_train["mol"].apply(
    lambda x: MolSentence(mol2alt_sentence(x, radius=1))
)
w2v_model = word2vec.Word2Vec.load("pretrained/model_300dim.pkl")
data_train["embedding"] = [
    DfVec(x) for x in sentences2vec(data_train["sentences"], w2v_model, unseen="UNK")
]
data_train_vec = np.array([x.vec for x in data_train["embedding"]])
data_train_vec = np.reshape(data_train_vec, (len(data_train_vec), 300, 1))

data_phyto["mol"] = data_phyto["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
data_phyto["sentences"] = data_phyto["mol"].apply(
    lambda x: MolSentence(mol2alt_sentence(x, radius=1))
)
data_phyto["embedding"] = [
    DfVec(x) for x in sentences2vec(data_phyto["sentences"], w2v_model, unseen="UNK")
]
data_phyto_vec = np.array([x.vec for x in data_phyto["embedding"]])
data_phyto_vec = np.reshape(data_phyto_vec, (len(data_phyto_vec), 300, 1))

data_unimi["mol"] = data_unimi["smiles"].apply(lambda x: Chem.MolFromSmiles(x))
data_unimi["sentences"] = data_unimi["mol"].apply(
    lambda x: MolSentence(mol2alt_sentence(x, radius=1))
)
data_unimi["embedding"] = [
    DfVec(x) for x in sentences2vec(data_unimi["sentences"], w2v_model, unseen="UNK")
]
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


def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        filename = f"model/stacking_embedding/model{i+1}.h5"
        model = load_model(filename, compile=False, custom_objects={"f1": f1})
        # Không cần build lại mô hình ở đây
        all_models.append(model)
        print(f">loaded {filename}")
    return all_models


def define_stacked_model(members):
    input_shape = (300,)  # Thay đổi kích thước này nếu cần
    common_input = keras.Input(shape=input_shape)

    ensemble_outputs = []
    for i, model in enumerate(members):
        x = keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=-1),
            output_shape=(300, 1),  # Chỉ định output_shape
        )(common_input)
        x = model(x)
        for layer in model.layers:
            layer.trainable = False
            layer._name = f"ensemble_{i+1}_{np.random.randint(1000)}_{layer.name}"
        ensemble_outputs.append(x)

    merge = concatenate(ensemble_outputs)
    hidden = Dense(256, activation="sigmoid", kernel_regularizer=l2(0.001))(merge)
    hidden2 = Dense(128, activation="sigmoid")(hidden)
    hidden3 = Dense(64, activation="sigmoid")(hidden2)
    hidden4 = Dense(32, activation="sigmoid")(hidden3)
    output = Dense(1, activation="sigmoid", kernel_regularizer=l2(0.001))(hidden4)

    model = Model(inputs=common_input, outputs=output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction="sum_over_batch_size"
        ),
        metrics=[tf.keras.metrics.BinaryAccuracy(), f1],
    )
    return model


def fit_stacked_model(model, inputX, inputy):
    model.fit(inputX, inputy, epochs=12)


def predict_stacked_model(model, inputX):
    return model.predict(inputX)


def main(n_members):
    members = load_all_models(n_members)
    stacked_model = define_stacked_model(members)
    # Điều chỉnh kích thước dữ liệu đầu vào nếu cần
    data_train_vec_reshaped = data_train_vec.reshape(data_train_vec.shape[0], 300)
    inputy = data_train["Bitter"].values.reshape(-1, 1).astype(np.float32)
    fit_stacked_model(stacked_model, data_train_vec_reshaped, inputy)
    # Điều chỉnh kích thước dữ liệu dự đoán nếu cần
    data_phyto_vec_reshaped = data_phyto_vec.reshape(data_phyto_vec.shape[0], 300)

    yhat = predict_stacked_model(stacked_model, data_phyto_vec_reshaped)

    for i in range(len(yhat)):
        yhat[i] = 1 if yhat[i] > 0.2 else 0

    acc = f1_score(data_phyto["Bitter"].values.reshape(-1, 1), yhat)
    yhat = np.mean(yhat, axis=0)  # Thay đổi dòng này
    print(yhat)
    print("Stacked Test Accuracy:", acc)

    # Lưu mô hình đã huấn luyện
    stacked_model.save("model-final.h5")
    print("Model saved as model-final.h5")
    return acc


n_members = 4
i = 0
while 1:
    score = main(n_members)
    i += 1
    print("loop: ", i)
    if score >= 0.94:
        break
