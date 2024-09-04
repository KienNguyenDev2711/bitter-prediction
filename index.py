# -*- coding: utf-8 -*-

from flask import Flask, request, jsonify
import numpy as np
from rdkit import Chem
from keras.layers import Dense, concatenate, Lambda
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input, Dropout
import h5py
from keras.regularizers import l2
from tensorflow.keras import backend as K

print("TensorFlow version:", tf.__version__)


def lambda_func(x):
    return tf.squeeze(x, axis=-1)


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


def load_all_models(n_models):
    all_models = list()
    for i in range(n_models):
        filename = "model/model-final.h5"
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


print("Checking model structure...")
with h5py.File("model/model-final.h5", "r") as f:
    print("Keys in the file:", list(f.keys()))
    if "model_weights" in f:
        print("Layers in model_weights:", list(f["model_weights"].keys()))

print("Creating new model...")
members = load_all_models(1)
model = define_stacked_model(members)

print("Attempting to load model weights...")
try:
    model.load_weights("model/model-final.h5")
    print("Model weights loaded successfully")
except Exception as e:
    print("Error loading model weights:", str(e))
    print("Attempting to load full model...")
    try:
        members = load_all_models(1)
        model = define_stacked_model(members)
        # model = load_model("model/model-final.h5", custom_objects={'f1': f1})
        print("Full model loaded successfully")
    except Exception as e:
        print("Error loading full model:", str(e))
        raise

# Tai mo hinh Word2Vec
w2v_model = word2vec.Word2Vec.load("pretrained/model_300dim.pkl")


def smiles_to_embedding(smiles):
    mol = Chem.MolFromSmiles(smiles)
    sentence = MolSentence(mol2alt_sentence(mol, radius=1))
    embedding = DfVec(sentences2vec([sentence], w2v_model, unseen="UNK")[0])
    return np.array(embedding.vec).reshape(1, 300)


# Khoi tao Flask app

app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    smiles = data.get("smiles")

    if not smiles:
        return jsonify({"error": "No SMILES string provided"}), 400

    try:
        embedding = smiles_to_embedding(smiles)
        print("Embedding shape:", embedding.shape)
        prediction = model.predict(embedding)
        print("Prediction:", prediction)
        is_bitter = bool(prediction[0][0] > 0.5)
        return jsonify({"bitter": is_bitter})
    except Exception as e:
        print("Error during prediction:", str(e))
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
