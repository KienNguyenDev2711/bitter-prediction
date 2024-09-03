from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from rdkit import Chem
from mol2vec.features import mol2alt_sentence, sentences2vec
from gensim.models import word2vec
import numpy as np

app = Flask(__name__)

# Tải mô hình và mô hình word2vec
model = load_model("model/bitter_model.h5")
w2v_model = word2vec.Word2Vec.load("pretrained/model_300dim.pkl")


def smiles_to_mol(smiles):
    return Chem.MolFromSmiles(smiles)


def mol_to_sentence(mol):
    return mol2alt_sentence(mol, radius=1)


def sentence_to_vec(sentence, model):
    return sentences2vec([sentence], model, unseen="UNK")[0]


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        # Kiểm tra đầu vào
        if "smiles" not in data:
            return jsonify({"error": "Missing SMILES input"}), 400

        smiles = data["smiles"]
        mol = smiles_to_mol(smiles)
        if mol is None:
            return jsonify({"error": "Invalid SMILES input"}), 400

        sentence = mol_to_sentence(mol)
        vec = sentence_to_vec(sentence, w2v_model)
        vec = np.array(vec).reshape(1, -1)
        prediction = model.predict(vec)
        result = {"bitter": bool(prediction[0][0] > 0.5)}
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
