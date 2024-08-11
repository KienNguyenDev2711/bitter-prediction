from rdkit import Chem
from rdkit.Chem import AllChem
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load data
data = pd.read_csv('dataset/bitter-or-not/true_train.csv')
test_data = pd.read_csv('dataset/test_set/phyto_test.csv')

# Convert SMILES strings to molecular objects
molecules = [Chem.MolFromSmiles(smile) for smile in data['smiles']]

# Generate ECFP descriptors for each molecule
fingerprints = [AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048) for molecule in molecules]

# Convert ECFP descriptors to numpy array
X = np.asarray(fingerprints)

# Convert bitter/non-bitter labels to numpy array
y = np.asarray(data['Bitter'])

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

test_molecules = [Chem.MolFromSmiles(smile) for smile in test_data['smiles']]
test_fingerprints = [AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=2048) for molecule in test_molecules]
x_phyto = np.asarray(test_fingerprints)
y_phyto = np.asarray(test_data["Bitter"])

#
#
# Define neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(2048,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(x_phyto, y_phyto), epochs=10, batch_size=32)

# Evaluate model on testing data
loss, accuracy = model.evaluate(x_phyto, y_phyto)

print("Acc: ", accuracy)

