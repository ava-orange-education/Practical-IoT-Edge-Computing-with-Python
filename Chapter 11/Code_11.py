#Requires numpy <2.0 and Python < 3.11
from secml.adv.attacks import CAttackEvasionPGD
from secml.adv.attacks import CAttackPoisoningSVM
from secml.ml.classifiers import CClassifierSVM
from secml.data import CDataset
import pandas as pd
# Load a dataset
df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
dataset = CDataset(X, y)

# Train a simple SVM classifier
clf = CClassifierSVM(kernel='linear')
clf.fit(dataset.X,dataset.Y)

# Create an evasion attack
attack = CAttackEvasionPGD(classifier=clf, distance='l2')

# Perform the attack
x_adv = attack.run(dataset.X[0, :], dataset.Y[0])
print("Adversarial example:", x_adv)

attack = CAttackPoisoningSVM(classifier=clf, training_data=dataset, val=dataset, solver_params={'eta': 0.1})

# Perform the poisoning attack
poisoned_data = attack.run(dataset.X[0, :], dataset.Y[0])
print(f"Poisoned Data: {poisoned_data}")
