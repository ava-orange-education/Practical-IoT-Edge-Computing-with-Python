import tenseal as ts
import numpy as np
from sklearn.svm import SVC
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#Split the dataset

df = pd.read_csv("heart_failure_clinical_records_dataset.csv")
X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    poly_modulus_degree=8192,
    coeff_mod_bit_sizes=[60, 40, 40, 60]
)
context.generate_galois_keys()
context.global_scale = 2**40

# Train the SVM model
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train, y_train)

# Extracting relevant parameters (for a linear SVM)

weights = svm_model.coef_[0]
bias = svm_model.intercept_[0]

# Parameter Encryption
enc_weights = ts.ckks_vector(context, weights.tolist())
enc_bias = ts.ckks_vector(context, [bias]) # Bias as a single-element vector

# Data Encryption (for a single test example)
sample_x = X_test[0]
enc_sample_x = ts.ckks_vector(context, sample_x.tolist())

#Encrypted Inference 
encrypted_prediction_result = enc_sample_x.dot(enc_weights) + enc_bias

#  Decryption
decrypted_prediction_result = encrypted_prediction_result.decrypt()

print(f"Original SVM prediction for sample_x: {svm_model.predict(sample_x.reshape(1, -1))}")
print(f"Decrypted homomorphic prediction result: {decrypted_prediction_result}")