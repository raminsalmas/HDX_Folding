import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, roc_auc_score
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.tree import export_graphviz
import graphviz


class DataProcessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self):
        data_1 = np.loadtxt(self.file_path, delimiter=",", dtype=np.str)
        return data_1

    def preprocess_data(self, data_1):
        (
            lnp, sasa, phi, psi, r_matrix,
            SS, k_int, k_obs, res, L, R, LL, RR
        ) = (
            data_1[:, 2].astype(np.float32),
            data_1[:, 4].astype(np.float32),
            data_1[:, 5].astype(np.float32),
            data_1[:, 6].astype(np.float32),
            data_1[:, 7].astype(np.float32),
            data_1[:, 3],
            data_1[:, 8].astype(np.float32),
            data_1[:, 9].astype(np.float32),
            data_1[:, 0],
            data_1[:, 11],
            data_1[:, 12],
            data_1[:, 13],
            data_1[:, 14]
        )

        le = LabelEncoder()
        res = le.fit_transform(res)
        L = le.fit_transform(L)
        R = le.fit_transform(R)
        LL = le.fit_transform(LL)
        RR = le.fit_transform(RR)

        Kio = np.concatenate(
            (res.reshape(-1, 1), k_int.reshape(-1, 1), r_matrix.reshape(-1, 1), k_obs.reshape(1, -1).T),
            axis=1,
        )

        aa = np.concatenate(
            (phi.reshape(-1, 1), psi.reshape(1, -1).T, sasa.reshape(1, -1).T), axis=1
        )

        aa_phi_psi = np.concatenate((phi.reshape(-1, 1), psi.reshape(1, -1).T), axis=1)

        aa_phi_psi_lnp = np.concatenate((phi.reshape(-1, 1), psi.reshape(1, -1).T, lnp.reshape(1, -1).T), axis=1)

        aa_phi_psi_sasa_lnp = np.concatenate(
            (phi.reshape(-1, 1), psi.reshape(1, -1).T, sasa.reshape(1, -1).T, lnp.reshape(1, -1).T),
            axis=1,
        )

        aa_phi_psi_sasa = np.concatenate((phi.reshape(-1, 1), psi.reshape(1, -1).T, sasa.reshape(1, -1).T), axis=1)

        return Kio, aa_phi_psi, aa_phi_psi_lnp, aa_phi_psi_sasa_lnp, aa_phi_psi_sasa


class ModelCreator:
    def __init__(self, hparam):
        self.hparam = hparam

    def create_model(self):
        model_gb = GradientBoostingClassifier(**self.hparam, n_jobs=-1)
        return model_gb


class ModelEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate_model(self, X, y):
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=32)
        predicted_targets = np.array([])
        actual_targets = np.array([])
        predicted_targets_prob = np.array([])

        for train_ix, test_ix in cv.split(X, y):
            train_x, train_y, test_x, test_y = X[train_ix], y[train_ix], X[test_ix], y[test_ix]

            classifiers = self.model
            classifiers.fit(train_x, train_y)
            predicted_labels = classifiers.predict(test_x)
            predicted_prob = classifiers.predict_proba(test_x)[:, 1]

            predicted_targets = np.append(predicted_targets, predicted_labels)
            predicted_targets_prob = np.append(predicted_targets_prob, predicted_prob)
            actual_targets = np.append(actual_targets, test_y)

        return predicted_targets, actual_targets, predicted_targets_prob


if __name__ == "__main__":
    # Name of the dataset file
    file_1 = "data5.csv"

    # Initialize DataProcessor and load data
    data_processor = DataProcessor(file_1)
    data_1 = data_processor.load_data()

    # Preprocess data
    Kio, aa_phi_psi, aa_phi_psi_lnp, aa_phi_psi_sasa_lnp, aa_phi_psi_sasa = data_processor.preprocess_data(data_1)

    # Create model
    hparam = dict(
        learning_rate=0.2,
        max_depth=1,
        max_features='log2',
        n_estimators=500,
        random_state=32,
    )
    model_creator = ModelCreator(hparam)
    model_gb = model_creator.create_model()

    # Evaluate model
    model_evaluator = ModelEvaluator(model_gb)
    predicted_targets, actual_targets, predicted_targets_prob = model_evaluator.evaluate_model(Kio, aa_phi_psi)

    
