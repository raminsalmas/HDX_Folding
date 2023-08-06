import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, confusion_matrix, accuracy_score, balanced_accuracy_score, roc_auc_score)
import sys
from random import sample
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (cross_val_score, cross_val_predict, cross_validate, RepeatedStratifiedKFold, KFold, StratifiedKFold, HalvingGridSearchCV, GridSearchCV, train_test_split)
from sklearn.preprocessing import MinMaxScaler, StandardScaler, label_binarize, LabelEncoder
from sklearn.calibration import calibration_curve, CalibrationDisplay
from sklearn.tree import export_graphviz
import graphviz

# Graphs in png and pdf formats
def plot(name):
    # Function to save the plot in png format
    plt.savefig(name,
                dpi=600,
                facecolor='w',
                edgecolor='w',
                orientation='portrait',
                format="png",
                transparent=None,
                bbox_inches="tight", )
    return 0

def plot_pdf(name):
    # Function to save the plot in pdf format
    plt.savefig(name,
                facecolor='w',
                edgecolor='w',
                orientation='portrait',
                format="pdf",
                transparent=None,
                bbox_inches="tight", )
    return 0

def load_data(file_path):
    # Function to load data from a CSV file
    data = np.loadtxt(file_path, delimiter=",", dtype=np.str)
    return data

def prepare_labels(data, label_col):
    # Function to preprocess labels using LabelEncoder
    le = LabelEncoder()
    labels = le.fit_transform(data[:, label_col])
    return labels, le

def prepare_features(data, feature_cols):
    # Function to prepare features for training
    features = data[:, feature_cols].astype(np.float32)
    return features

def train_model(X, y, hparam):
    # Function to train the Gradient Boosting Classifier model
    model_gb = GradientBoostingClassifier(**hparam, n_jobs=-1)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=32)
    n_scores = cross_val_score(model_gb, X=X, y=y, scoring='accuracy', cv=cv, n_jobs=-1)
    return model_gb, np.mean(n_scores), np.std(n_scores)

def evaluate_model(data_x, data_y, model):
    # Function to evaluate the model using cross-validation
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=32)
    predicted_targets = np.array([])
    actual_targets = np.array([])
    predicted_targets_prob = np.array([])

    for train_ix, test_ix in cv.split(data_x, data_y):
        train_x, train_y, test_x, test_y = data_x[train_ix], data_y[train_ix], data_x[test_ix], data_y[test_ix]
        
        classifiers = model
        classifiers.fit(train_x, train_y)
        predicted_labels = classifiers.predict(test_x)
        predicted_prob = classifiers.predict_proba(test_x)[:, 1]
        
        predicted_targets = np.append(predicted_targets, predicted_labels)
        predicted_targets_prob = np.append(predicted_targets_prob, predicted_prob)
        actual_targets = np.append(actual_targets, test_y)
          
    return predicted_targets, actual_targets, predicted_targets_prob

def grid_search(model, X, y):
    # Function to perform Grid Search for hyperparameter tuning
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=32)

    param_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1, 2, 3],
                  'max_depth': np.arange(1, 10),
                  'n_estimators': [50, 100, 200, 400, 500],
                  'min_samples_split': [1, 2, 3, 4, 5],
                  'min_samples_leaf': [1, 2, 3, 4, 5]
                  }
    
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid_result = grid.fit(X, y)
    return grid_result.best_score_, grid_result.best_params_

def plot_confusion_matrix(predicted_targets, actual_targets):
    # Function to plot the confusion matrix
    plt.grid('off')
    cm = confusion_matrix(predicted_targets, actual_targets, normalize='true')
    ax = axes[1]
    ti = sns.heatmap(cm, cmap=plt.cm.Blues, annot=True, ax=ax)
    ti.invert_yaxis()
    accuracy = accuracy_score(predicted_targets, actual_targets)
    ax.set_title('Confusion matrix')
    ax.set_xlabel('True label')
    ax.set_ylabel('Predicted label')
    ax.grid('off')

def plot_calibration_curve(data_x, data_y, model):
    # Function to plot the calibration curve
    ax = axes[2]
    x_, y_ = calibration_curve(data_y, model.predict_proba(data_x)[:, 1], n_bins=20)
    sns.regplot(x=x_, y=y_, ax=ax)
    ax.set_title('Calibration curve')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_ylim(0, 1.03)
    ax.set_xlim(0, 1.03)

def plot_roc_curve(data_x, data_y, model):
    # Function to plot the ROC curve
    ax = axes[3]
   
