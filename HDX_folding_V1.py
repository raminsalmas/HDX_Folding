#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import (GaussianMixture, BayesianGaussianMixture)
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, confusion_matrix, 
                             accuracy_score, balanced_accuracy_score, roc_auc_score)
import sys
from random import sample
import pandas as pd
from collections import Counter
from sklearn.utils import shuffle
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import (cross_val_score, cross_val_predict, cross_validate,
                                     RepeatedStratifiedKFold, KFold, StratifiedKFold, 
                                     HalvingGridSearchCV, GridSearchCV, train_test_split)

from sklearn.preprocessing import (MinMaxScaler, StandardScaler, label_binarize, LabelEncoder)
from sklearn.calibration import (calibration_curve, CalibrationDisplay)
from sklearn.tree import export_graphviz
import graphviz


#Graphs in png and pdf formats
def plot(name):
    plt.savefig(name,
                dpi=600,
                facecolor='w',
                edgecolor='w',
                orientation='portrait',
                format="png",
                transparent=None,
                bbox_inches="tight",  )
    return 0

def plot_pdf(name):
    plt.savefig(name,
                facecolor='w',
                edgecolor='w',
                orientation='portrait',
                format="pdf",
                transparent=None,
                bbox_inches="tight",  )
    return 0



#Name of the dataset file

file_1 = "data5.csv"


data_1 = np.loadtxt(file_1, delimiter=",", dtype=np.str)


( lnp, sasa, phi, psi, r_matrix, 
 SS, k_int, k_obs, res, L, R, LL, RR) = (


            (data_1[:, 2].astype(np.float32),
             data_1[:, 4].astype(np.float32),
             data_1[:,5].astype(np.float32),
             data_1[:,6].astype(np.float32),
             data_1[:,7].astype(np.float32),
             data_1[:, 3],
             data_1[:, 8].astype(np.float32), 
             data_1[:, 9].astype(np.float32),
             data_1[:, 0],
             data_1[:, 11],
             data_1[:, 12],
             data_1[:, 13],
             data_1[:, 14]  )
 )



le = LabelEncoder()


res = le.fit_transform(res)
L = le.fit_transform(L)
R = le.fit_transform(R)
LL = le.fit_transform(LL)
RR = le.fit_transform(RR)


Kio = np.concatenate((res.reshape(-1,1),
                       k_int.reshape(-1,1),
                       r_matrix.reshape(-1,1),
                       k_obs.reshape(1,-1).T), axis=1)




aa = np.concatenate((phi.reshape(-1,1),
                     psi.reshape(1,-1).T, 
                     sasa.reshape(1,-1).T), axis=1)

aa_phi_psi = np.concatenate((phi.reshape(-1,1),
                     psi.reshape(1,-1).T), axis=1)

aa_phi_psi_lnp = np.concatenate((phi.reshape(-1,1),
                     psi.reshape(1,-1).T, 
                     lnp.reshape(1,-1).T), axis=1)

aa_phi_psi_sasa_lnp = np.concatenate((phi.reshape(-1,1),
                     psi.reshape(1,-1).T, 
                     sasa.reshape(1,-1).T, 
                     lnp.reshape(1,-1).T), axis=1)

aa_phi_psi_sasa = np.concatenate((phi.reshape(-1,1),
                     psi.reshape(1,-1).T, 
                     sasa.reshape(1,-1).T), axis=1)





fig, axes = plt.subplots(1, 4, figsize=(10.6,2.4 ),  )



gmm = GaussianMixture(n_components=2, random_state=0 ).fit(aa_phi_psi)
labels_SS = gmm.predict(aa_phi_psi)





ax1=axes[0]
ax1.set_xlabel('Phi (φ)')
ax1.set_ylabel('Psi (ψ)')
sns.kdeplot(x=phi, y=psi, ax=ax1, alpha=0.5, legend=False )
sns.scatterplot(x=phi, y=psi, hue=labels_SS,
               palette=["red", "green"] ,ax= ax1, legend=False )


ax1.set_title('Ramachandran plot')





lnp_smote, labels_SS_smote = Kio,labels_SS
X=lnp_smote; y=labels_SS_smote

x_ = pd.DataFrame(X)
y_ = pd.DataFrame(y)
ff = pd.concat([x_, y_], axis=1)

scaler  = MinMaxScaler()
scaler2 = StandardScaler()
X = scaler2.fit_transform(X)



hparam = dict(learning_rate=0.2,
              max_depth=1,
              max_features='log2',
              n_estimators=500,
              random_state=32)

model_gb = GradientBoostingClassifier(**hparam, n_jobs=-1)



cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=32)

n_scores = cross_val_score(model_gb, X=X, y=y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance

#print(n_scores)
print('Mean Accuracy: %.2f (%.2f)' % (np.mean(n_scores), np.std(n_scores)))




def evaluate_model(data_x, data_y, model):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=32)
    predicted_targets = np.array([])
    actual_targets = np.array([])   
    predicted_targets_prob = np.array([])

    for train_ix, test_ix in cv.split(data_x, data_y):
        train_x, train_y, test_x, test_y = ((data_x)[train_ix], data_y[train_ix],
         data_x[test_ix], data_y[test_ix] )
        
                
        classifiers = model
        classifiers.fit(train_x, train_y)
        predicted_labels = classifiers.predict(test_x)
        predicted_prob = classifiers.predict_proba(test_x)[:,1]
        
        predicted_targets = np.append(predicted_targets, predicted_labels)
        predicted_targets_prob = np.append(predicted_targets_prob, predicted_prob)
        actual_targets = np.append(actual_targets, test_y)
          
    return predicted_targets, actual_targets, predicted_targets_prob



#########



cv = RepeatedStratifiedKFold(n_splits=5,
                             n_repeats=1,
                             random_state=32)

# sh = HalvingGridSearchCV(model_gb, param_grid, cv=cv,
#                          factor=2,resource='n_estimators',
#                          max_resources=500
#                         ).fit(X, y)
# print(sh.best_estimator_)
# print(sh.best_params_)
# print(sh.best_score_)
#print(sh.cv_results_)

grid = 'n'

if grid == 'y':
    
    param_grid = {'learning_rate': [ 0.1, 0.2, 0.3, 0.4,0.5,0.6, 0.7, 0.8, 1, 2, 3,], 
                  'max_depth': np.arange(1, 10) ,
                  'n_estimators' : [50, 100, 200, 400, 500,],
                  'min_samples_split' : [1, 2, 3, 4, 5],
                  'min_samples_leaf' : [1, 2, 3, 4, 5]           
                                
                  }
    
    grid = GridSearchCV(estimator=model_gb,
                        param_grid=param_grid,
                        cv=cv,
                        scoring='accuracy',
                        n_jobs=-1)
    
    grid_result = grid.fit(X, y)
    #summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    #means = grid_result.cv_results_['mean_test_score']
    #stds = grid_result.cv_results_['std_test_score']
    #params = grid_result.cv_results_['params']
    #for mean, stdev, param in zip(means, stds, params):
      #  print("%f (%f) with: %r" % (mean, stdev, param))





(predicted_targets,
actual_targets,
predicted_targets_prob) = evaluate_model(X, y, model_gb)


plt.grid('off')
cm=confusion_matrix(predicted_targets, actual_targets, normalize='true')
ax= axes[1]
ti = sns.heatmap(cm, cmap=plt.cm.Blues,annot=True, ax= ax)
ti.invert_yaxis()
accuracy = accuracy_score(predicted_targets, actual_targets)
#ax.set_title('Cross-validation accuracy: %.2f' % (accuracy))
ax.set_title('Confusion matrix')
#plt.colorbar()
ax.set_xlabel('True label')
ax.set_ylabel('Predicted label')
ax.grid('off')




ax= axes[2]
x_, y_ = calibration_curve(actual_targets,predicted_targets_prob, n_bins=20)


sns.regplot(x=x_, y=y_, ax=ax)

ax.set_title('Calibration curve')

ax.set_xlabel('Mean predicted probability')
ax.set_ylabel('Fraction of positives')
ax.set_ylim(0,1.03)
ax.set_xlim(0,1.03)




#plt.show()
ax= axes[3]

fpr, tpr, thresholds = roc_curve(actual_targets,predicted_targets_prob)

auc_ = roc_auc_score(actual_targets,predicted_targets_prob)

sns.lineplot(x=fpr, y=tpr, label='AUC = %0.2f' % auc_, color='red', linewidth=2, 
             ax=ax  )
sns.lineplot(x=[0, 1], y=[0, 1], color='navy', lw=1, linestyle='--',  )
plt.legend(loc=0, frameon=True, prop={'size':10 }
           ,handlelength=0, borderaxespad=0.25, handletextpad=0 )

ax.set_ylim(0,1.03)
ax.set_xlim(0,1.03)
ax.set_title('ROC curve')
ax.set_xlabel('FPR')
ax.set_ylabel('TPR')
# plot the roc curve for the model

print(auc_)
# show the plot
ax.grid(False)
plt.tight_layout()
plot('Res_fig.png')
plt.show()




clf= model_gb.fit(X, y)


print(clf.feature_importances_)

sns.barplot(x=["Residue", "K_int", "R_matrix", "K_obs"], 
            y= list(clf.feature_importances_),
             )
ax.tick_params(axis='x', rotation=20)
ax.set_ylabel('Importance')
ax.set_title("Feature Importances")
plt.tight_layout()
plot('Res_fig_import.png')
plt.show()


path = "/"
for i in range(500):        
    sub_tree_ = clf.estimators_[i, 0]        
    dot_data = export_graphviz(
          sub_tree_,        
    )    
    graph = graphviz.Source(dot_data, path +"%.d_GB" % i, format='png', ) 
    graph.view()
    
    
