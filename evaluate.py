#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 18:26:34 2021

@author: Aaron
"""

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, accuracy_score, f1_score, confusion_matrix, precision_score, recall_score
import matplotlib.pylab as plt
from matplotlib import pyplot
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.metrics import plot_confusion_matrix

#%% Defined multiclass ROC plot function
def plot_multiclass_roc(preds, y_pred, X_test, y_test, n_classes, title, figsize=(12,4), flag=False, save=None):
       
    # colors = ['#E45C3A', '#F4A261', '#7880B5']
    colors = ['#E45C3A', '#7880B5', '#F4A261']
    plt.rcParams['font.size'] = '14'

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    # Compute ROC and AUROC for each class
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], preds[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_aspect(1)
    ax.grid('on')
    ax.minorticks_on()
    ax.grid(b=True, which='major', linestyle='-', linewidth=0.5, alpha=0.5, zorder=0)
    ax.grid(b=True, which='minor',  color='gray', linestyle='-', linewidth=0.25, alpha=0.25, zorder=0)
    ax.set_title(title, fontsize=16, fontweight='bold')
    titles = ['Object Level', 'Pixel Level', 'Structure Level']
    # titles = ['Damaged', 'Undamaged']
    for i in range(n_classes):
        # ax.plot(fpr[i], tpr[i], color=colors[i], label=f'Label {i}')
        ax.plot(fpr[i], tpr[i], color=colors[i], label=titles[i], linewidth=3)
    ax.legend(loc="lower right")
    
    # Plot confusion matrix
    np.set_printoptions(precision=2)
    ax2 = sns.heatmap(confusion_matrix(np.argmax(y_test,axis = 1), np.argmax(y_pred,axis = 1), normalize = 'true'), annot=True, 
                      cmap=plt.cm.Blues, vmin=0.0, vmax=1.0, annot_kws={'size':16})
    for _, spine in ax2.spines.items():
        spine.set_visible(True)
    ax2.set_xlabel('Predicted label')
    ax2.set_ylabel('True label')
    ax2.set_aspect(1)

    fig.tight_layout()

    if save: plt.savefig(save, dpi=300)
    plt.show()
    
#%%

preds = model.predict(X_test)

#%%
y_pred = np.round(preds)
print(accuracy_score(y_test, y_pred))
print(precision_score(y_test, y_pred, average = 'weighted'))
print(recall_score(y_test, y_pred, average = 'weighted'))
# print(balanced_accuracy_score(y_test, y_pred))
print(f1_score(np.argmax(y_test,axis = 1), np.argmax(y_pred,axis = 1), average='weighted'))
print(confusion_matrix(np.argmax(y_test,axis = 1), np.argmax(y_pred,axis = 1)))

# plot_multiclass_roc(preds, y_pred, X_test, y_test, 2, 'Task 2: Damage State', figsize=(9.5,5), flag=False, save='Task2.png')
plot_multiclass_roc(preds, y_pred, X_test, y_test, 3, 'Task 1: Scene Level', figsize=(9.5,5), flag=False, save='Task3.png')