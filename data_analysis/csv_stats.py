import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

df_gt = pd.read_excel('/Users/Mohammad/Desktop/derm_survey.xlsx').sort_values(by='order')
# print("ground truth labels \n", df_gt.head())

df_labels= pd.read_csv('/Users/Mohammad/Downloads/Synthetic Data.csv')
# print("predicted labels \n", df_labels.head())

# list_of_synth_question = [q for q in list(df_labels) if q[:23]=='Is this image synthetic']
# list_of_diagnosis_question = [q for q in list(df_labels) if q[:19]=='Diagnosis of lesion']
sam_answers_synth = df_labels.iloc[1][[q for q in list(df_labels) if q[:23]=='Is this image synthetic']]
sam_answers_diagnosis = df_labels.iloc[1][[q for q in list(df_labels) if q[:19]=='Diagnosis of lesion']]

sam_label_synth = np.asarray([1 if i == 'Yes' else 0 for i in sam_answers_synth])
sam_label_diagnosis = np.asarray([1 if i == 'malignant melanoma' else 0 for i in sam_answers_diagnosis])

synth_label_gt = np.asarray(df_gt.synthetic)
diagnosis_label_gt = np.asarray(df_gt.label)

tn_synth, fp_synth, fn_synth, tp_synth = confusion_matrix(y_true=synth_label_gt, y_pred=sam_label_synth).ravel()
# print("Confusion matrix for synthetic images: \n", confusion_matrix(y_true=synth_label_gt, y_pred=sam_label_synth))
print("tn_synth, fp_synth, fn_synth, tp_synth: ", tn_synth, fp_synth, fn_synth, tp_synth)
print("Synthetic images classification accuracy: : ", accuracy_score(y_true=synth_label_gt, y_pred=sam_label_synth))

tn_diagn, fp_diagn, fn_diagn, tp_diagn = confusion_matrix(y_true=diagnosis_label_gt, y_pred=sam_label_diagnosis).ravel()
# print("Confusion matrix for MEL diagnosis: \n", confusion_matrix(y_true=diagnosis_label_gt, y_pred=sam_label_diagnosis))
print("tn_diagn, fp_diagn, fn_diagn, tp_diagn:", tn_diagn, fp_diagn, fn_diagn, tp_diagn)
print("MEL classification accuracy: ", accuracy_score(y_true=diagnosis_label_gt, y_pred=sam_label_diagnosis))


