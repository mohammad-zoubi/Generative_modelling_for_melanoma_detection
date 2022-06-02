from unittest import TestProgram
import numpy as np
import pandas as pd
import torch
import os
from utils import *
from melanoma_classifier import test
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import seaborn as sns
# Classifier paths
real_base_line_path = '/ISIC256/project_scripts/GITHUB_REPO/Generative_modelling_for_melanoma_detection/stylegan2-ada-pytorch/training_classifiers_events/test_all_melanoma/5_21/classifier_efficientnet-b2_0.9419_2022-05-21 19:42:58.888589_baseline_real_f.pth'
synth_path = '/ISIC256/project_scripts/GITHUB_REPO/Generative_modelling_for_melanoma_detection/stylegan2-ada-pytorch/training_classifiers_events/test_all_melanoma/5_20/classifier_efficientnet-b2_0.8988_2022-05-20 17:34:35.410224_baseline_f.pth'
no_bias_path = '/ISIC256/project_scripts/GITHUB_REPO/Generative_modelling_for_melanoma_detection/stylegan2-ada-pytorch/training_classifiers_events/test_all_melanoma/5_20/classifier_efficientnet-b2_0.8003_2022-05-20 16:17:55.213067_no_bias_nf.pth'
no_frame_path = '/ISIC256/project_scripts/GITHUB_REPO/Generative_modelling_for_melanoma_detection/stylegan2-ada-pytorch/training_classifiers_events/test_all_melanoma/5_20/classifier_efficientnet-b2_0.9014_2022-05-20 13:08:13.518058_no_frames_f.pth'
latent_path = '/ISIC256/project_scripts/GITHUB_REPO/Generative_modelling_for_melanoma_detection/stylegan2-ada-pytorch/training_classifiers_events/test_all_melanoma/5_21/classifier_efficientnet-b2_0.7755_2022-05-21 14:08:42.921919_latent_f.pth'
mixed_path = '/ISIC256/project_scripts/GITHUB_REPO/Generative_modelling_for_melanoma_detection/stylegan2-ada-pytorch/training_classifiers_events/test_all_melanoma/5_21/classifier_efficientnet-b2_0.9471_2022-05-21 20:44:19.096623_mixed_f.pth'
tmp = '/ISIC256/not_our_models/classifier_efficientnet-b2_conditional_train_reals+15melanoma.pth'
model_path = tmp

list_of_paths = [real_base_line_path, synth_path, no_bias_path, no_frame_path, latent_path, mixed_path]
# columns = [np.array(["real", "real", "synthetic", "synthetic", "unbiased", "unbiased", "no frames", "no frames", "edited", "edited", "mixed", "mixed"]), np.array(["fpr", "tpr", "fpr", "tpr", "fpr", "tpr","fpr", "tpr", "fpr", "tpr", "fpr", "tpr"])]
list_of_legends = np.array(["real", "synthetic", "unbiased", "no frames", "edited", "mixed"])
# Same test set for all 
# test_df = pd.read_csv('/ISIC256/test_loaders/test_1.csv')
test_df = pd.read_csv('/ISIC256/test_full.csv')
# test_df = pd.read_csv('/ISIC256/test_loaders/train_1.csv')
# test_df = pd.read_csv('/ISIC256/test_loaders/val_1.csv')
testing_dataset = CustomDataset(df = test_df, train = True, transforms = testing_transforms ) 
test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=16, num_workers=4, worker_init_fn=seed_worker, shuffle = False)



# sns.axes_style("white")
colors = ["#ddc7af","#d3a293", "#ba707e", "#995374", "#4f3159", "#231f37"]
# sns.set_palette("flare")
sns.set(font_scale=2)
sns.set_style("ticks")
sns.set_palette(colors)
# colors = ["#ddc7af"]
# sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
# plt.style.use("flare")
# roc_auc = roc_auc_score(fpr, tpr)
# print(test_pred, test_gt, test_accuracy, test_accuracy)
df = pd.DataFrame()


plt.figure(figsize=(16,12))

i=0
for pth in list_of_paths:
    print()
    model = load_model()
    model.eval()
    model.load_state_dict(torch.load(pth))
    print(list_of_legends[i])
    test_pred, test_gt, test_accuracy = test(model, test_loader) 
    tn, fp, fn, tp = confusion_matrix(test_gt, np.round(test_pred)).ravel()
    print("TP", tp)
    print("FP", fp)
    print("FN", fn)
    print("TN", tn)
    print("test_accuracy", test_accuracy)
    print("TPR", tp/(tp+fn))
    print("TNR", tn/(tn+fp))
    fpr, tpr, thresholds = roc_curve(test_gt, test_pred)


    # g = sns.lineplot(fpr, tpr)

    # plt.plot(
    #     fpr,
    #     tpr,
    #     # color=colors,
    #     lw=lw
    #     # label="ROC curve (area = %0.2f)" % roc_auc,
    # )
    plt.xlim([-0.01, 1.0])
    plt.ylim([-0.01, 1.05])
    ax = sns.lineplot(fpr, tpr, legend='full', label=list_of_legends[i], lw=4)
    i+=1
    
leg = plt.legend( loc="lower right" )
leg_lines = leg.get_lines()
plt.setp(leg_lines, linewidth=5)

# plt.legend(handlelength=1, handleheight=1)

plt.setp(ax.get_legend().get_texts(), fontsize='22')
# for legobj in leg.legendHandles:
    # legobj.set_linewidth(2.0)
# g.add_legend(handlelength = 10)
plt.xlabel("1 - Specificity (FPR)", fontsize=35)
plt.ylabel("Sensitivity (TPR)", fontsize=35)
plt.title("Classification ROC",fontsize=35)
plt.plot([-0.1, 1], [-0.1, 1], color="navy", linestyle="--")
plt.savefig("/ISIC256/tmp_fig.png")