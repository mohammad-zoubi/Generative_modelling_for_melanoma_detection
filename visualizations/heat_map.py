import matplotlib.pyplot as plt
import seaborn as sns
import torch
import numpy as np
matrix_path = "/data/git_repo/SeFa_matrices/SAM_cond.pt"
# w_matrix = np.load("/data/w_matrix.npy")

eig_vec_matrix = torch.load(matrix_path)['eigvec']
ma = sns.color_palette("ch:s=-.2,r=.6", as_cmap=True)
ma =sns.diverging_palette(250, 30, l=65, center="dark", as_cmap=True)


fig = plt.figure(figsize=(12,12))
fig = sns.heatmap(eig_vec_matrix[:30,:30], cmap=ma)
fig = fig.get_figure() 
fig.savefig("/data/GITHUB_REPO/Generative_modelling/visualizations/plots/heatmap_eig_SAM.jpg") 

# print(np.trace(np.matmul(eig_vec_matrix, eig_vec_matrix.T)))
# mul = np.matmul(eig_vec_matrix, eig_vec_matrix.T)
# np.fill_diagonal(mul.numpy(),0)
# if (np.abs(mul) > 1e-5).any(): 
#     raise Exception("not orthogonal")

# np.savetxt("foo.tsv", w_matrix[:512,:], delimiter="\t")