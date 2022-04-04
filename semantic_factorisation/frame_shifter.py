'''load pt file of the eigen vectors
    take one, any one eigenvector

    '''
import numpy as np
import torch
import os
import pickle
from latent_space_projector import vector_to_mat
from PIL import Image
import PIL

# Notes:
# idea: project w matrix to the new space and apply, binary classifier

model_path = "/ISIC256/models/00001-ISIC256_malignant-auto2-bgcfnc-resumecustom/network-snapshot-009400.pkl"
vectors_path = "/data/generated_imgs_frames/latent_code"
eigen_matrix_path = "/data/Master_Thesis/semantic_factorization/SeFa_matrices/mal_uncond_eig_vec.pt"
outdir = "/data/eigen_directed_imgs_fifth_direction_ISIC_MAL/imgs"
eigen_matrix = torch.load(eigen_matrix_path)['eigvec']

eigen_vector_index = 4
eigen_vector_index2 = 6
# w_vector_index = 11
w_vector_index = 242
eigen_direction = eigen_matrix[eigen_vector_index, :].numpy()
eigen_direction2 = eigen_matrix[eigen_vector_index2, :].numpy()
z_matrix = vector_to_mat(vectors_path, save=False)
z_vector = z_matrix[w_vector_index,:]
degree = 4
steps = 100
# directed_img_matrix = np.zeros((steps, 1, 14, 512)) # input shape: (steps, vector, layers, featurespace size)
directed_img_matrix = np.zeros((steps, 1, 512)) # input shape: (steps, vector, layers, featurespace size)

with open(model_path, 'rb') as f:
    G = pickle.load(f)['G_ema'].cuda()

for i in range(steps):# input
    # for j in range(14): # number of layers since we feed all layers with the same stylevector
        # if j < 3:
            # directed_img_matrix[i,0,j,:] = w_vector - i*degree*eigen_direction + i*6* eigen_matrix[eigen_vector_index2, :].numpy() # only this part is relevant when moving along an eigenvactor (inspired from GANspace)
            # directed_img_matrix[i,0,j,:] = w_vector + i*2* eigen_matrix[eigen_vector_index2, :].numpy() # only this part is relevant when moving along an eigenvactor (inspired from GANspace)
            # directed_img_matrix[i,0,j,:] = z_vector - i*eigen_matrix[eigen_vector_index, :].numpy() 
    directed_img_matrix[i,0,:] = z_vector + i*eigen_matrix[eigen_vector_index, :].numpy() 
        # else:
        #     directed_img_matrix[i,0,j,:] =  w_vector 

label = torch.zeros([1, G.c_dim], device='cuda')
# img = G.synthesis(torch.tensor(np.repeat(z_vector[np.newaxis, np.newaxis,:], 14,axis=1), device='cuda'), noise_mode='const', force_fp32=True)
img = G(torch.tensor(z_vector[np.newaxis, :], device='cuda'), label, noise_mode='const', force_fp32=True)
img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/original.jpg')# input

for i in range(steps):# input
        # img = G(torch.tensor(directed_img_matrix[i,:,:,:], device='cuda'), noise_mode='const', force_fp32=True)
        img = G(torch.tensor(directed_img_matrix[i,:,:], device='cuda'), label, noise_mode='const', force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/directed_{i}.jpg')# input