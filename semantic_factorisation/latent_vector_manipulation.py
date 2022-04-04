'''latent vector manipulation: 
    Take a vector that produces an image with frame
    using the pre trained binary classifier, move the point perpendicularly to the plane
    For now, experiment how the features in the image changes when going along the vector
    Is the quality of the image preserved?
    
    Tools: binary classifier
            latent vector representing an image with a frame
            '''

from joblib import load
import numpy as np
import PIL 
import PIL.Image
from sklearn.svm import SVC
import pickle
import torch

# make this better
# what is needed to run this function?
#  model_path, out_dir, bin_classifier_path, w_vector, 
    # model_path = "/SAM256/models/256_nopreprocessing/00001--cond-auto1-bgcfnc-resumecustom/network-snapshot-007200.pkl" # input
    # outdir = "/data/stylegan2-ada-pytorch/para_imgs_SAM/" # input


def latent_manipulator(model_path, outdir, w_vector_path, classifier_path, threshold, steps, alpha, diff_direction, eigenvector):
    
    with open(model_path, 'rb') as f:
        G = pickle.load(f)['G_ema'].cuda()

    # path to a vector that generates an image with frame: /data/generated_imgs_smaller/w_matrix_txt_files/seed0003.class.1.txt
    if diff_direction == False:    
        svm = load(classifier_path)# input
        b = svm.intercept_
        w = svm.coef_
    # w_vector = np.transpose( np.loadtxt('/data/generated_imgs_smaller/w_matrix_txt_files/seed0003.class.1.txt')[:, np.newaxis] ) 
    # w_vector = np.transpose( np.loadtxt('/data/stylegan2-ada-pytorch/seed0020.class.1.txt')[:, np.newaxis] ) # input
    w_vector = np.transpose( np.loadtxt(w_vector_path)[:, np.newaxis] ) # input

    # the normal of he hyperplane is in the direction of 1 label
    w_matrix = np.zeros((steps, 1, 14, 512)) # input shape: (steps, vector, layers, featurespace size)

    if diff_direction:
        w = eigenvector
        b=0
        d0 = np.linalg.norm(np.dot(w_vector, w.T) + b)/np.linalg.norm(w)
    # distance from point w (the image) to the plane
    # print("d0",d0)
    # d = np.linalg.norm(np.dot(w_matrix[0,0,0,:], w.T) + b)/np.linalg.norm(w)
    # print("d", d)
    # by parametrising, we get: 
    stop_idx = 0
    for i in range(steps):# input
        done = False
        for j in range(14): # number of layers since we feed all layers with the same stylevector
            w_matrix[i,0,j,:] = w_vector + i*alpha*w # only this part is relevant when moving along an eigenvactor (inspired from GANspace)

            if diff_direction == False:
                d = np.linalg.norm(np.dot(w_matrix[i,0,j,:], w.T) + b)/np.linalg.norm(w)
            # print("d",d)
                if threshold:
                    if d > 3*d0:# input
                        # print(w_matrix[i,0,j,:])
                        # print(d)
                        stop_idx = i
                        # print(stop_idx)
                        done = True
                    #     break
        if done:
            break

    # print(w_matrix.shape)
    # print(w_matrix[i,:][np.newaxis].shape)
    steps = steps - stop_idx
    for i in range(steps):# input
        img = G.synthesis(torch.tensor(w_matrix[i,:,:,:], device='cuda'), noise_mode='const', force_fp32=True)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/para_{i}.jpg')# input