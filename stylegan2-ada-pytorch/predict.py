#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File       : server_mp.py
# Modified   : 22.01.2022
# By         : Sandra Carrasco <sandra.carrasco@ai.se>

from ast import arg
import numpy as np 
import re
import os
from typing import List
import matplotlib.pyplot as plt  
from pathlib import Path
from PIL import Image 
import torch
#import torchtoolbox.transform as transforms
from torchvision import transforms
from efficientnet_pytorch import EfficientNet
import seaborn as sb
from argparse import ArgumentParser 
from melanoma_classifier import test
from utils import load_model, load_isic_data, load_synthetic_data,  CustomDataset , confussion_matrix, make_df, load_isic_test, load_synth_images
import pandas as pd
from sklearn.model_selection import train_test_split
from datetime import date, datetime


testing_transforms = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(256),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
                                                            
def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''   
    # Process a PIL image for use in a PyTorch model
    
    pil_image = Image.open(image_path)
    
    # Resize
    if pil_image.size[0] > pil_image.size[1]:
        pil_image.thumbnail((5000, 256))
    else:
        pil_image.thumbnail((256, 5000))
        
    # Crop 
    left_margin = (pil_image.width-256)/2
    bottom_margin = (pil_image.height-256)/2
    right_margin = left_margin + 256
    top_margin = bottom_margin + 256
    
    pil_image = pil_image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Normalize
    np_image = np.array(pil_image)/255
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
  
    # PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array
    # Color channel needs to be first; retain the order of the other two dimensions.
    np_image = np_image.transpose((2, 0, 1))
    
    return np_image

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    if title is not None:
        ax.set_title(title)
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def predict(image_path, model, topk=1): #just 2 classes from 1 single output
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''  
    #image = process_image(image_path)
    
    # Convert image to PyTorch tensor first
    #image = torch.from_numpy(image).type(torch.cuda.FloatTensor)
    #print(image.shape)
    #print(type(image))
    
    # Returns a new tensor with a dimension of size one inserted at the specified position.
    #image = image.unsqueeze(0)
    
    output = model(testing_transforms(Image.open(image_path)).type(torch.cuda.FloatTensor).unsqueeze(0))  # same output
    
    probabilities = torch.sigmoid(output)
    
    # Probabilities and the indices of those probabilities corresponding to the classes
    top_probabilities, top_indices = probabilities.topk(topk)
    
    # Convert to lists
    top_probabilities = top_probabilities.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    top_indices = top_indices.detach().type(torch.FloatTensor).numpy().tolist()[0] 
    
    top_classes = []
    
    if probabilities > 0.5 :
        top_classes.append("Melanoma")
    else:
        top_classes.append("Benign")

    
    return top_probabilities, top_classes

def plot_diagnosis(predict_image_path, model,label):
    img_nb = predict_image_path.split('/')[-1].split('.')[0]
    probs, classes = predict(predict_image_path, model)   
    print(probs)
    print(classes)

    # Display an image along with the diagnosis of melanoma or benign
    # Plot Skin image input image
    plt.figure(figsize = (6,10))
    plot_1 = plt.subplot(2,1,1)

    image = process_image(predict_image_path)

    imshow(image, plot_1)
    font = {"color": 'g'} if 'Benign' in classes and label == 0 or 'Melanoma' in classes and label == 1 else {"color": 'r'}
    plot_1.set_title(f"Diagnosis: {classes}, Output (prob) {probs[0]:.4f}, Label: {label}", fontdict=font);
    plt.savefig(f'{args.out_path}/prediction_{img_nb}.png')




if __name__ == "__main__":
    parser = ArgumentParser() 
    parser.add_argument('--seeds', type=num_range, help='List of random seeds Ex. 0-3 or 0,1,2')
    parser.add_argument("--data_path", type=str, default='/workspace/generated-no-valset')
    parser.add_argument("--model_path", type=str, default='/workspace/stylegan2-ada-pytorch/CNN_trainings/melanoma_model_0_0.9225_16_12_train_reals+15melanoma.pth')
    parser.add_argument("--out_path", type=str, default='', help='output path for confussion matrix')
    parser.add_argument("--dataset_source", type=str, default='ISIC', help='SAM, ISIC or else')
    parser.add_argument("--label_path", type=str, help='write csv path')
    
    args = parser.parse_args()

    # Setting up GPU for processing or CPU if GPU isn't available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    
    if args.dataset_source == "SAM":
        input_images = [str(f) for f in sorted(Path(args.data_path).rglob('*jpeg')) if os.path.isfile(f)]
        y = [1 for i in range(len(input_images))]
        test_df = pd.DataFrame({'image_name': input_images, 'target': y})
    elif args.dataset_source == "ISIC":
        # For testing with ISIC dataset
        # _, test_df = load_isic_data(args.data_path)
        test_df = load_isic_test(args.data_path)
    else: 
        test_df = load_synth_images(args.data_path, None, None, only_mel=True)
    
    
    print('test_df',test_df)
    testing_dataset = CustomDataset(df = test_df, train = True, transforms = testing_transforms ) 
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=16, shuffle = False)                                                    
    test_pred, test_gt, test_accuracy = test(model, test_loader)
    # print(test_gt)
    confussion_matrix(test_gt, test_pred, test_accuracy, args.out_path)
    testing_df = make_df(args.data_path, "20,20")
    print(test_pred)
    test_df['predicted_target'] = np.around(test_pred)
    # test_df['target'] = torch.ceil(torch.tensor(test_pred))

    # test_pred = torch.ceil(torch.tensor(test_pred))
    test_df['predicted_labels'] = test_pred
    # testing_df['predicted_labels'] = test_pred
    print(test_df)
    test_df.to_csv(args.label_path)

    # Plot diagnosis 
    """ for seed_idx, seed in enumerate(args.seeds):
        print('Predicting image for seed %d (%d/%d) ...' % (seed, seed_idx, len(args.seeds)))
        path = '/home/Data/generated/seed' + str(seed).zfill(4) 
        path += '_0.png' if seed <= 5000 else '_1.png'
        plot_diagnosis(path, model) """