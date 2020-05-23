import sklearn 
import PIL 
import numpy as np 
import argparse 
import os 


def colour_stats(image): 
    (R, G, B) = image.split() 
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]

    return features