import sklearn 
import numpy as np 
import argparse 
import os 
import pandas as pd 
import copy 
import seaborn as sb
import matplotlib.pyplot as plt

columns = ["sepal length", "sepal width", "petal length", "petal width", "class"]
data_set = pd.read_csv("iris_data.data", header = None, names = columns)

new_dataset = data_set.select_dtypes(include = ['object']).copy() 

class_count = new_dataset['class'].value_counts() 
sb.set(style= "darkgrid")
sb.barplot(class_count.index, class_count.values,alpha = 0.9)

plt.title('Class Distribution')
plt.ylabel('Number')
plt.xlabel('Class')
plt.show()


def colour_stats(image): 
    (R, G, B) = image.split() 
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]

    return features