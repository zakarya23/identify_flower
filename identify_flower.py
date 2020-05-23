import sklearn 
import numpy as np 
import pandas as pd 
import copy 
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

columns = ["sepal length", "sepal width", "petal length", "petal width", "Class"]
dataset = pd.read_csv("iris_data.data", header = None, names = columns)

new_dataset = dataset.select_dtypes(include = ['object']).copy() 

class_count = new_dataset['Class'].value_counts() 
sb.set(style= "darkgrid")
sb.barplot(class_count.index, class_count.values,alpha = 0.9)

plt.title('Class Distribution')
plt.ylabel('Number')
plt.xlabel('Class')


replace_classes = {'Class': {'Iris-setosa': 1, 'Iris-versicolor': 2, 'Iris-virginica':3}}

num_dataset = dataset.copy() 
num_dataset.replace(replace_classes, inplace = True)

X = num_dataset[["sepal length", "sepal width", "petal length", "petal width"]]
Y = num_dataset['Class']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y , test_size = 0.2)

mlp = MLPClassifier(hidden_layer_sizes= 10, solver = 'sgd', learning_rate_init = 0.01, max_iter= 500) 

mlp.fit(X_train, Y_train)

MLPClassifier (activation = 'relu', alpha = 0.001, batch_size = 'auto', beta_1 = 0.9, beta_2 = 0.999, 
               early_stopping = False, epsilon = 1e-08, hidden_layer_sizes = 10, learning_rate = 'constant', 
               learning_rate_init = 0.01, max_iter = 500, momentum = 0.9, nesterovs_momentum = True, 
               power_t = 0.5, random_state = None, shuffle = True, solver = 'sgd', tol = 0.001, 
               validation_fraction = 0.1, verbose = False, warm_start = False)


print(mlp.score(X_test, Y_test))

def colour_stats(image): 
    (R, G, B) = image.split() 
    features = [np.mean(R), np.mean(G), np.mean(B), np.std(R), np.std(G), np.std(B)]

    return features