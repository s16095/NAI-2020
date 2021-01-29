# -*- coding: utf-8 -*-
"""
Problem: Recognize type of clothes using neutral network
Dataset: https://github.com/zalandoresearch/fashion-mnist
Authors: Wociech Iracki s13066@pjwstk.edu.pl, Adrian Wojewoda 
Created following tutorial by user: https://www.kaggle.com/vesuvius13

"""

pip install visualkeras

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from sklearn.model_selection import train_test_split
import visualkeras
from tensorflow.keras.utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pydot
import graphviz
from PIL import Image

"""Importing training and test datasets


"""

test_df = pd.read_csv('/content/Data Sources/fashion-mnist_test.csv')
train_df = pd.read_csv('/content/Data Sources/fashion-mnist_train.csv')

train_df.head()

test_df.head()

"""Storing the labels in a dictionary


"""

classes = {0 : 'T-shirt/top',
1 :  'Trouser',
2 : 'Pullover',
3 : 'Dress',
4 : 'Coat',
5 : 'Sandal',
6 : 'Shirt',
7 : 'Sneaker',
8 : 'Bag',
9 : 'Ankle boot'}

"""Let's map these classes to the training and test dataframes to visualize the distribution of different labels


"""

train_df['label_names'] = train_df['label'].map(classes)

plt.figure(figsize=(10,6))
sns.countplot(x = 'label_names', data = train_df, facecolor=(0, 0, 0, 0),
                   linewidth=5,
                   edgecolor=sns.color_palette("dark", 3))
plt.xlabel('Class Names')
plt.ylabel('Count')
plt.title('Distribution of Training set images wrt classes')
plt.show()

test_df['label_names'] = test_df['label'].map(classes)

plt.figure(figsize=(10,6))
sns.countplot(x = 'label_names', data = test_df, facecolor=(0, 0, 0, 0),
                   linewidth=5,
                   edgecolor=sns.color_palette("dark", 3))
plt.xlabel('Class Names')
plt.ylabel('Count')
plt.title('Distribution of Test set images wrt classes')
plt.show()

"""Let's see the shape of training and test set.


"""

train_df.shape

test_df.shape

"""Let's prepare our data for preprocessing. First, we're going to divide training and test sets into features and labels.


"""

X_train = train_df.drop(['label', 'label_names'], axis = 1)
y_train = train_df.label

X_test = test_df.drop(['label', 'label_names'], axis = 1)
y_test = test_df.label

"""Now we're going to reshape our images


"""

X_train = X_train.values.reshape(X_train.shape[0], 28, 28)
X_test = X_test.values.reshape(X_test.shape[0], 28, 28)

"""Let's check the shapes again


"""

X_train.shape

X_test.shape

"""Let's visualize a single image from our training set


"""

single_image = X_train[1]
plt.imshow(single_image)

"""This looks like an ankle boot to me. Let's check the labels to confirm it"""

y_train

"""Index 1 has 9 as a value which is the key of Ankle boot as per classes dictionary, so we stand correct

OK, now back to data preprocessing. We're gonna convert our data from class vector to a binary matrix. So, we're gonna use Keras's to_categorical method to convert our labels into one-hot vectors.
"""

y_cat_train = to_categorical(y_train, num_classes= 10)
y_cat_test = to_categorical(y_test, num_classes= 10)

"""to_categorical takes number of classes on its own based on the label's unique values, here it was from 0 to 9, hence, it took 10. You can specify them too, like I did using num_classes

Let's check a single example from our training set, how it looks like
"""

y_cat_train[0]

"""Notice here, the first value of y_train was 2(check the cell below the ankle boot image), so to_categorical() has transformed that into an entire row. So, now at index no. 2, we have 1 and all the other values are 0.

Now we're going to normalize our images. Right now the pixel values ranges from 0 to 255, where 255 represents the lighest colour cell and 0 represents the darkest colour cell. We're gonna convert them into the values from 0 to 1.

I'm going to visualize that Ankle shoe image in the form of pixels now
"""

def plot_digit(digit, dem = 28, font_size = 12):
    max_ax = font_size * dem
    
    fig = plt.figure(figsize=(13, 13))
    plt.xlim([0, max_ax])
    plt.ylim([0, max_ax])
    plt.axis('off')
    black = '#000000'
    
    for idx in range(dem):
        for jdx in range(dem):

            t = plt.text(idx * font_size, max_ax - jdx*font_size, digit[jdx][idx], fontsize = font_size, color = black)
            c = digit[jdx][idx] / 255.
            t.set_bbox(dict(facecolor=(c, c, c), alpha = 0.5, edgecolor = 'black'))
            
    plt.show()

plot_digit(X_train[1])

"""We can normalize/scale the image by two methods, first one is using Scikit learn's MinMaxScaler function to scale the images and then fit_transform the training set and transform the test set and the other method is just dividing both training and test set features by 255.

We're gonna use the second method
"""

X_train = X_train/255
X_test = X_test/255

"""Now we're gonna add a colour channel. Since the images are grey scale, we'll add 1 colour channel


"""

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

X_train.shape

X_test.shape

"""Now we're done with our image preprocessing step, let's divide our data into training and validation set. We're gonna set validation set size to 20%


"""

X_train, X_val, y_cat_train, y_val = train_test_split(X_train, y_cat_train, test_size=0.2, random_state=42)

"""Now let's structure our Convolutional Neural Network model which we will train


"""

model = Sequential()

model.add(Conv2D(filters = 32, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(filters = 64, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate = 0.3))
model.add(BatchNormalization())
model.add(Conv2D(filters = 128, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(Conv2D(filters = 128, kernel_size= (4,4), padding = 'same', input_shape = (28,28,1), activation= 'relu'))
model.add(Dropout(rate = 0.3))
model.add(BatchNormalization())
model.add(MaxPool2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(rate = 0.3))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(rate = 0.3))
model.add(Dense(10, activation='softmax'))

opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)

model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])

"""filters:- how many filters to apply on an image

kernel_size:- size of the matrix which strides through the whole image

stride:- (x,y) steps while moving the kernel

padding:- Padding is the extra layer we add to the corner of the image to prevent shrinkage and loss of info, such as add a padding of 0 on the outside of the image matrix, so that the corner matrix is also covered more than once while striding.

flatten:- flattens our layer, eg, our image is 28x28 so the flattened image will be 28*28=784 pixels.

dropout:- It is a regularization technique which shuts off neurons randomly at each epoch to prevent overfitting.Here we've set rate to 0.3, so it means that 30% of neurons will be shut off randomly while training at each epoch.

batch normalization:- this technique makes neural networks faster and more stable through normalization of the input layer by re-centering and re-scaling.

Adam:- Adam is an optimizer used to change the attributes of your neural network such as weights and learning rate in order to reduce the losses.


"""

model.summary()

"""Visualizing the network


"""

visualkeras.layered_view(model)

"""Using Early Stopping to monitor the validation loss and stop the training when the loss begins to increase. Patience let's us set the number of epochs after which the training should be stopped in case of validation loss. We're setting it at 18."""

early_stop = EarlyStopping(monitor = 'val_loss', patience = 18)

model.fit(x = X_train, y = y_cat_train, batch_size = 128, epochs = 100, validation_data = (X_val, y_val), 
         callbacks = [early_stop])

metrics = pd.DataFrame(model.history.history)

metrics.head()

"""Plotting the loss and accuracy metrics


"""

metrics[['loss', 'val_loss']].plot()

metrics[['accuracy', 'val_accuracy']].plot()

"""Evaluating the model on the test set


"""

model.metrics_names

model.evaluate(X_test, y_cat_test, verbose = 0)

"""It has a **93.4%** accuracy on the test set which is not that bad. As you can see from the loss metric graph that the val_loss started to go up a little bit. To prevent that, you can apply some more regularizations in the neural network. You can reduce the patience level in Early stopping or use more dropout layers.

Let's visualize the true and predicted classes
"""

predictions = model.predict(X_test)
X_test_reshape = X_test.reshape(X_test.shape[0], 28, 28)

fig, axis = plt.subplots(4, 4, figsize=(12, 14))
for i, ax in enumerate(axis.flat):
    ax.imshow(X_test_reshape[i], cmap='binary')
    ax.set(title = f"Real Class is {classes[y_cat_test[i].argmax()]}\nPredict Class is {classes[predictions[i].argmax()]}");

predictions = model.predict(X_test)
# Convert predictions classes to one hot vectors 
predictions_classes = np.argmax(predictions, axis = 1)
# Convert test set observations to one hot vectors
y_true = np.argmax(y_cat_test, axis = 1)

print(classification_report(y_true, predictions_classes))

"""Precision:- Precision is the number of positive class predictions that actually belong to the positive class.

Recall:- Recall is the number of positive class predictions made out of all positive examples in the dataset.

F1-score:- It provides a single score that balances both the concerns of precision and recall in one number.
"""

print(confusion_matrix(y_true, predictions_classes))

plt.figure(figsize = (12, 8))
sns.heatmap(confusion_matrix(y_true, predictions_classes), annot=True)

"""Confusion Matrix:- It is a tabular summary of the number of correct and incorrect predictions made by a classifier.


"""