import torch
from torchvision import datasets, transforms
import numpy as np
from PIL import Image
from random import sample, shuffle


def print_image(x):
    pilImage = Image.fromarray(x.reshape(img_shape))
    pilImage.show()


dataset1 = datasets.MNIST('../data', train=True, download=True)
dataset2 = datasets.MNIST('../data', train=False)

img_shape = np.asarray(dataset1[0][0]).shape
imgs = {}

for i in range(10):
    imgs[i] = [] 


for i in dataset1:
    label = i[1]
    imgs[label].append(np.asarray(i[0]).flatten())

pos_lbl = 0
neg_lbl = 7

w = np.zeros((img_shape[0] * img_shape[1]))
#print(sample(imgs[pos_lbl], 1))

samples = []

k = 10
epochs = 2

for i in range(k):
    samples.append((np.array(sample(imgs[pos_lbl],1)), pos_lbl))
    samples.append((np.array(sample(imgs[neg_lbl],1)), neg_lbl))

shuffle(samples)


for j in range(epochs):
    m=0
    for i in range(2*k):
        s = samples[i]
        if np.dot(w, s[0].reshape(w.shape)) < 0:
            m += 1
        if s[1] == pos_lbl:
            w += s[0].reshape(w.shape)
        else:
            w -= s[0].reshape(w.shape)
        print_image(w)
    print('Mistakes made: ', m)

#print(imgs[0][0].reshape(img_shape))
#print_image(imgs[0][0])

