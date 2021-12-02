import imageio
import os
import glob 
import pandas
import numpy

import matplotlib.pyplot as pyplot

from sklearn import preprocessing
from sklearn.mixture import GaussianMixture

def getrgb(filepath):
	imimage = imageio.imread(filepath, pilmode="RGB")
	imimage = imimage/255
	imimage = imimage.sum(axis=0).sum(axis=0)/(imimage.shape[0]*imimage.shape[1])
	imimage = imimage/numpy.linalg.norm(imimage, ord=None)
	return imimage


dataset = pandas.DataFrame()

for filepath in glob.glob("data/*"):
	image_features = pandas.DataFrame(getrgb(filepath))
	image_features = pandas.DataFrame.transpose(image_features)
	image_features ['path'] = filepath
	dataset = pandas.concat([dataset, image_features])

print(dataset)

data = dataset.iloc[:,0:3]
data = preprocessing.normalize(data)

gmm_machine = GaussianMixture(n_components = 4)
gmm_machine.fit(data)
gmm_results = gmm_machine.predict(data)
# print(gmm_results)

dataset['results'] = gmm_results

fig = pyplot.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(dataset[0],dataset[1],dataset[2], c=gmm_results)
ax.set_xlabel('R')
ax.set_ylabel('G')
ax.set_zlabel('B')

fig.savefig('scatterplot.png')

dataset = dataset.sort_values(by=['path'])
print(dataset)















