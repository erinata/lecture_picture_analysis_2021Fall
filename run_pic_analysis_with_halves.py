import imageio
import os
import glob 
import pandas
import numpy
import math

import matplotlib.pyplot as pyplot

from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

def getrgb(filepath):
	imimage = imageio.imread(filepath, pilmode="RGB")
	imimage = imimage/255
	imimage_top = imimage[0:math.ceil(imimage.shape[0]/2),:]
	imimage_bottom = imimage[math.ceil(imimage.shape[0]/2):imimage.shape[0],:]

	imimage_top = imimage_top.sum(axis=0).sum(axis=0)/(imimage_top.shape[0]*imimage_top.shape[1])
	imimage_top = imimage_top/numpy.linalg.norm(imimage_top, ord=None)

	imimage_bottom = imimage_bottom.sum(axis=0).sum(axis=0)/(imimage_bottom.shape[0]*imimage_bottom.shape[1])
	imimage_bottom = imimage_bottom/numpy.linalg.norm(imimage_bottom, ord=None)

	imimage = numpy.concatenate((imimage_top, imimage_bottom))
	return imimage


dataset = pandas.DataFrame()

for filepath in glob.glob("data/*"):
	image_features = pandas.DataFrame(getrgb(filepath))
	image_features = pandas.DataFrame.transpose(image_features)
	image_features ['path'] = filepath
	dataset = pandas.concat([dataset, image_features])

print(dataset)

data = dataset.iloc[:,0:6]
print(data)
data = preprocessing.normalize(data)

gmm_machine = GaussianMixture(n_components = 5)
# gmm_machine = KMeans(n_clusters = 5)
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















