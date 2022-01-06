import inline as inline
import matplotlib
from numpy.core.defchararray import center
#
# This is a sample Notebook to demonstrate how to read "MNIST Dataset"
#
import numpy as np  # linear algebra
import struct
from array import array
from PIL import Image
from os.path import join
import matplotlib.pyplot as plt


#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)
    #


# Verify Reading Dataset via MnistDataloader class
#

inline
import random
import matplotlib.pyplot as plt

#
# Set file paths based on added MNIST Datasets
#

training_images_filepath =  'train-images.idx3-ubyte'
training_labels_filepath ='train-labels.idx1-ubyte'
test_images_filepath = 't10k-images.idx3-ubyte'
test_labels_filepath = 't10k-labels.idx1-ubyte'


#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15);
        index += 1


#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                   test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


##########(b)###########
def PCA(p, images):
    covariance = images @ images.transpose()
    covariance = 1 / len(images[0]) * covariance
    eigVals, vectors = np.linalg.eig(covariance)
    eigVals = eigVals.real
    vectors = vectors.real
    x = np.arange(0, len(eigVals))
    y = np.array(eigVals)
    plt.title("Eigen Values")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.plot(x, y)
    plt.show()
    Up = vectors[:, 0:p]
    returnedImages = np.array([Up.transpose() @ a for a in images.transpose()])
    return returnedImages, Up


###########(c)################
def KMeans(k, images, centers):
    imCenters = np.zeros(len(images))  ###################################-2
    clusters = np.empty([k], dtype=object)
    optimized = 0
    while (optimized == 0):
        prevClusters = clusters
        newClusters = np.empty([k], dtype=object)
        for i in range(k):
            newClusters[i] = list()
        for i in range(len(images)):
            dist = np.zeros(k)
            for j in range(k):
                dist[j] = ((images[i] - centers[j]) ** 2).sum()
            index = np.argmin(dist)
            newClusters[index].append(images[i])
            optimized = 1
            if index != imCenters[i]:
                imCenters[i] = index
                optimized = 0
        for i in range(k):
            length = len(newClusters[i])
            if length > 0:
                sum = 0
                for j in range(length):
                    sum = sum + newClusters[i][j]
                average = sum / length
                centers[i] = average
    return clusters, centers, imCenters


############(e)############
def assignClusters(centers, labels):
    clusters = np.empty([10], dtype=object)
    clVal = np.zeros(10)
    for i in range(10):
        clusters[i] = list()
    for i in range(len(centers)):
        val = int(labels[i])
        clusters[int(centers[i])].append(val)
    for i in range(10):
        count = np.zeros(10)
        for j in clusters[i]:
            count[j] = count[j] + 1
        temp = np.argmax(count)
        clVal[i] = temp
    return clusters, clVal


############(f)############
def checkSuccess(val, clusters):
    a, count = 0
    for i in range(10):
        for j in (clusters[i]):
            if j == val[i]:
                a = a + 1
            count = count + 1
    return 100 * a / count


##########(a)############
x_train = np.array(x_train)
x_test = np.array(x_test)
x_train = x_train / 255 - 0.5
x_test = x_test / 255 - 0.5

###########(b)#########
x_train = np.array([a.flatten() for a in x_train])
x_train = x_train.transpose()
x_test = np.array([b.flatten() for b in x_train])
x_test = x_train.transpose()
pics, Up = PCA(20, x_train)
# plot an example of a reconstructed image
i = random.randint(0, 27)
randomImage = x_train[:, i]
randomImage = randomImage.reshape(28, 28)
plt.imshow(randomImage, cmap='gray')
plt.show()
reconstructedImage = pics[i]
reconstructedImage = Up @ reconstructedImage
reconstructedImage = reconstructedImage.reshape(28, 28)
plt.imshow(reconstructedImage, cmap='gray')
plt.show()

#####(d)######
pics, Up = PCA(20, x_test)
centers = (np.random.random((10, 20)) - 0.5)
clusters, centers, returnedCenters = KMeans(10, pics, centers)

#########(e)##########
clusterDigits, label = assignClusters(returnedCenters, y_test)

#########(f)##########
percentage = checkSuccess(label, clusterDigits)
# print(percentage)

#########(g)##########
pics, Up = PCA(20, x_test)
for i in range(3):
    centers = (np.random.random((10, 20)) - 0.5)
    clusters, centers, returnedCenters = KMeans(10, pics, centers)
    clusterDigits, label = assignClusters(returnedCenters, y_train)
    percentage = checkSuccess(label, clusterDigits)
    # print(percentage)

#########(h)##########
pics, Up = PCA(12, x_test)
centers = (np.random.random((10, 20)) - 0.5)
clusters, centers, returnedCenters = KMeans(10, pics, centers)
clusterDigits, label = assignClusters(returnedCenters, y_train)
percentage = checkSuccess(label, clusterDigits)
# print(percentage)

#########(i)##########
pics, Up = PCA(20, x_test)
centers = np.zeros((10, 20))
for i in range(10):
    count = 0
    j = 0
    while count < 10:
        if i == y_train[j]:
            count = count + 1
            centers[i] = centers[i] + pics[j]
        j = j + 1
    centers[i] = center[i] / 10
clusters, centers, returnedCenters = KMeans(10, pics, centers)
clusterDigits, label = assignClusters(returnedCenters, y_train)
percentage = checkSuccess(label, clusterDigits)
# print(percentage)
