import numpy as np
import matplotlib.pyplot as plt
import skimage.io
import os
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network import MLPClassifier

IMAGE_REDUCED_SIZE = 64
FOLDER = '../dataset/'

#For example purpose only
#example = example[:, :, 0] #keep only R component (from RGB)
#showImage(example)

def showImage(image):
    print(image.shape)
    fig = plt.figure()
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    filenames = []
    classes = []
    for filename in os.listdir(FOLDER):
        if filename.endswith('.jpg'):
            filenames.append(FOLDER + filename)
            classes.append(filename.split('_')[0])

    dataInputs = []
    for i in range(30):
        filename = filenames[i]
        example = skimage.io.imread(filename)
        example = skimage.transform.resize(example, (IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE))

        showImage(example)

        line = example.reshape(1, -1)
        dataInputs.append(line[0])
        print(line.shape)
        print(line)

        example = line.reshape(IMAGE_REDUCED_SIZE, IMAGE_REDUCED_SIZE, -1)
        showImage(example)

    dataInputs = np.array(dataInputs)
    print(dataInputs)
