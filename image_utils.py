import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
import os, shutil
import tarfile
from os.path import exists, join, basename
from os import makedirs, remove
from six.moves import urllib


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def saveImg(img, outFileName):
	img = img / 2 + 0.5
	npimg = img.numpy()
	scipy.misc.imsave(outFileName, np.transpose(npimg, (1, 2, 0)))

def dl_dataset(dest="dataset"):
    output_image_dir = join(dest, "BSDS300/images")

    if not exists(output_image_dir):
        makedirs(dest)
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("downloading url ", url)

        data = urllib.request.urlopen(url)

        file_path = join(dest, basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, dest)
        source_train = join(output_image_dir, "train")
        source_test = join(output_image_dir, "test")

        dest_train = join(source_train,"train")
        dest_test = join(source_test,"test")

        source_train = source_train + "/"
        source_test = source_test + "/"

        makedirs(dest_train)
        makedirs(dest_test)

        files_train = os.listdir(source_train)
        files_test = os.listdir(source_test)

        for img in files_train:
        	shutil.move(source_train+img, dest_train)

        for img in files_test:
        	shutil.move(source_test+img, dest_test)

        remove(file_path)

    return output_image_dir