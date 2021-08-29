import io
import numpy
import tensorflow

from tensorflow.keras.callbacks import LambdaCallback
from PIL import Image

def combineGenerator(gen1, gen2):
    while True:
	    yield(gen1.next(), gen2.next() )


def noiseAugmentor(image):
    return image + numpy.random.random()*numpy.random.random(image.shape)

	
