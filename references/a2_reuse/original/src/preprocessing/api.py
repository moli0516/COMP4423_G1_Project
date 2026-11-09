from .image_preprocessor import ImagePreprocessor
from .preprocessing_config import PreprocessingConfig


def load_image(source):
    return ImagePreprocessor().load_image(source)


def normalise_orientation_and_colour(image, pad_value=0):
    return ImagePreprocessor(PreprocessingConfig(pad_value=pad_value)).normalise_orientation_and_colour(image)


def preprocess_image(image, config=None):
    return ImagePreprocessor(config).transform(image)


def load_and_preprocess(source, config=None):
    return ImagePreprocessor(config).load(source)
