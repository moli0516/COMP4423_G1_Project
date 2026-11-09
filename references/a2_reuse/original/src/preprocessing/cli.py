import argparse

from .image_preprocessor import ImagePreprocessor
from .preprocessing_config import PreprocessingConfig, RESIZE_MODES
from src.exceptions.preprocessing_error import PreprocessingError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Check image preprocessing.")
    parser.add_argument("images", nargs="+")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--resize-mode", choices=RESIZE_MODES, default="squash")
    parser.add_argument("--pad-value", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        config = PreprocessingConfig(args.image_size, args.resize_mode, args.pad_value)
    except ValueError as error:
        parser.error(str(error))
    print(config.to_dict())
    processor = ImagePreprocessor(config)
    code = 0
    for source in args.images:
        try:
            image = processor.load(source)
            print(f"OK {source}: shape={image.shape}, dtype={image.dtype}")
        except PreprocessingError as error:
            print(f"ERROR {source}: {error}")
            code = 1
    return code
