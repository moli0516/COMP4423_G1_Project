"""Migrated from test_preprocessing-3.py: SourceAndMetadataIndependenceTests."""

from tests.support.preprocessing_fixtures import *

class SourceAndMetadataIndependenceTests(unittest.TestCase):

    def test_equivalent_sources_give_identical_arrays(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'scene.png'
            left_right((90, 60)).save(path)
            data = path.read_bytes()
            config = PreprocessingConfig(48, 'pad', 5)
            reference = preprocessing_api.load_and_preprocess(path, config)
            for name, source in {'str': str(path), 'bytes': data, 'stream': io.BytesIO(data), 'pil': Image.open(path)}.items():
                with self.subTest(source=name):
                    np.testing.assert_array_equal(preprocessing_api.load_and_preprocess(source, config), reference)

    def test_path_names_and_parent_folders_do_not_change_pixels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first = root / 'inside' / 'in_001' / 'campus.png'
            second = root / 'outside' / 'out_999' / 'different_name.png'
            first.parent.mkdir(parents=True)
            second.parent.mkdir(parents=True)
            left_right().save(first)
            shutil_copy = first.read_bytes()
            second.write_bytes(shutil_copy)
            np.testing.assert_array_equal(preprocessing_api.load_and_preprocess(first), preprocessing_api.load_and_preprocess(second))

    def test_non_orientation_exif_metadata_is_ignored(self):
        with tempfile.TemporaryDirectory() as temp:
            plain = Path(temp) / 'plain.jpg'
            tagged = Path(temp) / 'tagged.jpg'
            image = left_right((64, 48))
            image.save(plain, format='JPEG', quality=95)
            exif = image.getexif()
            exif[270] = 'location text that must not matter'
            exif[305] = 'test camera software'
            image.save(tagged, format='JPEG', quality=95, exif=exif)
            np.testing.assert_array_equal(preprocessing_api.load_and_preprocess(plain), preprocessing_api.load_and_preprocess(tagged))

    def test_preprocess_image_matches_load_and_preprocess(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'scene.png'
            left_right((90, 60)).save(path)
            config = PreprocessingConfig(32, 'center_crop')
            np.testing.assert_array_equal(preprocessing_api.preprocess_image(Image.open(path), config), preprocessing_api.load_and_preprocess(path, config))
