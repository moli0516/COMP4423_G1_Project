"""Migrated from test_preprocessing-3.py: LoadImageTests."""

from tests.support.preprocessing_fixtures import *

class LoadImageTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'sample.png'
        left_right().save(self.path)
        self.data = self.path.read_bytes()

    def test_supported_sources_decode_to_identical_pixels(self):
        with open(self.path, 'rb') as handle:
            sources = {'str': str(self.path), 'path': self.path, 'bytes': self.data, 'bytearray': bytearray(self.data), 'bytesio': io.BytesIO(self.data), 'open_file': handle, 'pil': Image.open(self.path)}
            arrays = {name: np.asarray(preprocessing_public.load_image(source).convert('RGB')) for name, source in sources.items()}
        reference = arrays.pop('path')
        for name, array in arrays.items():
            with self.subTest(source=name):
                np.testing.assert_array_equal(array, reference)

    def test_stream_position_is_reset_before_decoding(self):
        stream = io.BytesIO(self.data)
        stream.read(25)
        self.assertEqual(preprocessing_public.load_image(stream).size, (40, 20))

    def test_pil_input_is_copied_not_mutated(self):
        original = left_right()
        before = np.asarray(original).copy()
        loaded = preprocessing_public.load_image(original)
        self.assertIsNot(loaded, original)
        loaded.paste(GREEN, (0, 0, 5, 5))
        np.testing.assert_array_equal(np.asarray(original), before)

    def test_bad_sources_raise_preprocessing_error(self):
        corrupt = self.root / 'corrupt.jpg'
        corrupt.write_bytes(b'not an image')
        text_as_image = self.root / 'notes.png'
        text_as_image.write_text('hello', encoding='utf-8')
        bad_sources = [corrupt, text_as_image, self.root / 'missing.png', self.root, b'', b'\x00' * 20, 123, None, ['a']]
        for source in bad_sources:
            with self.subTest(source=repr(source)[:40]), self.assertRaises(preprocessing_public.PreprocessingError):
                preprocessing_public.load_image(source)

    def test_preprocessing_error_is_a_value_error(self):
        self.assertTrue(issubclass(preprocessing_public.PreprocessingError, ValueError))
