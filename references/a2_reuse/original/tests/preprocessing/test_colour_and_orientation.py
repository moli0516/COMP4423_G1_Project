"""Migrated from test_preprocessing-3.py: ColourAndOrientationTests."""

from tests.support.preprocessing_fixtures import *

class ColourAndOrientationTests(unittest.TestCase):

    def test_every_supported_mode_becomes_rgb(self):
        images = {'L': Image.new('L', (20, 20), 128), '1': Image.new('1', (20, 20), 1), 'P': Image.new('P', (20, 20), 3), 'CMYK': Image.new('CMYK', (20, 20), (0, 0, 0, 0)), 'RGBA': Image.new('RGBA', (20, 20), (10, 20, 30, 255)), 'LA': Image.new('LA', (20, 20), (100, 255)), 'RGB': Image.new('RGB', (20, 20), (1, 2, 3))}
        for mode, image in images.items():
            with self.subTest(mode=mode):
                converted = preprocessing_public.normalise_orientation_and_colour(image, 0)
                self.assertEqual(converted.mode, 'RGB')
                self.assertEqual(converted.size, (20, 20))

    def test_cmyk_and_bilevel_conversion_values(self):
        white_cmyk = preprocessing_public.normalise_orientation_and_colour(Image.new('CMYK', (4, 4), (0, 0, 0, 0)), 0)
        white_bilevel = preprocessing_public.normalise_orientation_and_colour(Image.new('1', (4, 4), 1), 0)
        self.assertEqual(white_cmyk.getpixel((0, 0)), (255, 255, 255))
        self.assertEqual(white_bilevel.getpixel((0, 0)), (255, 255, 255))

    def test_opaque_rgba_colours_are_unchanged(self):
        result = preprocessing_public.normalise_orientation_and_colour(Image.new('RGBA', (5, 5), (10, 200, 30, 255)), 255)
        self.assertEqual(result.getpixel((2, 2)), (10, 200, 30))

    def test_transparent_pixels_use_pad_value_background(self):
        for pad_value in (0, 77, 255):
            transparent = Image.new('RGBA', (6, 6), (255, 0, 0, 0))
            la = Image.new('LA', (6, 6), (200, 0))
            palette = Image.new('P', (6, 6), 0)
            palette.info['transparency'] = 0
            for name, image in {'RGBA': transparent, 'LA': la, 'P': palette}.items():
                with self.subTest(pad_value=pad_value, mode=name):
                    result = preprocessing_public.normalise_orientation_and_colour(image, pad_value)
                    self.assertEqual(result.getpixel((3, 3)), (pad_value,) * 3)

    def test_partial_alpha_is_blended_with_background(self):
        result = preprocessing_public.normalise_orientation_and_colour(Image.new('RGBA', (4, 4), (255, 0, 0, 128)), 0)
        red, green, blue = result.getpixel((1, 1))
        self.assertTrue(126 <= red <= 130)
        self.assertEqual((green, blue), (0, 0))

    def test_exif_orientation_is_applied(self):
        expected = {1: ((40, 20), RED, BLUE, 'horizontal'), 3: ((40, 20), BLUE, RED, 'horizontal'), 6: ((20, 40), RED, BLUE, 'vertical'), 8: ((20, 40), BLUE, RED, 'vertical')}
        with tempfile.TemporaryDirectory() as temp:
            for orientation, (size, first, second, axis) in expected.items():
                with self.subTest(orientation=orientation):
                    path = Path(temp) / f'orientation_{orientation}.jpg'
                    save_jpeg_with_orientation(path, left_right(), orientation)
                    result = preprocessing_public.normalise_orientation_and_colour(preprocessing_public.load_image(path), 0)
                    self.assertEqual(result.size, size)
                    start, end = ((2, 10), (37, 10)) if axis == 'horizontal' else ((10, 2), (10, 37))
                    self.assertGreater(result.getpixel(start)[np.argmax(first)], 200)
                    self.assertGreater(result.getpixel(end)[np.argmax(second)], 200)

    def test_orientation_changes_final_array_for_pad_mode(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'portrait.jpg'
            save_jpeg_with_orientation(path, solid((20, 40), GREEN), 6)
            result = preprocessing_api.load_and_preprocess(path, PreprocessingConfig(64, 'pad', 0))
        np.testing.assert_array_equal(result[5, 32], [0, 0, 0])
        self.assertGreater(result[32, 5, 1], 200)
