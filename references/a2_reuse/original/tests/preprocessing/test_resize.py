"""Migrated from test_preprocessing-3.py: ResizeTests."""

from tests.support.preprocessing_fixtures import *

class ResizeTests(unittest.TestCase):

    def test_shape_dtype_and_memory_layout_for_difficult_sizes(self):
        sizes = [(1, 1), (7, 300), (300, 7), (64, 64), (500, 300), (1000, 20), (20, 1000)]
        for output_size in (16, 64):
            for mode in preprocessing_public.RESIZE_MODES:
                for size in sizes:
                    with self.subTest(output_size=output_size, mode=mode, source_size=size):
                        result = preprocessing_api.preprocess_image(solid(size, (120, 130, 140)), PreprocessingConfig(output_size, mode))
                        self.assertEqual(result.shape, (output_size, output_size, 3))
                        self.assertEqual(result.dtype, np.uint8)
                        self.assertTrue(result.flags['C_CONTIGUOUS'])

    def test_square_images_are_identical_in_all_modes(self):
        rng = np.random.default_rng(1)
        image = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
        results = [preprocessing_api.preprocess_image(image, PreprocessingConfig(64, mode)) for mode in preprocessing_public.RESIZE_MODES]
        np.testing.assert_array_equal(results[0], np.asarray(image))
        np.testing.assert_array_equal(results[0], results[1])
        np.testing.assert_array_equal(results[0], results[2])

    def test_squash_keeps_whole_scene_and_changes_aspect_ratio(self):
        result = preprocessing_api.preprocess_image(left_right((200, 100)), PreprocessingConfig(64, 'squash'))
        self.assertGreater(result[32, 0, 0], 200)
        self.assertGreater(result[32, -1, 2], 200)

    def test_center_crop_removes_outer_edges_without_distortion(self):
        image = solid((200, 100), RED)
        image.paste(GREEN, (0, 0, 20, 100))
        image.paste(GREEN, (180, 0, 200, 100))
        squash = preprocessing_api.preprocess_image(image, PreprocessingConfig(64, 'squash'))
        crop = preprocessing_api.preprocess_image(image, PreprocessingConfig(64, 'center_crop'))
        self.assertGreater(squash[:, :4, 1].mean(), 200)
        self.assertLess(crop[:, :, 1].max(), 20)
        self.assertGreater(crop[:, :, 0].min(), 235)

    def test_center_crop_vertical_images_keep_vertical_center(self):
        image = solid((100, 300), RED)
        image.paste(BLUE, (0, 100, 100, 200))
        result = preprocessing_api.preprocess_image(image, PreprocessingConfig(64, 'center_crop'))
        self.assertGreater(result[:, :, 2].min(), 235)
        self.assertLess(result[:, :, 0].max(), 20)

    def test_pad_preserves_aspect_ratio_and_uses_pad_value(self):
        image = solid((200, 100), (200, 200, 200))
        result = preprocessing_api.preprocess_image(image, PreprocessingConfig(64, 'pad', 7))
        np.testing.assert_array_equal(result[:16], np.full((16, 64, 3), 7, dtype=np.uint8))
        np.testing.assert_array_equal(result[48:], np.full((16, 64, 3), 7, dtype=np.uint8))
        self.assertTrue(np.all(result[18:46] > 150))

    def test_pad_centres_tall_images(self):
        result = preprocessing_api.preprocess_image(solid((100, 200), (220, 220, 220)), PreprocessingConfig(64, 'pad', 9))
        np.testing.assert_array_equal(result[:, :16], np.full((64, 16, 3), 9, dtype=np.uint8))
        np.testing.assert_array_equal(result[:, 48:], np.full((64, 16, 3), 9, dtype=np.uint8))

    def test_modes_produce_different_results_for_non_square_input(self):
        image = left_right((200, 100))
        squash, crop, pad = [preprocessing_api.preprocess_image(image, PreprocessingConfig(64, mode)) for mode in preprocessing_public.RESIZE_MODES]
        self.assertFalse(np.array_equal(squash, crop))
        self.assertFalse(np.array_equal(squash, pad))
        self.assertFalse(np.array_equal(crop, pad))

    def test_resize_is_deterministic_and_does_not_mutate_input(self):
        rng = np.random.default_rng(2)
        image = Image.fromarray(rng.integers(0, 256, (80, 120, 3), dtype=np.uint8))
        before = np.asarray(image).copy()
        for mode in preprocessing_public.RESIZE_MODES:
            with self.subTest(mode=mode):
                first = preprocessing_api.preprocess_image(image, PreprocessingConfig(64, mode))
                second = preprocessing_api.preprocess_image(image, PreprocessingConfig(64, mode))
                np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(np.asarray(image), before)

    def test_returned_array_is_independent_of_source_image(self):
        image = solid((20, 20), RED)
        result = preprocessing_api.preprocess_image(image, PreprocessingConfig(16))
        image.paste(BLUE, (0, 0, 20, 20))
        self.assertGreater(result[:, :, 0].min(), 235)

    def test_default_configuration_produces_documented_shape(self):
        self.assertEqual(preprocessing_api.preprocess_image(solid((32, 24), RED)).shape, (256, 256, 3))
