"""Migrated from test_preprocessing-3.py: ConfigTests."""

from tests.support.preprocessing_fixtures import *

class ConfigTests(unittest.TestCase):

    def test_defaults_are_documented_values(self):
        config = PreprocessingConfig()
        self.assertEqual(config.to_dict(), {'image_size': 256, 'resize_mode': 'squash', 'pad_value': 0})
        self.assertEqual(preprocessing_public.DEFAULT_CONFIG, config)

    def test_valid_settings_are_accepted(self):
        for mode in preprocessing_public.RESIZE_MODES:
            with self.subTest(mode=mode):
                self.assertEqual(PreprocessingConfig(16, mode, 255).resize_mode, mode)

    def test_invalid_settings_are_rejected(self):
        invalid = [{'image_size': 15}, {'image_size': 0}, {'image_size': -1}, {'image_size': 64.0}, {'image_size': '64'}, {'image_size': None}, {'resize_mode': 'stretch'}, {'resize_mode': ''}, {'resize_mode': None}, {'pad_value': -1}, {'pad_value': 256}, {'pad_value': 1.5}, {'pad_value': '0'}]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                PreprocessingConfig(**values)

    def test_config_is_immutable_and_hashable(self):
        config = PreprocessingConfig(64)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.image_size = 128
        self.assertEqual(hash(config), hash(PreprocessingConfig(64)))

    def test_dict_round_trip_and_defaults(self):
        config = PreprocessingConfig(96, 'pad', 17)
        self.assertEqual(PreprocessingConfig.from_dict(config.to_dict()), config)
        self.assertEqual(PreprocessingConfig.from_dict({'image_size': 32}), PreprocessingConfig(32))

    def test_from_dict_rejects_unknown_keys(self):
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            PreprocessingConfig.from_dict({'image_size': 64, 'colour_space': 'hsv'})

    def test_to_dict_returns_independent_copy(self):
        config = PreprocessingConfig()
        values = config.to_dict()
        values['image_size'] = 999
        self.assertEqual(config.image_size, 256)
