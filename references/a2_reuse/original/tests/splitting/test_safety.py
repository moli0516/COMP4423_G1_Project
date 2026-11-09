"""Migrated from test_split_dataset-4.py: SafetyTests."""

from tests.support.splitting_fixtures import *

class SafetyTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def assert_fails_without_changes(self, dataset, message, **kwargs):
        images_before = (dataset / 'images.csv').read_bytes()
        with self.assertRaisesRegex(load_split_module().SplitError, message):
            load_split_module().split_dataset(dataset, **kwargs)
        self.assertEqual((dataset / 'images.csv').read_bytes(), images_before)
        self.assertFalse((dataset / 'split_report.json').exists())

    def test_requires_three_groups_in_each_class(self):
        self.assert_fails_without_changes(make_dataset(self.temp.name, [3, 3, 3], [4, 4]), 'outside has 2 group')

    def test_missing_outside_class_is_rejected(self):
        self.assert_fails_without_changes(make_dataset(self.temp.name, [3, 3, 3], []), 'outside has 0 group')

    def test_unverified_groups_are_rejected_by_default_but_can_be_overridden(self):
        dataset = make_dataset(self.temp.name, [3, 3, 3], [4, 4, 4], verified='pending')
        self.assert_fails_without_changes(dataset, 'not boundary-verified')
        report = load_split_module().split_dataset(dataset, allow_unverified=True, trials=1)
        self.assertTrue(report['warnings'])

    def test_existing_split_is_frozen_unless_forced(self):
        dataset = make_dataset(self.temp.name, [3, 3, 3], [4, 4, 4], splits={'in_001': 'test'})
        self.assert_fails_without_changes(dataset, 'already contains split')
        load_split_module().split_dataset(dataset, force=True, trials=1, seed=1)
        self.assertTrue(all((row['split'] for row in read_images(dataset))))

    def test_invalid_existing_split_value_is_rejected_when_forcing(self):
        dataset = make_dataset(self.temp.name, [3, 3, 3], [4, 4, 4], splits={'in_001': 'dev'})
        with self.assertRaisesRegex(load_split_module().SplitError, 'existing split'):
            load_split_module().split_dataset(dataset, force=True)

    def test_group_must_have_matching_location_record(self):
        dataset = make_dataset(self.temp.name, [3, 3, 3], [4, 4, 4])
        rows = list(csv.reader((dataset / 'locations.csv').open(encoding='utf-8')))
        write_csv(dataset / 'locations.csv', rows[0], [row for row in rows[1:] if row[0] != 'in_002'])
        self.assert_fails_without_changes(dataset, 'in_002 is missing')

    def test_location_class_must_match_image_labels(self):
        dataset = make_dataset(self.temp.name, [3, 3, 3], [4, 4, 4])
        rows = list(csv.reader((dataset / 'locations.csv').open(encoding='utf-8')))
        for row in rows[1:]:
            if row[0] == 'in_001':
                row[1] = 'outside'
        write_csv(dataset / 'locations.csv', rows[0], rows[1:])
        self.assert_fails_without_changes(dataset, 'locations.csv says outside')

    def test_group_cannot_mix_labels(self):
        dataset = make_dataset(self.temp.name, [3, 3, 3], [4, 4, 4])
        rows = list(csv.reader((dataset / 'images.csv').open(encoding='utf-8')))
        rows[1][2] = '0'
        write_csv(dataset / 'images.csv', rows[0], rows[1:])
        self.assert_fails_without_changes(dataset, 'both inside and outside')

    def test_bad_labels_empty_fields_and_missing_columns_are_rejected(self):
        dataset = make_dataset(self.temp.name, [3, 3, 3], [4, 4, 4])
        rows = list(csv.reader((dataset / 'images.csv').open(encoding='utf-8')))
        rows[1][2] = '2'
        write_csv(dataset / 'images.csv', rows[0], rows[1:])
        self.assert_fails_without_changes(dataset, 'label must be 0 or 1')
        dataset = make_dataset(Path(self.temp.name) / 'empty', [3, 3, 3], [4, 4, 4])
        rows = list(csv.reader((dataset / 'images.csv').open(encoding='utf-8')))
        rows[1][3] = ''
        write_csv(dataset / 'images.csv', rows[0], rows[1:])
        self.assert_fails_without_changes(dataset, 'empty group_id')
        dataset = make_dataset(Path(self.temp.name) / 'columns', [3, 3, 3], [4, 4, 4])
        write_csv(dataset / 'images.csv', ['image_id', 'relative_path'], [['a', 'b']])
        with self.assertRaisesRegex(load_split_module().SplitError, 'missing columns'):
            load_split_module().split_dataset(dataset)

    def test_invalid_trial_count_is_rejected(self):
        dataset = make_dataset(self.temp.name, [3, 3, 3], [4, 4, 4])
        self.assert_fails_without_changes(dataset, 'at least 1', trials=0)
