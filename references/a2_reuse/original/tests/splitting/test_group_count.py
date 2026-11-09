"""Migrated from test_split_dataset-4.py: GroupCountTests."""

from tests.support.splitting_fixtures import *

class GroupCountTests(unittest.TestCase):

    def test_documented_examples(self):
        expected = {3: (1, 1, 1), 4: (2, 1, 1), 5: (3, 1, 1), 8: (4, 2, 2), 13: (7, 3, 3), 25: (15, 5, 5)}
        for total, counts in expected.items():
            with self.subTest(total=total):
                self.assertEqual(load_split_module().group_counts(total), counts)

    def test_all_splits_are_nonempty_and_cover_all_groups(self):
        for total in range(3, 80):
            with self.subTest(total=total):
                train, validation, test = load_split_module().group_counts(total)
                self.assertEqual(train + validation + test, total)
                self.assertTrue(train >= 1 and validation >= 1 and (test >= 1))
                self.assertGreaterEqual(train, validation)

    def test_too_few_groups_are_rejected(self):
        for total in (0, 1, 2):
            with self.subTest(total=total), self.assertRaises(load_split_module().SplitError):
                load_split_module().group_counts(total)
