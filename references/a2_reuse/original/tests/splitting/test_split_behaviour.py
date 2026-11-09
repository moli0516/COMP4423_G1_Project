"""Migrated from test_split_dataset-4.py: SplitBehaviourTests."""

from tests.support.splitting_fixtures import *

class SplitBehaviourTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)

    def dataset(self, *args, **kwargs):
        return make_dataset(self.temp.name, *args, **kwargs)

    def test_whole_groups_stay_together_and_every_split_has_both_classes(self):
        dataset = self.dataset([5, 8, 3, 9, 4, 7, 6], [4, 6, 8, 5, 7, 9])
        report = load_split_module().split_dataset(dataset, seed=7, trials=50)
        rows = read_images(dataset)
        by_group = {}
        for row in rows:
            by_group.setdefault(row['group_id'], set()).add(row['split'])
        self.assertTrue(all((len(values) == 1 and next(iter(values)) in load_split_module().SPLITS for values in by_group.values())))
        for split_name in load_split_module().SPLITS:
            labels = {row['label'] for row in rows if row['split'] == split_name}
            self.assertEqual(labels, {'0', '1'})
        self.assertEqual(report['group_counts_per_class']['inside'], {'train': 5, 'validation': 1, 'test': 1})
        self.assertEqual(report['group_counts_per_class']['outside'], {'train': 4, 'validation': 1, 'test': 1})

    def test_group_counts_are_stratified_by_class(self):
        dataset = self.dataset([3] * 13, [3] * 5)
        report = load_split_module().split_dataset(dataset, seed=1, trials=1)
        self.assertEqual(report['summary']['inside']['train']['groups'], 7)
        self.assertEqual(report['summary']['inside']['validation']['groups'], 3)
        self.assertEqual(report['summary']['inside']['test']['groups'], 3)
        self.assertEqual(report['summary']['outside']['train']['groups'], 3)

    def test_same_seed_is_reproducible_and_seed_is_recorded(self):
        first = self.dataset([5, 8, 3, 9, 4, 7], [4, 6, 8, 5, 7, 9])
        other = make_dataset(Path(self.temp.name) / 'copy', [5, 8, 3, 9, 4, 7], [4, 6, 8, 5, 7, 9])
        load_split_module().split_dataset(first, seed=123, trials=25)
        report = load_split_module().split_dataset(other, seed=123, trials=25)
        self.assertEqual(read_images(first), read_images(other))
        self.assertEqual(report['seed'], 123)

    def test_other_columns_row_order_and_extra_columns_are_preserved(self):
        dataset = self.dataset([2, 3, 4], [5, 2, 3])
        before = read_images(dataset)
        load_split_module().split_dataset(dataset, seed=3, trials=10)
        after = read_images(dataset)
        self.assertEqual(list(after[0]), IMAGE_HEADERS)
        for old, new in zip(before, after):
            old = dict(old)
            old['split'] = new['split']
            self.assertEqual(old, new)

    def test_report_matches_csv_and_documents_authority(self):
        dataset = self.dataset([2, 3, 4, 5], [5, 2, 3])
        load_split_module().split_dataset(dataset, seed=5, trials=20)
        report = json.loads((dataset / 'split_report.json').read_text(encoding='utf-8'))
        rows = read_images(dataset)
        for split_name, groups in report['groups'].items():
            self.assertEqual(sorted({row['group_id'] for row in rows if row['split'] == split_name}), groups)
        self.assertIn('authoritative', report['note'])

    def test_search_never_scores_worse_than_first_random_assignment(self):
        sizes = [1, 30, 2, 25, 3, 8, 12, 1, 18, 4]
        first = make_dataset(Path(self.temp.name) / 'one', sizes, sizes)
        many = make_dataset(Path(self.temp.name) / 'many', sizes, sizes)
        one_report = load_split_module().split_dataset(first, seed=11, trials=1)
        many_report = load_split_module().split_dataset(many, seed=11, trials=300)
        self.assertLessEqual(many_report['image_balance_score'], one_report['image_balance_score'])

    def test_search_uses_only_group_sizes_not_image_content(self):
        groups = {'inside': {'in_001': 5, 'in_002': 4, 'in_003': 3}, 'outside': {'out_001': 5, 'out_002': 4, 'out_003': 3}}
        first, _, _ = load_split_module().choose_assignment(groups, seed=9, trials=20)
        second, _, _ = load_split_module().choose_assignment(groups, seed=9, trials=20)
        self.assertEqual(first, second)

    def test_dry_run_changes_nothing(self):
        dataset = self.dataset([2, 3, 4], [5, 2, 3])
        before = (dataset / 'images.csv').read_bytes()
        load_split_module().split_dataset(dataset, dry_run=True)
        self.assertEqual((dataset / 'images.csv').read_bytes(), before)
        self.assertFalse((dataset / 'split_report.json').exists())

    def test_bom_and_crlf_inputs_are_accepted(self):
        dataset = self.dataset([2, 3, 4], [5, 2, 3])
        rows = read_images(dataset)
        write_csv(dataset / 'images.csv', IMAGE_HEADERS, [[row[name] for name in IMAGE_HEADERS] for row in rows], 'utf-8-sig', '\r\n')
        load_split_module().split_dataset(dataset, seed=2, trials=1)
        self.assertTrue(all((row['split'] for row in read_images(dataset))))
