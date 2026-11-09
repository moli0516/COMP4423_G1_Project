"""Migrated from test_split_dataset-4.py: CommandLineTests."""

from tests.support.splitting_fixtures import *

class CommandLineTests(unittest.TestCase):

    def test_cli_success_dry_run_and_failure_codes(self):
        with tempfile.TemporaryDirectory() as temp:
            dataset = make_dataset(temp, [3, 4, 5, 6], [4, 5, 6, 7])
            dry = subprocess.run([sys.executable, str(SCRIPT), '--dataset-dir', str(dataset), '--dry-run', '--trials', '5'], capture_output=True, text=True)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            self.assertIn('Dry run only', dry.stdout)
            self.assertFalse((dataset / 'split_report.json').exists())
            real = subprocess.run([sys.executable, str(SCRIPT), '--dataset-dir', str(dataset), '--trials', '5'], capture_output=True, text=True)
            self.assertEqual(real.returncode, 0, real.stderr)
            self.assertIn('Split summary', real.stdout)
            self.assertTrue((dataset / 'split_report.json').exists())
            again = subprocess.run([sys.executable, str(SCRIPT), '--dataset-dir', str(dataset)], capture_output=True, text=True)
            self.assertEqual(again.returncode, 1)
            self.assertIn('already contains split', again.stderr)

    def test_main_prints_error_for_missing_dataset(self):
        stderr = io.StringIO()
        with tempfile.TemporaryDirectory() as temp, mock.patch('sys.argv', ['split_dataset.py', '--dataset-dir', temp]):
            with contextlib.redirect_stderr(stderr):
                code = load_split_module().main()
        self.assertEqual(code, 1)
        self.assertIn('Missing file', stderr.getvalue())
