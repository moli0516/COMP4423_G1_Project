"""Migrated from test_preprocessing-3.py: CommandLineTests."""

from tests.support.preprocessing_fixtures import *

class CommandLineTests(unittest.TestCase):

    def run_main(self, arguments):
        stdout, stderr = (io.StringIO(), io.StringIO())
        with mock.patch('sys.argv', ['preprocessing.py', *arguments]):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                try:
                    code = preprocessing_public.main()
                except SystemExit as error:
                    code = error.code
        return (code, stdout.getvalue(), stderr.getvalue())

    def test_success_reports_shape_and_returns_zero(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'scene.png'
            left_right().save(path)
            code, out, _ = self.run_main([str(path), '--image-size', '32', '--resize-mode', 'pad', '--pad-value', '3'])
        self.assertEqual(code, 0)
        self.assertIn('shape=(32, 32, 3)', out)
        self.assertIn("'resize_mode': 'pad'", out)

    def test_failure_returns_one_and_continues_to_report_other_files(self):
        with tempfile.TemporaryDirectory() as temp:
            good = Path(temp) / 'good.png'
            bad = Path(temp) / 'bad.jpg'
            left_right().save(good)
            bad.write_bytes(b'bad')
            code, out, _ = self.run_main([str(bad), str(good)])
        self.assertEqual(code, 1)
        self.assertIn('ERROR', out)
        self.assertIn('OK', out)

    def test_invalid_mode_is_rejected_by_argument_parser(self):
        code, _, err = self.run_main(['image.png', '--resize-mode', 'stretch'])
        self.assertEqual(code, 2)
        self.assertIn('invalid choice', err)
