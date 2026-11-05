import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from audio_book.cli import main
from audio_book.tts import RemoteEngine


class CLITests(unittest.TestCase):
    def test_remote_info_and_synthesis_work_without_credentials(self):
        info = io.BytesIO(json.dumps({'identity': 'test-model'}).encode())
        audio = io.BytesIO(b'audio-bytes')
        audio.headers = {'X-Voice-Identity': 'test-model'}
        with tempfile.TemporaryDirectory() as directory:
            with patch('audio_book.tts.urlopen', side_effect=[info, audio]) as request:
                engine = RemoteEngine('https://example.modal.run')
                target = Path(directory) / 'chunk.wav'
                engine.synthesize('Hello', target)
                self.assertEqual(target.read_bytes(), b'audio-bytes')
                self.assertEqual(len(request.call_args_list), 2)
                for call in request.call_args_list:
                    self.assertEqual(dict(call.args[0].header_items()),
                                     {'Content-type': 'application/json'})

    def test_missing_job_fails_before_contacting_model(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / 'missing-job')
            with patch('sys.argv', ['audiobook', 'render', missing, '--endpoint', 'https://example.modal.run']), patch('audio_book.cli.RemoteEngine') as remote, patch('sys.stderr', new_callable=io.StringIO) as stderr:
                self.assertEqual(main(), 1)
                remote.assert_not_called()
                self.assertIn('No prepared job', stderr.getvalue())

    def test_progress_is_visible_before_blocking_info_request(self):
        with patch('sys.stderr', new_callable=io.StringIO) as stderr:
            def interrupted_request(*args, **kwargs):
                self.assertIn('Connecting', stderr.getvalue())
                self.assertIn('/info', stderr.getvalue())
                raise TimeoutError('Simulated pending connection')
            with patch('audio_book.tts.urlopen', side_effect=interrupted_request):
                with self.assertRaises(TimeoutError):
                    RemoteEngine('https://example.modal.run')


if __name__ == '__main__':
    unittest.main()
