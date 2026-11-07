import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import wave

from fastapi.testclient import TestClient

from audio_book.server import create_app
from audio_book.tts import RemoteEngine


class CountingEngine:
    identity = 'kokoro:test-settings-v1'

    def __init__(self):
        self.calls = 0

    def synthesize(self, text, target):
        self.calls += 1
        with wave.open(str(target), 'wb') as stream:
            stream.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            stream.writeframes(b'\x00\x00' * 100)


class ServerTests(unittest.TestCase):
    def test_failed_volume_commit_is_retried_before_returning_cached_audio(self):
        engine = CountingEngine()
        commits = []

        def commit():
            commits.append(True)
            if len(commits) == 1:
                raise RuntimeError('Simulated Volume commit failure')

        with tempfile.TemporaryDirectory() as directory:
            app = create_app(lambda: engine, cache_dir=Path(directory), commit_cache=commit)
            with TestClient(app, raise_server_exceptions=False) as client:
                first = client.post('/synthesize', json={'text': 'Hello'})
                self.assertEqual(first.status_code, 500)
                retry = client.post('/synthesize', json={'text': 'Hello'})
                self.assertEqual(retry.status_code, 200)
                self.assertEqual(retry.headers['x-audio-cache'], 'hit')
                self.assertEqual(engine.calls, 1)
                self.assertEqual(len(commits), 2)

    def test_cache_survives_container_restart_and_isolates_voice_and_text(self):
        commits = []
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            first = CountingEngine()
            with TestClient(create_app(lambda: first, cache_dir=cache,
                                       commit_cache=lambda: commits.append(True))) as client:
                self.assertEqual(client.get('/health').status_code, 200)
                self.assertEqual(client.get('/info').json()['identity'], first.identity)
                response = client.post('/synthesize', json={'text': 'Hello'})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers['x-audio-cache'], 'miss')
                with wave.open(io.BytesIO(response.content), 'rb') as audio:
                    self.assertEqual(audio.getframerate(), 24000)
                self.assertEqual(client.post('/synthesize',
                                             json={'text': 'Hello'}).headers['x-audio-cache'], 'hit')
                self.assertEqual(first.calls, 1)
                self.assertEqual(client.post('/synthesize', json={'text': ' '}).status_code, 422)
                self.assertEqual(client.post('/synthesize',
                                             json={'text': 'x' * 4001}).status_code, 422)
                client.app.state.lock.acquire()
                try:
                    self.assertEqual(client.post('/synthesize',
                                                 json={'text': 'Hello'}).status_code, 429)
                finally:
                    client.app.state.lock.release()
            restarted = CountingEngine()
            with TestClient(create_app(lambda: restarted, cache_dir=cache)) as client:
                result = client.post('/synthesize', json={'text': 'Hello'})
                self.assertEqual(result.headers['x-audio-cache'], 'hit')
                self.assertEqual(restarted.calls, 0)
                client.post('/synthesize', json={'text': 'Different text'})
                self.assertEqual(restarted.calls, 1)
            changed = CountingEngine()
            changed.identity = 'chatterbox:other-model'
            with TestClient(create_app(lambda: changed, cache_dir=cache)) as client:
                result = client.post('/synthesize', json={'text': 'Hello'})
                self.assertEqual(result.headers['x-audio-cache'], 'miss')
                self.assertEqual(changed.calls, 1)
            self.assertEqual(len(commits), 2)

    def test_model_rollout_during_job_is_rejected_by_client(self):
        response = unittest.mock.MagicMock()
        response.__enter__.return_value = response
        response.headers = {'X-Voice-Identity': 'changed-model'}
        engine = RemoteEngine.__new__(RemoteEngine)
        engine.endpoint = 'https://example.test'
        engine.headers = {}
        engine.identity = 'original-model'
        engine.timeout = 300
        engine.voice = None
        with tempfile.TemporaryDirectory() as directory, patch('audio_book.tts.urlopen', return_value=response):
            target = Path(directory) / 'audio.wav'
            with self.assertRaisesRegex(ValueError, 'voice changed'):
                engine.synthesize('Hello', target)
            self.assertFalse(target.exists())


if __name__ == '__main__':
    unittest.main()
