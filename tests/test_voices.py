import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import wave

from fastapi.testclient import TestClient

from audio_book.neural_tts import KokoroEngine
from audio_book.server import create_app
from audio_book.tts import RemoteEngine


class VoiceEngine:
    available_voices = ('am_onyx', 'af_heart')
    default_voice = 'am_onyx'

    def __init__(self, voice='am_onyx', calls=None):
        self.identity = f'test-kokoro:{voice}'
        self.voice = voice
        self.calls = [] if calls is None else calls

    def for_voice(self, voice):
        if voice not in self.available_voices:
            raise ValueError('Unknown voice')
        return VoiceEngine(voice, self.calls)

    def synthesize(self, text, target):
        self.calls.append(self.voice)
        with wave.open(str(target), 'wb') as stream:
            stream.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
            stream.writeframes(b'\x01\x00' * 100)


class VoiceTests(unittest.TestCase):
    def test_request_voice_selects_identity_and_separate_cache(self):
        engine = VoiceEngine()
        with tempfile.TemporaryDirectory() as directory:
            with TestClient(create_app(lambda: engine, cache_dir=Path(directory))) as client:
                default = client.get('/info').json()
                heart = client.get('/info', params={'voice': 'af_heart'}).json()
                self.assertEqual(default['default_voice'], 'am_onyx')
                self.assertEqual(default['voices'], ['am_onyx', 'af_heart'])
                self.assertNotEqual(default['identity'], heart['identity'])
                for voice in ('am_onyx', 'af_heart', 'am_onyx', 'af_heart'):
                    response = client.post('/synthesize', json={'text': 'Same text.', 'voice': voice})
                    self.assertEqual(response.status_code, 200)
                    self.assertEqual(response.headers['x-voice-identity'], f'test-kokoro:{voice}')
                self.assertEqual(engine.calls, ['am_onyx', 'af_heart'])
                self.assertEqual(response.headers['x-audio-cache'], 'hit')
                default_audio = client.post('/synthesize', json={'text': 'Same text.'})
                self.assertEqual(default_audio.headers['x-audio-cache'], 'hit')
                self.assertEqual(engine.voice, 'am_onyx')
                self.assertEqual(client.get('/info?voice=missing').status_code, 400)
                self.assertEqual(client.post('/synthesize', json={'text': 'Hello', 'voice': 'missing'}).status_code, 400)
                self.assertEqual(client.post('/synthesize', json={'text': 'Hello', 'voice': ''}).status_code, 422)
                self.assertEqual(engine.calls, ['am_onyx', 'af_heart'])

    def test_remote_client_selects_voice_for_both_requests(self):
        info = io.BytesIO(json.dumps({'identity': 'test-heart'}).encode())
        audio = io.BytesIO(b'audio')
        audio.headers = {'X-Voice-Identity': 'test-heart'}
        with tempfile.TemporaryDirectory() as directory:
            with patch('audio_book.tts.urlopen', side_effect=[info, audio]) as transport:
                engine = RemoteEngine('https://example.test', voice='af_heart')
                engine.synthesize('Hello', Path(directory) / 'audio.wav')
                first, second = [call.args[0] for call in transport.call_args_list]
                self.assertEqual(first.full_url, 'https://example.test/info?voice=af_heart')
                self.assertEqual(json.loads(second.data), {'text': 'Hello', 'voice': 'af_heart'})

    def test_kokoro_shares_model_without_mutating_default_voice(self):
        kokoro = types.ModuleType('kokoro')
        kokoro.KModel = MagicMock()
        kokoro.KPipeline = MagicMock()
        torch = types.ModuleType('torch')
        torch.load = MagicMock(side_effect=lambda path, **kwargs: path.stem)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory / 'voices').mkdir()
            for name in ('config.json', 'kokoro-v1_0.pth', 'voices/am_onyx.pt', 'voices/af_heart.pt'):
                (directory / name).write_bytes(name.encode())
            with patch.dict('sys.modules', {'kokoro': kokoro, 'torch': torch}):
                engine = KokoroEngine(directory, voice='am_onyx', voices=('am_onyx', 'af_heart'))
                heart = engine.for_voice('af_heart')
                self.assertEqual(engine.voice, 'am_onyx')
                self.assertEqual(heart.voice, 'af_heart')
                self.assertIs(engine.pipeline, heart.pipeline)
                self.assertNotEqual(engine.identity, heart.identity)
                kokoro.KModel.assert_called_once()
                self.assertEqual(torch.load.call_count, 2)
                with self.assertRaisesRegex(ValueError, 'Unknown Kokoro voice'):
                    engine.for_voice('../missing')

    def test_single_voice_endpoint_rejects_voice_selection(self):
        engine = MagicMock(spec=['identity', 'synthesize'])
        engine.identity = 'single-voice'
        with TestClient(create_app(lambda: engine)) as client:
            self.assertEqual(client.get('/info').status_code, 200)
            self.assertEqual(client.get('/info?voice=af_heart').status_code, 400)
            self.assertEqual(client.post('/synthesize', json={'text': 'Hello', 'voice': 'af_heart'}).status_code, 400)
            engine.synthesize.assert_not_called()


if __name__ == '__main__':
    unittest.main()
