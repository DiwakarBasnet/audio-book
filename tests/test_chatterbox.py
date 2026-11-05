from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import MagicMock, patch
import wave

from audio_book.neural_tts import ChatterboxEngine, reference_fingerprint


def sample(path, value=1):
    with wave.open(str(path), 'wb') as audio:
        audio.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        audio.writeframes(value.to_bytes(2, 'little', signed=True) * 24000)


class ChatterboxTests(unittest.TestCase):
    def test_reference_conditions_once_and_distinguishes_voices(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            for name in ('ve.safetensors', 't3_cfg.safetensors', 's3gen.safetensors', 'tokenizer.json', 'conds.pt'):
                (directory / name).write_bytes(b'test model bytes')
            reference = directory / 'narrator.wav'
            renamed = directory / 'renamed.wav'
            sample(reference)
            renamed.write_bytes(reference.read_bytes())
            module = types.ModuleType('chatterbox.tts')
            module.ChatterboxTTS = MagicMock()
            with patch.dict('sys.modules', {'chatterbox': types.ModuleType('chatterbox'), 'chatterbox.tts': module}):
                model = module.ChatterboxTTS.from_local.return_value
                builtin = ChatterboxEngine(directory)
                model.prepare_conditionals.assert_not_called()
                cloned = ChatterboxEngine(directory, reference_audio=reference)
                model.prepare_conditionals.assert_called_once_with(str(reference), exaggeration=0.5)
                self.assertNotEqual(builtin.identity, cloned.identity)
                same = ChatterboxEngine(directory, reference_audio=renamed)
                self.assertEqual(cloned.identity, same.identity)
                sample(reference, value=2)
                changed = ChatterboxEngine(directory, reference_audio=reference)
                self.assertNotEqual(cloned.identity, changed.identity)

    def test_invalid_reference_fails_before_loading_gpu_model(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'bad.wav'
            reference.write_bytes(b'not a WAV')
            with self.assertRaisesRegex(ValueError, 'readable PCM16'):
                ChatterboxEngine(Path(directory), reference_audio=reference)
            with self.assertRaisesRegex(ValueError, 'not found'):
                reference_fingerprint(Path(directory) / 'missing.wav')

    def test_empty_and_long_references_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            reference = Path(directory) / 'sample.wav'
            for frames in (0, 31 * 8000):
                with wave.open(str(reference), 'wb') as audio:
                    audio.setparams((1, 2, 8000, 0, 'NONE', 'not compressed'))
                    audio.writeframes(b'\x01\x00' * frames)
                with self.assertRaisesRegex(ValueError, 'at most 30 seconds'):
                    reference_fingerprint(reference)


if __name__ == '__main__':
    unittest.main()
